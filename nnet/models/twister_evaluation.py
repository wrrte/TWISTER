"""Evaluate episodes with batched inference and process-based Atari emulators."""

from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import numpy as np
import torch
from tqdm import tqdm

from nnet.envs.wrappers import ResetOnException, TimeLimit
from seeding import seed_environment, seed_everything


def evaluate_final(model, verbose_progress_bar=1):
    """Evaluate saved final weights on fresh environments without loading replay."""
    from nnet.datasets import VoidDataset

    dataset = VoidDataset(model.config.final_eval_episodes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    previous_env = model.env_eval
    previous_num_envs = model.config.num_eval_envs
    previous_mode = model.training
    model.env_eval = None
    try:
        model.config.num_eval_envs = model.config.final_eval_num_envs
        seed_everything(model.config.seed)
        model.env_eval = ResetOnException(TimeLimit(
            model.config.env_class(**dict(model.config.env_params, **model.config.eval_env_params)),
            time_limit=model.config.time_limit_eval))
        if model.config.seed is not None:
            seed_environment(model.env_eval, (model.config.seed + model.config.num_envs) % (2 ** 31 - 1),
                             model.env_type)
        return model.evaluate(loader, verbose_progress_bar=verbose_progress_bar)
    finally:
        if model.env_eval is not None:
            _close_environment(model.env_eval)
        model.env_eval = previous_env
        model.config.num_eval_envs = previous_num_envs
        model.train(previous_mode)


class _EvaluationProgress(tqdm):
    """Keep episode counts and the bar visible when the terminal is resized."""
    @staticmethod
    def status_printer(file):
        if getattr(file, "isatty", lambda: False)():
            def print_status(text):
                # Padding to the previous width would wrap after a shrink.
                file.write("\r\033[K" + str(text))
                file.flush()
            return print_status
        return tqdm.status_printer(file)

    def __init__(self, total, verbose):
        self._metric_options = [""]
        self._next_refresh = monotonic() + 1
        super().__init__(
            total=total, dynamic_ncols=True, ascii=True, leave=True,
            disable=verbose == 0, mininterval=0.5,
            bar_format="{n_fmt}/{total_fmt} {percentage:3.0f}% |{bar:12}| [{elapsed}<{remaining}]{postfix}")

    @property
    def format_dict(self):
        data = super().format_dict
        columns = data["ncols"]
        if columns is not None:
            # Keep a spare terminal column so the cursor never wraps the bar.
            columns = data["ncols"] = max(1, columns - 1)
        if columns is not None and columns < 46:
            data["bar_format"] = "{n_fmt}/{total_fmt} {percentage:3.0f}% |{bar:8}|{postfix}"
        # Drop whole optional fields rather than clipping the progress bar or
        # leaving half a metric at the right edge. Recheck width on each redraw.
        for metrics in self._metric_options:
            candidate = dict(data, ncols=None, postfix=metrics)
            if columns is None or len(self.format_meter(**candidate)) <= columns:
                data["postfix"] = metrics
                break
        else:
            data["postfix"] = ""
        return data

    def show_metrics(self, totals, infos, completed):
        score = "score={:g}".format(float(totals["score"]) / completed)
        steps = "steps={:g}".format(float(totals["steps"]) / completed)
        last_score = "ep_score={:g}".format(infos["ep_score"])
        last_steps = "ep_steps={:g}".format(infos["ep_steps"])
        self._metric_options = [
            " ".join((score, steps, last_score, last_steps)),
            " ".join((score, last_score)), score, ""]

    def heartbeat(self):
        # Long Atari episodes can run for minutes between completed episodes.
        # Redraw periodically so resizing also works during that interval.
        now = monotonic()
        if not self.disable and now >= self._next_refresh:
            self.refresh()
            self._next_refresh = now + 1


def _close_environment(env):
    # Atari/DMC adapters and wrappers do not all expose close directly.
    while env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            close()
            return
        env = getattr(env, "env", None)


def _play_batch(model, environments, executor, progress=None):
    observations = [env.reset() for env in environments]
    batch_size = len(environments)
    previous_latent = model.transfer_to_device(model.rssm.initial(
        batch_size=batch_size, seq_length=1,
        dtype=observations[0].reward.dtype, detach_learned=True))
    previous_action = torch.zeros(
        batch_size, model.env.num_actions,
        dtype=observations[0].reward.dtype, device=model.device)
    scores = torch.zeros(batch_size, dtype=observations[0].reward.dtype)
    frames = torch.zeros(batch_size, dtype=torch.int64)

    while environments:
        state = model.transfer_to_device(torch.stack([obs.state for obs in observations]))
        with torch.no_grad():
            latent = model.encoder_network(model.preprocess_inputs(state, time_stacked=False))
            latent = {key: value.unsqueeze(1) for key, value in latent.items()}
            latent, _ = model.rssm(
                states=latent, prev_states=previous_latent,
                prev_actions=previous_action.unsqueeze(1),
                # Fresh batches have no mid-episode reset. Keep this constant on
                # CPU, like play(), to avoid a CUDA sync in TSSM's any() check.
                is_firsts=torch.zeros(len(environments), 1))
            action = model.policy_network(model.rssm.get_feat(latent).squeeze(1)).mode()
        latent["hidden"] = model.rssm.slice_hidden(latent["hidden"])

        # One device transfer for all actions, followed by independent CPU steps.
        env_actions = (action.argmax(-1) if model.config.policy_discrete else action).cpu()
        if executor is None:
            observations = [env.step(env_action) for env, env_action in zip(environments, env_actions)]
        else:
            futures = [executor.submit(env.step, env_action)
                       for env, env_action in zip(environments, env_actions)]
            observations = [future.result() for future in futures]
        scores += torch.stack([obs.reward for obs in observations])
        frames += torch.tensor([env.action_repeat for env in environments])
        if progress is not None:
            progress.heartbeat()

        keep = []
        for index, obs in enumerate(observations):
            if bool(obs.done) or frames[index] >= model.config.time_limit_eval:
                yield {"score": scores[index].clone(), "steps": frames[index].clone()}
            else:
                keep.append(index)
        if not keep:
            break

        if len(keep) < len(environments):
            # Remove every finished lane from observations AND recurrent KV caches.
            # A new batch always starts from a fresh initial latent state.
            indices = torch.tensor(keep, device=model.device)
            latent = {
                key: ([(k.index_select(0, indices), v.index_select(0, indices)) for k, v in value]
                      if key == "hidden" else value.index_select(0, indices))
                for key, value in latent.items()
            }
            action = action.index_select(0, indices)
            environments = [environments[index] for index in keep]
            observations = [observations[index] for index in keep]
            scores, frames = scores[keep], frames[keep]
        previous_latent, previous_action = latent, action


def evaluate_parallel(model, num_episodes, verbose_progress_bar=1):
    if num_episodes <= 0:
        return {}, {}

    if model.env_type == "atari100k" and model.config.eval_env_backend == "process":
        return evaluate_vector(model, num_episodes, verbose_progress_bar)

    num_envs = min(model.config.num_eval_envs, num_episodes)
    environments = [model.env_eval]
    executor = None
    totals = {"score": torch.tensor(0., dtype=torch.float64),
              "steps": torch.tensor(0., dtype=torch.float64)}
    completed = 0
    try:
        for index in range(1, num_envs):
            env = ResetOnException(TimeLimit(
                model.config.env_class(**dict(model.config.env_params, **model.config.eval_env_params)),
                time_limit=model.config.time_limit_eval))
            environments.append(env)
            if model.config.seed is not None:
                seed_environment(env, (model.config.seed + model.config.num_envs + index) % (2 ** 31 - 1),
                                 model.env_type)

        # DMC rendering contexts stay on the main thread; its model inference is
        # still batched. Atari emulator steps use independent worker threads.
        if model.env_type == "atari100k":
            executor = ThreadPoolExecutor(max_workers=num_envs)
        print("Evaluation: {} episodes, up to {} environments".format(num_episodes, num_envs))
        with _EvaluationProgress(num_episodes, verbose_progress_bar) as progress:
            for start in range(0, num_episodes, num_envs):
                width = min(num_envs, num_episodes - start)
                for outputs in _play_batch(model, environments[:width], executor, progress=progress):
                    completed += 1
                    for key, value in outputs.items():
                        totals[key] += value
                        model.infos["ep_" + key] = value.item()
                    if verbose_progress_bar >= 2:
                        progress.show_metrics(totals, model.infos, completed)
                    progress.update(1)
        return {}, {key: value / completed for key, value in totals.items()}
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in environments[1:]:
            _close_environment(env)


def _play_vector(model, vector, num_episodes, progress=None):
    """Refill completed lanes without sharing recurrent state across episodes."""
    state = vector.reset()
    width = vector.num_envs
    initial = model.transfer_to_device(model.rssm.initial(
        batch_size=width, seq_length=1, dtype=torch.float32, detach_learned=True))
    previous_latent = {key: value.clone() if key != "hidden" else None
                       for key, value in initial.items()}
    previous_action = torch.zeros(width, model.env.num_actions, device=model.device)
    hidden_lengths = torch.zeros(width, dtype=torch.long, device=model.device)
    is_firsts = torch.zeros(width, 1)  # TSSM's any() check stays on CPU.
    scores = np.zeros(width, dtype=np.float64)
    frames = np.zeros(width, dtype=np.int64)
    action_repeat = model.env_eval.action_repeat
    completed = 0

    while completed < num_episodes:
        # No worker writes the shared observation array until step_async below.
        state = model.transfer_to_device(torch.from_numpy(state))
        with torch.no_grad():
            latent = model.encoder_network(model.preprocess_inputs(state, time_stacked=False))
            latent = {key: value.unsqueeze(1) for key, value in latent.items()}
            latent, _ = model.rssm(
                states=latent, prev_states=previous_latent,
                prev_actions=previous_action.unsqueeze(1), is_firsts=is_firsts,
                hidden_lengths=hidden_lengths)
            action = model.policy_network(model.rssm.get_feat(latent).squeeze(1)).mode()
        latent["hidden"] = model.rssm.slice_hidden(latent["hidden"])
        env_actions = action.argmax(-1).cpu().numpy()
        vector.step_async(env_actions)
        # Queue this GPU bookkeeping while the emulators advance on the CPU.
        hidden_lengths = (hidden_lengths + 1).clamp(max=model.rssm.att_context_left)
        state, rewards, dones, infos = vector.step_wait()
        scores += rewards
        frames += action_repeat
        if progress is not None:
            progress.heartbeat()

        for index in np.flatnonzero(dones):
            yield {"score": torch.tensor(scores[index]), "steps": torch.tensor(frames[index])}
            completed += 1
            if completed == num_episodes:
                return
            scores[index], frames[index] = 0., 0

        # AsyncVectorEnv has already reset every completed emulator. Replace
        # the lane's latent/action and mask its stale KV history for ALL steps
        # of the next episode, until a full context has been collected.
        reset = np.flatnonzero(dones | np.array([info["error"] for info in infos]))
        if len(reset):
            indices = torch.as_tensor(reset, device=model.device)
            for key, value in latent.items():
                if key != "hidden":
                    value.index_copy_(0, indices, initial[key].index_select(0, indices))
            action.index_fill_(0, indices, 0.)
            hidden_lengths.index_fill_(0, indices, 0)
        previous_latent, previous_action = latent, action


def evaluate_vector(model, num_episodes, verbose_progress_bar=1):
    from nnet.envs.wrappers.atari_eval_env import create_eval_vector

    width = min(model.config.num_eval_envs, num_episodes)
    vector = create_eval_vector(model, width)
    totals = {"score": torch.tensor(0., dtype=torch.float64),
              "steps": torch.tensor(0., dtype=torch.float64)}
    completed = 0
    print("Evaluation: {} episodes, {} process environments (shared memory, continuous refill)".format(
        num_episodes, width))
    try:
        with _EvaluationProgress(num_episodes, verbose_progress_bar) as progress:
            for outputs in _play_vector(model, vector, num_episodes, progress=progress):
                completed += 1
                for key, value in outputs.items():
                    totals[key] += value
                    model.infos["ep_" + key] = value.item()
                if verbose_progress_bar >= 2:
                    progress.show_metrics(totals, model.infos, completed)
                progress.update(1)
        return {}, {key: value / completed for key, value in totals.items()}
    finally:
        vector.close(timeout=5)
