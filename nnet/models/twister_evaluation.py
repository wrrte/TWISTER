"""Evaluate independent episodes with one batched model and threaded Atari steps."""

from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from nnet.envs.wrappers import ResetOnException, TimeLimit
from seeding import seed_environment


def _close_environment(env):
    # Atari/DMC adapters and wrappers do not all expose close directly.
    while env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            close()
            return
        env = getattr(env, "env", None)


def _play_batch(model, environments, executor):
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
        with tqdm(total=num_episodes, dynamic_ncols=True, disable=verbose_progress_bar == 0) as progress:
            for start in range(0, num_episodes, num_envs):
                width = min(num_envs, num_episodes - start)
                for outputs in _play_batch(model, environments[:width], executor):
                    completed += 1
                    for key, value in outputs.items():
                        totals[key] += value
                        model.infos["ep_" + key] = value.item()
                    if verbose_progress_bar >= 2:
                        model.display_step({}, totals, model.infos, progress, completed)
                    progress.update(1)
        return {}, {key: value / completed for key, value in totals.items()}
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in environments[1:]:
            _close_environment(env)
