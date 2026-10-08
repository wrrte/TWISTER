"""Shared warmup boundaries, shared replay references, and branch resume."""

import json
from pathlib import Path

import numpy as np

from retrieval_runs import SharedWarmupComplete
from training_branches import capture_rng_state, restore_rng_state


class TWISTERRetrievalRun:
    def __init__(self, model):
        self.model = model
        self.shared = model.config.retrieval_enabled == "Both"
        self.paused = [False] * model.config.num_envs
        self.resume_rng = None
        self.reset_seed = None
        self.callback_path = None
        self.plan = None

    def configure(self, callback_path, plan=None):
        self.callback_path = Path(callback_path)
        self.plan = plan
        if self.shared and plan is None:
            raise ValueError('Use configs/twister.py and override_config to launch retrieval_enabled="Both"')
        if not self.shared and plan is not None and plan.get("callback_path") == str(self.callback_path.resolve()):
            raise ValueError("Each branch must use its own callback_path, separate from shared warmup")

    @property
    def decisions(self):
        return int(self.model.action_step) // self.model.env.action_repeat

    @property
    def target_reached(self):
        return self.decisions >= self.model.retrieval.config["warmup_steps"]

    def active_environments(self):
        if self.target_reached:
            # Includes warmup=0, and a resumed run at a fresh episode boundary.
            for index, episode in enumerate(self.model.episode_history.episodes):
                if len(episode.is_firsts) == 1:
                    self.paused[index] = True
        return [not paused for paused in self.paused]

    def mark_boundaries(self, observations, active):
        if self.target_reached:
            for index, enabled in enumerate(active):
                if enabled and bool(observations.is_last[index]):
                    self.paused[index] = True

    def state_dict(self):
        return dict(version=1, num_envs=self.model.config.num_envs,
                    rng=capture_rng_state(), reset_seed=self.reset_seed)

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["num_envs"] != self.model.config.num_envs:
            raise ValueError("TWISTER retrieval-run resume requires matching version and num_envs")
        self.resume_rng = state["rng"]
        self.reset_seed = state["reset_seed"]
        # Simulator state is not saved. Every resumed environment starts fresh.
        self.paused = [False] * self.model.config.num_envs

    def _seed_environments(self):
        environments = list(self.model.env.envs)
        if self.model.env_eval is not None:
            environments.append(self.model.env_eval)
        for index, env in enumerate(environments):
            seed = (self.reset_seed + index) % (2 ** 31 - 1) + 1
            if self.model.env_type == "atari100k":
                env.seed(seed)
            else:
                # DMC's task owns its RandomState; its environment has no Gym seed API.
                while type(env).__name__ in ("ResetOnException", "TimeLimit"):
                    env = env.env
                env.env.task.random.seed(seed)

    def on_train_begin(self):
        if self.resume_rng is not None:
            restore_rng_state(self.resume_rng)
            self.resume_rng = None
            if self.reset_seed is None:
                self.reset_seed = int(np.random.randint(1, 2 ** 31 - 1))
            self._seed_environments()
            self.model.replay_buffer.streams.clear()
            self.model.ep_rewards.zero_()
            self.model.reset_episode_history()
        self.set_epoch_length()
        self.maybe_split()

    def set_epoch_length(self):
        length = self.model.config.epoch_length
        remaining = length - int(self.model.model_step) % length
        self.model.replay_buffer.epoch_length = remaining

    def maybe_split(self):
        if not self.shared or any(self.active_environments()):
            return
        if self.callback_path is None or self.plan is None:
            raise ValueError("Shared warmup must be configured by the training entry point")
        model = self.model
        if int(model.model_step) >= model.config.epochs * model.config.epoch_length:
            raise RuntimeError("No training budget remains after shared warmup; no branches were launched")
        directory = self.callback_path / "shared_warmup"
        directory.mkdir(parents=True, exist_ok=True)
        self.reset_seed = int(np.random.randint(1, 2 ** 31 - 1))
        checkpoint = f"checkpoints_epoch_{int(model.model_step) // model.config.epoch_length}_step_{int(model.model_step)}.ckpt"
        model.save(str(directory / checkpoint))
        plan = dict(self.plan, callback_path=str(self.callback_path.resolve()),
                    checkpoint=checkpoint, warmup_decisions=self.decisions,
                    warmup_model_steps=int(model.model_step))
        path = directory / "run.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(path)
        print(f"Shared warmup finished after {self.decisions} decisions and "
              f"{int(model.model_step)} model updates: {directory}", flush=True)
        raise SharedWarmupComplete(directory)
