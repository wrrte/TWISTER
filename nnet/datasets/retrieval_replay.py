"""A sparse, non-wrapping retrieval view over TWISTER's overlapping windows."""

import collections
import random

import torch


class _FieldView:
    def __init__(self, replay, field):
        self.replay = replay
        self.field = field

    def __getitem__(self, index):
        pointer, env_slice = index
        env = env_slice.start
        value = self.replay.steps[pointer, env][self.field]
        if self.field == 0:
            value = value.permute(1, 2, 0)
        return value.unsqueeze(0)


class RetrievalReplay:
    # RetrievalContextManager uses modulo arithmetic. These IDs never wrap.
    pointer_limit = 1 << 60
    store_on_gpu = True  # The shared API uses this flag for tensor (vs numpy) storage.

    def __init__(self, replay, num_envs):
        self.replay = replay
        self.num_envs = num_envs
        self.max_length = self.pointer_limit * num_envs
        self.steps = {}
        self.references = collections.Counter()
        self.windows = {}
        self.next_base = [0] * num_envs
        self.pending = set()
        self.evicted = set()
        self.obs_buffer = _FieldView(self, 0)
        self.action_buffer = _FieldView(self, 1)

    @property
    def length(self):
        return len(self.steps)

    def add_window(self, traj_id, env, traj, base=None):
        if base is None:
            base = self.next_base[env]
            self.next_base[env] += 1
        self.windows[traj_id] = (base, env)
        for t in range(self.replay.sample_length):
            key = (base + t, env)
            if key not in self.steps:
                self.steps[key] = [field[t] for field in traj]
                self.pending.add(key)
            self.references[key] += 1

    def remove_window(self, traj_id):
        base, env = self.windows.pop(traj_id)
        for t in range(self.replay.sample_length):
            key = (base + t, env)
            self.references[key] -= 1
            if not self.references[key]:
                del self.references[key]
                del self.steps[key]
                self.pending.discard(key)
                self.evicted.add(key)

    def is_valid_context(self, pointer, env, length):
        if length < 1 or not 0 <= env < self.num_envs:
            return False
        keys = [(p, env) for p in range(pointer - length + 1, pointer + 1)]
        if any(key not in self.steps for key in keys):
            return False
        # done is arrival-aligned; is_first also catches time limits and errors.
        return (all(self.steps[key][3].item() <= 0.5 for key in keys[:-1])
                and all(self.steps[key][4].item() <= 0.5 for key in keys[1:]))

    def context_field(self, indices, length, field, device):
        return torch.stack([
            torch.stack([self.steps[p - offset, env][field]
                         for offset in range(length - 1, -1, -1)])
            for p, env in indices
        ]).to(device)

    def state_dict(self):
        return {"windows": dict(self.windows), "next_base": list(self.next_base),
                "num_envs": self.num_envs}

    def restore(self, state=None):
        self.steps.clear()
        self.references.clear()
        self.windows.clear()
        self.pending.clear()
        self.evicted.clear()
        if state is not None:
            if state["num_envs"] != self.num_envs:
                raise ValueError("Retrieval replay resume requires the same num_envs")
            self.next_base = list(state["next_base"])
            for traj_id, traj in self.replay.ram_buffer.items():
                base, env = state["windows"][traj_id]
                self.add_window(traj_id, env, traj, base=base)
        else:
            # Legacy checkpoints have no stream IDs. Treat each stored window as
            # an independent segment rather than inventing cross-window history.
            self.next_base = [0] * self.num_envs
            for traj_id, traj in self.replay.ram_buffer.items():
                base = self.next_base[0]
                self.add_window(traj_id, 0, traj, base=base)
                self.next_base[0] += self.replay.sample_length + 1
        # Simulator state and unfinished streams are not saved by TWISTER.
        self.next_base = [base + self.replay.sample_length + 1 for base in self.next_base]

    def _encode(self, indices, world_model, device, sample_mode):
        obs = torch.stack([self.steps[key][0] for key in indices]).to(device)
        return world_model.encode_obs(obs.unsqueeze(1).float() / 255.0,
                                      sample_mode=sample_mode).squeeze(1)

    @torch.no_grad()
    def update_hashes(self, manager, world_model, chunk_size):
        # Remove evicted replay entries without changing IDs of surviving frames.
        for key in self.evicted:
            if key in manager.index_to_bucket:
                bucket_key = manager.index_to_bucket.pop(key)
                bucket = manager.hash_memory[bucket_key]
                bucket.remove(key)
                if not bucket:
                    del manager.hash_memory[bucket_key]
        self.evicted.clear()
        manager.active_anchors = collections.deque(
            (anchor, key) for anchor, key in manager.active_anchors
            if anchor in self.steps)
        indices = sorted(self.pending)
        self._hash_indices(indices, manager, world_model, chunk_size)
        self.pending.clear()

    def _hash_indices(self, indices, manager, world_model, chunk_size):
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start:start + chunk_size]
            latent = self._encode(chunk, world_model, manager.device, manager.hash_sample_mode)
            for (pointer, env), key in zip(chunk, manager._hash_keys(latent)):
                manager._insert_into_bucket(pointer, env, key)

    @torch.no_grad()
    def rebuild_hash_buckets(self, manager, world_model, chunk_size):
        manager.hash_memory.clear()
        manager.index_to_bucket.clear()
        # Anchor keys belong to the previous projection.
        manager.active_anchors.clear()
        indices = sorted(self.steps)
        if manager.use_pca and len(indices) > manager.hash_bits:
            selected = random.sample(indices, min(len(indices), manager.max_pca_samples))
            latents = [self._encode(selected[start:start + chunk_size], world_model,
                                    manager.device, manager.hash_sample_mode).float()
                       for start in range(0, len(selected), chunk_size)]
            latents = torch.cat(latents)
            mean = latents.mean(0, keepdim=True)
            _, _, projection = torch.pca_lowrank(latents - mean, q=manager.hash_bits, center=False)
            if torch.isfinite(projection).all():
                manager.hash_proj = projection
                manager.hash_mean = mean
        self._hash_indices(indices, manager, world_model, chunk_size)
        self.pending.clear()
        self.evicted.clear()
