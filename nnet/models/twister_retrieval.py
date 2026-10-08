"""Retrieval-only training state; never constructed on the baseline path."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))
from retrieval import RetrievalContextManager


class TWISTERRetrieval:
    defaults = dict(
        context_length=8, warmup_steps=5000, max_anchors=10, max_contexts=256,
        multiplier=5, target=5, threshold=1.0, trigger_mode="absolute",
        anchor_offset=-2, anchor_weight=0.5, z_score_threshold=2.0, ema_alpha=0.01,
        hash_bits=12, hash_sample_mode="probs", max_bucket_size=512,
        use_pca=True, max_pca_samples=10000, chunk_size=256,
        global_rebuild_enable=True, global_rebuild_threshold=0.2,
        global_rebuild_cooldown=2000, batch_size_reduction="none",
    )

    def __init__(self, model):
        self.model = model
        unknown = set(model.config.retrieval) - self.defaults.keys()
        if unknown:
            raise ValueError("Unknown TWISTER retrieval settings: {}".format(sorted(unknown)))
        self.config = dict(self.defaults, **model.config.retrieval, enable=True)
        cfg = self.config
        for key in ("context_length", "warmup_steps", "max_anchors", "max_contexts",
                    "multiplier", "target", "anchor_offset", "hash_bits",
                    "max_bucket_size", "max_pca_samples", "chunk_size", "global_rebuild_cooldown"):
            if type(cfg[key]) is not int:
                raise ValueError("retrieval.{} must be an integer".format(key))
        if not 1 <= cfg["context_length"] < model.config.L - 1:
            raise ValueError("retrieval.context_length must be in [1, L-2]")
        if cfg["warmup_steps"] < 0:
            raise ValueError("TWISTER retrieval.warmup_steps must be nonnegative (environment decisions)")
        for key in ("max_anchors", "multiplier", "target", "max_bucket_size", "chunk_size"):
            if cfg[key] < 1:
                raise ValueError("retrieval.{} must be positive".format(key))
        latent_dim = model.config.model_stoch_size * model.config.model_discrete
        if not 1 <= cfg["hash_bits"] <= min(62, latent_dim):
            raise ValueError("retrieval.hash_bits must be in [1, min(62, latent_dim)]")
        if cfg["max_pca_samples"] <= cfg["hash_bits"]:
            raise ValueError("retrieval.max_pca_samples must exceed hash_bits")
        if cfg["max_contexts"] < 0 or cfg["global_rebuild_cooldown"] < 0:
            raise ValueError("retrieval context limit and rebuild cooldown must be nonnegative")
        if not 0 <= cfg["anchor_weight"] <= 1 or not 0 < cfg["ema_alpha"] <= 1:
            raise ValueError("Invalid retrieval anchor_weight or ema_alpha")
        if cfg["trigger_mode"] not in ("absolute", "z_score"):
            raise ValueError("retrieval.trigger_mode must be absolute or z_score")
        if cfg["hash_sample_mode"] not in ("probs", "mode", "sample"):
            raise ValueError("retrieval.hash_sample_mode must be probs, mode, or sample")
        if cfg["batch_size_reduction"] not in ("none", "retrieved", "anchors", "half"):
            raise ValueError("retrieval.batch_size_reduction must be none, retrieved, anchors, or half")
        self.manager = None
        self.hash_built = False
        self.last_rebuild_step = -cfg["global_rebuild_cooldown"]
        self.sample_weights = None
        self.initial_dones = None

    def ensure_manager(self):
        device = self.model.device
        if self.manager is None:
            self.manager = RetrievalContextManager(
                self.model.config.num_envs, self.config,
                self.model.config.model_stoch_size * self.model.config.model_discrete,
                device=device)
        elif torch.device(self.manager.device) != torch.device(device):
            state = self.manager.state_dict()
            self.manager.device = device
            self.manager.hash_bit_values = self.manager.hash_bit_values.to(device)
            self.manager.load_state_dict(state)
        return self.manager

    @contextmanager
    def evaluation(self):
        modules = (self.model.encoder_network, self.model.rssm, self.model.value_network)
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)

    def encode_obs(self, obs, sample_mode="probs"):
        # Shared retrieval supplies floats in [0, 1]; TWISTER expects [-.5, .5].
        encoder = self.model.encoder_network
        obs = obs.to(self.model.device)
        embedding = encoder.forward_cnn(obs - 0.5)
        logits = encoder.representation_network(embedding).reshape(
            *embedding.shape[:-1], encoder.stoch_size, encoder.discrete)
        dist = encoder.get_dist({"logits": logits}).base_dist
        if sample_mode == "probs":
            stoch = dist.probs
        elif sample_mode == "mode":
            stoch = torch.nn.functional.one_hot(dist.probs.argmax(-1), encoder.discrete).to(logits.dtype)
        elif sample_mode == "sample":
            stoch = dist.sample()
        else:
            raise ValueError("Unknown retrieval hash_sample_mode: {}".format(sample_mode))
        return stoch.flatten(-2, -1)

    def state_dict(self):
        return {"manager": self.manager.state_dict() if self.manager is not None else None,
                "last_rebuild_step": self.last_rebuild_step}

    def load_state_dict(self, state):
        if state and state.get("manager") is not None:
            manager = self.ensure_manager()
            if state["manager"]["hash_proj"].shape != manager.hash_proj.shape:
                raise ValueError("Retrieval checkpoint hash_bits/latent dimension differs from configuration")
            manager.load_state_dict(state["manager"])
        self.last_rebuild_step = (state or {}).get("last_rebuild_step", self.last_rebuild_step)
        # Reconcile replay contents, discarded streams, and legacy checkpoints.
        self.hash_built = False

    def _trigger(self, inputs, metadata, is_warmup):
        model, manager = self.model, self.manager
        _, _, rewards, dones, firsts, _ = inputs
        replay = model.replay_buffer.retrieval_view()
        batch, length = rewards.shape
        feats = model.rssm.get_feat(model.detached_posts).reshape(batch, length, -1)
        values = model.value_network(feats).mode().squeeze(-1)
        bases, envs = metadata.detach().cpu().numpy().T
        firsts_cpu, dones_cpu = firsts.detach().cpu(), dones.detach().cpu()
        valid = ((firsts_cpu[:, 1:] <= 0.5) & (firsts_cpu[:, :-1] <= 0.5)
                 & (dones_cpu[:, :-1] <= 0.5))
        segments = firsts_cpu.cumsum(1)
        for b in range(batch):
            for t in range(length - 1):
                anchor_t = t + manager.anchor_offset
                same_episode = (0 <= anchor_t < length
                                and segments[b, anchor_t] == segments[b, t])
                if not same_episode or not replay.is_valid_context(
                        int(bases[b]) + anchor_t, int(envs[b]), manager.context_length):
                    valid[b, t] = False
        # TWISTER stores (o_t, a_{t-1}, r_t, done_t). Shared TD uses r for
        # the outgoing edge: delta_t = r_{t+1} + gamma*(1-done_{t+1})*V_{t+1} - V_t.
        outgoing_rewards = torch.cat((rewards[:, 1:], rewards[:, -1:]), dim=1)
        outgoing_dones = torch.cat((dones[:, 1:], dones[:, -1:]), dim=1)
        return manager.add_batch_transitions(
            values, outgoing_rewards, outgoing_dones, model.config.gamma,
            bases, envs, replay.pointer_limit, skip_len=manager.context_length,
            is_warmup=is_warmup, transition_mask=valid)

    def _select_original_starts(self, dones, retrieved_count, anchors):
        """Subsample complete imagination states, including their cache and masks."""
        mode = self.config["batch_size_reduction"]
        reduction = {"none": 0, "retrieved": retrieved_count, "anchors": anchors,
                     "half": (retrieved_count + anchors) // 2}[mode]
        flat_dones = dones.flatten()
        count = flat_dones.numel()
        keep = max(0, count - reduction)
        if keep == count:
            return flat_dones
        model = self.model
        selected = (torch.randperm(count, device=model.device)[:keep].sort().values
                    if keep else torch.empty(0, dtype=torch.long, device=model.device))
        for key, value in model.detached_posts.items():
            model.detached_posts[key] = (
                [tuple(tensor.index_select(0, selected) for tensor in block) for block in value]
                if key == "hidden" else value.index_select(0, selected))
        model.detached_is_firsts = model.detached_is_firsts.index_select(0, selected)
        model.detached_is_firsts_hidden = model.detached_is_firsts_hidden.index_select(0, selected)
        return flat_dones.index_select(0, selected)

    def _append_contexts(self, obs, actions, indices, weights, dones, anchors=0):
        model = self.model
        replay = model.replay_buffer.retrieval_view()
        length, context = self.config["context_length"], model.config.att_context_left
        firsts = replay.context_field(indices, length, 4, model.device)
        ret_dones = replay.context_field(indices, 1, 3, model.device).flatten()
        latent = model.encoder_network(obs.to(model.device) - 0.5)
        posts, _ = model.rssm.observe(latent, actions.to(model.device).clone(), firsts)
        original_dones = self._select_original_starts(dones, len(indices), anchors)
        # Match the final time slice of WorldModel.forward's cache/mask flattening.
        for key, value in model.detached_posts.items():
            if key == "hidden":
                hidden = []
                for old_block, new_block in zip(value, posts[key]):
                    tensors = []
                    for old, new in zip(old_block, new_block):
                        new = new[:, -context:]
                        new = torch.cat((new.new_zeros(new.shape[0], context - new.shape[1], new.shape[2]), new), 1)
                        tensors.append(torch.cat((old, new.detach().to(old)), 0))
                    hidden.append(tuple(tensors))
                model.detached_posts[key] = hidden
            else:
                model.detached_posts[key] = torch.cat((value, posts[key][:, -1:].detach().to(value)), 0)
        hidden_firsts = torch.cat((
            firsts.new_zeros(firsts.shape[0], max(0, context - length)),
            firsts.new_ones(firsts.shape[0], 1),
            firsts[:, max(0, length - context):length - 1]), 1)
        model.detached_is_firsts = torch.cat((model.detached_is_firsts, firsts[:, -1:]), 0)
        model.detached_is_firsts_hidden = torch.cat((model.detached_is_firsts_hidden, hidden_firsts), 0)
        self.initial_dones = torch.cat((original_dones, ret_dones), 0).detach()
        self.sample_weights = torch.cat((
            torch.ones(original_dones.numel(), device=model.device),
            torch.tensor(weights, dtype=torch.float32, device=model.device)), 0)
        model.add_info("retrieval_original_starts", original_dones.numel())
        model.add_info("retrieval_imagination_starts", self.initial_dones.numel())

    def prepare(self, inputs, metadata, precision):
        self.sample_weights = None
        self.initial_dones = None
        model, cfg = self.model, self.config
        manager = self.ensure_manager()
        replay = model.replay_buffer.retrieval_view()
        # action_step counts emulator frames; warmup/cooldown count decisions
        # summed over environments, matching STORM/Drama's sample-step units.
        step = int(model.action_step.item()) // model.env.action_repeat
        warmup = model.config.retrieval_enabled == "Both" or step < cfg["warmup_steps"]
        autocast = (torch.autocast(device_type="cuda", dtype=precision)
                    if torch.device(model.device).type == "cuda" and precision != torch.float32
                    else nullcontext())
        with self.evaluation(), autocast:
            if not warmup and not self.hash_built:
                manager.rebuild_all_hash_buckets(model.replay_buffer, model, cfg["chunk_size"])
                self.hash_built = True
                self.last_rebuild_step = step
            else:
                replay.update_hashes(manager, model, cfg["chunk_size"])
            triggers = self._trigger(inputs, metadata, warmup)
            model.add_info("retrieval_triggers", triggers)
            model.add_info("retrieval_warmup", int(warmup))
            model.add_info("retrieval_contexts", 0)
            model.add_info("retrieval_original_starts", inputs[3].numel())
            model.add_info("retrieval_imagination_starts", inputs[3].numel())
            if warmup:
                return
            if cfg["max_contexts"] == 0:
                manager.active_anchors.clear()
                return
            queued = len(manager.active_anchors)
            obs, actions, candidates, hit_rate, weights, anchors, indices = manager.retrieve_contexts(
                model.replay_buffer, model, max_anchors=cfg["max_anchors"],
                multiplier=cfg["multiplier"], target=cfg["target"],
                max_contexts=cfg["max_contexts"], return_indices=True)
            if obs is not None:
                self._append_contexts(obs, actions, indices, weights, inputs[3], anchors=anchors)
            model.add_info("retrieval_contexts", len(indices))
            model.add_info("retrieval_anchors", anchors)
            model.add_info("retrieval_candidates", candidates)
            model.add_info("retrieval_hit_rate", hit_rate)
            model.add_info("retrieval_queue", len(manager.active_anchors))
            model.add_info("retrieval_weight_sum", self.sample_weights.sum().item()
                           if self.sample_weights is not None else inputs[3].numel())
            if (queued and indices and cfg["global_rebuild_enable"]
                    and hit_rate < cfg["global_rebuild_threshold"]
                    and step - self.last_rebuild_step >= cfg["global_rebuild_cooldown"]):
                manager.rebuild_all_hash_buckets(model.replay_buffer, model, cfg["chunk_size"])
                self.last_rebuild_step = step

    def loss_mean(self, loss):
        if self.sample_weights is None:
            return loss.mean()
        weights = self.sample_weights.to(loss.device)
        # Preserve TWISTER's time mean and continuation weights. Each retrieved
        # anchor group has total mass one, independent of its number of matches.
        return (loss.float().mean(dim=1) * weights).sum() / weights.sum()
