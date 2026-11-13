#!/usr/bin/env python3
"""Round-robin full FLASH search with a paper-score gate and per-game top four."""

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from configuration import load_run_config, validate_seed
from run_results import write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_STATE = ROOT / "results" / "reproduction" / "state.json"
KEEP = 4
PRIORITY_PAIRS = 3
HNS_MARGIN = 0.1
START_SEED = 3090
SEED_STEP = 10
MARKER = ".reproduction_trial.json"


def literal_assignment(path, name):
    # Read the existing reference table without importing the Excel exporter.
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def references():
    games = literal_assignment(ROOT / "results/append_missing_atari_jobs.py", "GAMES")
    table = literal_assignment(ROOT / "results/convert_csv_to_excel.py", "REFERENCE_SCORES")
    return {game: list(table[game.replace("_", "")]) for game in games}


def compare_score(score, reference):
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Full FLASH final score must be finite")
    random, human, paper = reference
    if human <= random:
        raise ValueError("Human score must exceed random score")
    scale = human - random
    cutoff = paper - HNS_MARGIN * scale
    return dict(score=score, hns=(score - random) / scale,
                paper_hns=(paper - random) / scale, delta_hns=(score - paper) / scale,
                minimum_score=cutoff, passes=score >= cutoff)


def retained_trials(state, game):
    return sorted((trial for trial in state["trials"] if trial["game"] == game
                   and not trial.get("imported_from")
                   and trial["retained"] and trial["full_flash"] is not None),
                  key=lambda trial: (-trial["full_flash"]["score"], trial["round"]))


def successful_trials(state, game):
    # Numeric history survives pruning and server handoffs. Every completed
    # ON/OFF pair counts, including pairs accepted by the mean-score gate.
    return [trial for trial in game_trials(state, game)
            if trial["full_flash"] is not None and trial["baseline"] is not None
            and trial["status"] == "complete"]


def local_successful_trials(state, game):
    # Released source copies belong to the handoff receiver's count. Imported
    # records in its active trial table count there, even without local files.
    return [trial for trial in state["trials"] if trial["game"] == game
            and trial["full_flash"] is not None and trial["baseline"] is not None
            and trial["status"] == "complete"]


def successful_pair_counts(state):
    return {game: progress["successful_pairs"] for game, progress in game_progress(state).items()}


def pending_full_flash(state, game):
    """Only live/retryable OFF branches can reserve one of the four pairs."""
    result = {}
    for trial in state["trials"]:
        if (trial["game"] != game or trial["status"] in {"complete", "cancelled"}
                or trial["full_flash"] is None or trial["baseline"] is not None
                or (trial.get("imported_from") and not trial.get("baseline_recovery"))):
            continue
        accepted = (trial["full_flash"]["passes"] or trial.get("baseline_gate", {}).get("accepted")
                    or baseline_gate(state, trial)["accepted"])
        if accepted:
            result[trial["seed"]] = trial["full_flash"]["score"]
    return result


def game_progress(state):
    progress = {}
    for game in state["config"]["games"]:
        pairs = successful_trials(state, game)
        progress[game] = dict(successful_pairs=len(pairs),
                              successful_pair_seeds={trial["seed"] for trial in pairs},
                              pending_full_flash=pending_full_flash(state, game))
    if state["config"].get("server"):
        from reproduction_cluster import shared_game_progress
        progress.update(shared_game_progress(state))
    return progress


def selected_pending_seeds(progress):
    needed = max(0, KEEP - progress["successful_pairs"])
    pending = {seed: score for seed, score in progress["pending_full_flash"].items()
               if seed not in progress["successful_pair_seeds"]}
    if len(pending) < needed:
        return None
    # Completed pairs always count. Fill remaining slots by full FLASH score;
    # a seed's baseline is protected while that seed belongs to this selection.
    return set(sorted(pending, key=lambda seed: (-pending[seed], seed))[:needed])


def cancellation_reason(trial, progress):
    selected = selected_pending_seeds(progress)
    if selected is not None and trial["seed"] not in selected:
        return "Four completed pairs or qualifying full FLASH results are already secured"
    return None


def game_trials(state, game):
    trials = [trial for trial in state["trials"] if trial["game"] == game]
    history = (state.get("imported_history", {}).get(game, [])
               + state.get("released_trials", {}).get(game, []))
    by_seed = {trial["seed"]: trial for trial in history}
    by_seed.update({trial["seed"]: trial for trial in trials})
    return list(by_seed.values())


def baseline_gate(state, trial):
    previous = [item for item in successful_trials(state, trial["game"])
                if item["seed"] != trial["seed"]]
    scores = [item["full_flash"]["score"] for item in previous] + [trial["full_flash"]["score"]]
    mean_score = math.fsum(scores) / len(scores)
    comparison = compare_score(mean_score, state["config"]["references"][trial["game"]])
    random, human, paper = state["config"]["references"][trial["game"]]
    minimum_mean = paper + HNS_MARGIN * (human - random)
    individual = trial["full_flash"]["passes"]
    mean_passes = bool(previous) and mean_score >= minimum_mean
    return dict(accepted=individual or mean_passes, individual_passes=individual,
                mean_passes=mean_passes, previous_pair_seeds=[item["seed"] for item in previous],
                mean_score=mean_score, mean_hns=comparison["hns"], minimum_mean_score=minimum_mean)


def next_seed(state, game):
    from reproduction_seeds import occupied_seeds

    start = (state["config"]["start_seed"]
             + state["config"].get("shared_seed_offsets", {}).get(game, 0))
    step = state["config"].get("seed_step", 1)
    used = {trial["seed"] for trial in game_trials(state, game)}
    seed = max([start] + [value + step for value in used if value >= start and (value - start) % step == 0])
    occupied = (used | set(state.get("reserved_seeds", {}).get(game, ()))
                | set(occupied_seeds(ROOT).get(game, ())))
    while seed in occupied:
        seed += step
    return validate_seed(seed)


def artifact_dir(state, trial):
    search_id = trial.get("artifact_search_id", state["search_id"])
    if not search_id.startswith("search_") or not search_id[7:].isalnum():
        raise ValueError("Invalid artifact search ID")
    return ROOT / "callbacks" / "reproduction" / search_id / trial["game"] / f"round_{trial['round']:06d}"


def identity(state, trial):
    return {key: value for key, value in (
        ("search_id", trial.get("artifact_search_id", state["search_id"])), ("game", trial["game"]),
        ("round", trial["round"]), ("seed", trial["seed"]))}


def trial_paths(state, trial):
    directory = artifact_dir(state, trial)
    run = directory.relative_to(ROOT / "callbacks") / f"seed{trial['seed']}"
    environment = "atari100k-" + trial["game"]
    branches = {name: ROOT / "callbacks" / (str(run) + suffix) / environment
                for name, suffix in (("warmup", "_warmup"), ("full_flash", "_O"), ("baseline", "_X"))}
    return directory, str(run), branches


def historical_baselines(state, game):
    if len(successful_trials(state, game)) >= KEEP:
        return []
    # Imported records are usable only if their original artifacts are also
    # accessible here. Numeric history alone never starts fresh training.
    return sorted((trial for trial in state["trials"] if trial["game"] == game
                   and trial["status"] == "complete" and trial["full_flash"] is not None
                   and trial["baseline"] is None),
                  key=lambda trial: (-trial["full_flash"]["score"], trial["round"]))


class Campaign:
    def __init__(self, path=DEFAULT_STATE):
        self.path = Path(path).expanduser().resolve()

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def read(self):
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("root") != str(ROOT):
            raise ValueError("Campaign version or original TWISTER path does not match")
        # All deletion paths are derived from these validated components.
        if not state["search_id"].startswith("search_") or not state["search_id"][7:].isalnum():
            raise ValueError("Invalid campaign search ID")
        games = state["config"]["games"]
        if not games or len(set(games)) != len(games) or set(games) - set(references()):
            raise ValueError("Invalid campaign game list")
        known_games = set(references())
        for trial in state["trials"]:
            # Released trials remain on their original server with all paths
            # intact; only config.games determines which work may be claimed.
            if trial["game"] not in known_games or type(trial["round"]) is not int or trial["round"] < 1:
                raise ValueError("Invalid trial identity")
            validate_seed(trial["seed"])
        return state

    def initialize(self, games=None, start_seed=None, no_wandb=None):
        with self.locked():
            if self.path.exists():
                state = self.read()
                for key, supplied in (("games", games), ("start_seed", start_seed), ("no_wandb", no_wandb)):
                    if supplied is not None and state["config"][key] != supplied:
                        raise ValueError(f"Existing campaign has different {key}; use a different --state")
                return state
            table = references()
            games = list(table) if games is None else games
            if not games or len(set(games)) != len(games) or set(games) - set(table):
                raise ValueError("--games must contain distinct Atari ROM names from the reference table")
            start_seed = START_SEED if start_seed is None else start_seed
            validate_seed(start_seed)
            override = load_run_config()
            override.update(retrieval_enabled="Both", final_eval_enabled=True,
                            load_replay_buffer_state_dict=True)
            override.pop("seed", None)
            state = dict(version=1, root=str(ROOT), search_id="search_" + uuid.uuid4().hex,
                         round=1, trials=[], config=dict(
                             games=games, start_seed=start_seed, seed_step=SEED_STEP, no_wandb=bool(no_wandb),
                             python=sys.executable, override_config=override,
                             references={game: table[game] for game in games}, keep=KEEP,
                             hns_margin=HNS_MARGIN))
            write_json(self.path, state)
            return state

    def _lease(self, trial):
        directory = self.path.parent / (self.path.stem + ".leases")
        directory.mkdir(exist_ok=True)
        lease = (directory / f"{trial['round']:06d}_{trial['game']}.lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            return None
        return lease

    def _cleanup(self, state):
        for trial in state["trials"]:
            if (trial.get("imported_from") or trial["game"] not in state["config"]["games"]
                    or not trial.get("cleanup_pending")):
                continue
            if any(item["game"] == trial["game"] and item["status"] not in {"complete", "cancelled"}
                   for item in state["trials"]):
                # Finish the new pair before deciding whether its high score
                # makes a pending historical baseline usable.
                continue
            # An old deletion intent may predate a newly qualifying mean.
            if self._recoverable(state, trial):
                trial.update(cleanup_pending=False, retained=True)
                trial.pop("cleanup_started", None)
                write_json(self.path, state)
                continue
            directory = artifact_dir(state, trial)
            if directory.is_symlink() or directory.resolve() != directory:
                raise ValueError(f"Refusing to prune a symlink path: {directory}")
            if directory.exists():
                marker = directory / MARKER
                if not trial.get("cleanup_started"):
                    if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != identity(state, trial):
                        raise ValueError(f"Refusing to prune an unowned trial directory: {directory}")
                    # A killed rmtree may already have removed the marker. Record
                    # ownership verification so that partial cleanup can resume.
                    trial["cleanup_started"] = True
                    write_json(self.path, state)
                print(f"Pruning {trial['game']} seed {trial['seed']}: {directory}", flush=True)
                shutil.rmtree(directory)
            trial["cleanup_pending"] = False
            write_json(self.path, state)

    def _recoverable(self, state, trial):
        if (trial["status"] != "complete" or trial["full_flash"] is None
                or trial["baseline"] is not None
                or len(successful_trials(state, trial["game"])) >= KEEP
                or not baseline_gate(state, trial)["accepted"]):
            return False
        from reproduction_recovery import available_baseline_source
        source, _ = available_baseline_source(state, trial)
        return source is not None

    def _claim_historical(self, state, game):
        from reproduction_recovery import available_baseline_source

        for trial in historical_baselines(state, game):
            gate = baseline_gate(state, trial)
            if not gate["accepted"]:
                # All remaining scores are lower under the same paired mean.
                break
            lease = self._lease(trial)
            if lease is None:
                continue
            reused = False
            try:
                reused = self._reuse_baseline(state, trial, gate)
            finally:
                if not reused:
                    lease.close()
            if reused:
                return "trial", state, dict(trial), lease
            source, reason = available_baseline_source(state, trial)
            if source is None:
                skipped = dict(reason=reason, previous_pair_seeds=gate["previous_pair_seeds"])
                if trial.get("baseline_recovery_skipped") != skipped:
                    trial["baseline_recovery_skipped"] = skipped
                    write_json(self.path, state)
                    print(f"{game} seed {trial['seed']}: skipping historical baseline: {reason}", flush=True)
                continue
            lease = self._lease(trial)
            if lease is None:
                continue
            old_gate = trial.get("baseline_gate")
            if old_gate is not None and old_gate != gate:
                trial.setdefault("baseline_gate_history", []).append(old_gate)
            trial.update(baseline_gate=gate, baseline_recovery=source, status="running", stage="baseline",
                         worker=f"{socket.gethostname()}:{os.getpid()}", error=None, cleanup_pending=False)
            if not trial.get("imported_from"):
                trial["retained"] = True
            trial.pop("cleanup_started", None)
            write_json(self.path, state)
            print(f"{game} seed {trial['seed']}: resuming historical baseline from saved {source['mode']}", flush=True)
            return "trial", state, dict(trial), lease
        return None

    def _reuse_baseline(self, state, trial, gate=None):
        """Caller holds this campaign lock; publish the binding before state."""
        from results.baseline_pairs import allocation_lock, load_pairings, load_pool, pairing_path

        gate = trial.get("baseline_gate", {}) if gate is None else gate
        if (trial["game"] == "gopher" or trial["baseline"] is not None or not gate.get("accepted")):
            return False
        bindings = load_pairings(ROOT).get(trial["game"], {})
        existing = bindings.get(str(trial["seed"]))
        # Independent servers cannot atomically share a pool lock. Let the
        # shared game's original owner consume its pool; all its GPUs can help.
        if (trial["game"] in state["config"].get("shared_games", [])
                and state["config"].get("shared_seed_offsets", {}).get(trial["game"], 0) != 0
                and existing is None):
            return False
        candidates = load_pool(ROOT).get(trial["game"], []) if existing is None else []
        if not candidates and existing is None:
            return False
        with allocation_lock(ROOT):
            bindings = load_pairings(ROOT).get(trial["game"], {})
            binding = bindings.get(str(trial["seed"]))
            if binding is None:
                used = {item["baseline"]["seed"] for item in bindings.values()}
                for previous in successful_trials(state, trial["game"]):
                    used.add(previous["baseline"].get("seed", previous["seed"]))
                candidate = next((entry for entry in candidates if entry["seed"] not in used), None)
                if candidate is None:
                    return False
                reason = cancellation_reason(trial, game_progress(state)[trial["game"]])
                if reason:
                    raise TrialCancelled(reason)
                binding = dict(version=1, kind="reused_baseline_cross_seed", game=trial["game"],
                               full_flash_seed=trial["seed"], full_flash_score=trial["full_flash"]["score"],
                               baseline=dict(candidate), gate=dict(gate),
                               search_id=state["search_id"], server=state["config"].get("server"))
                write_json(pairing_path(trial["game"], trial["seed"], ROOT), binding)
            elif not math.isclose(binding["full_flash_score"], trial["full_flash"]["score"],
                                  rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Saved baseline pairing has a different full FLASH score")
            baseline = dict(version=1, env_name="atari100k-" + trial["game"],
                            seed=binding["baseline"]["seed"], retrieval_enabled=False,
                            metrics=dict(score=binding["baseline"]["score"]),
                            source="reused_baseline", pairing=binding)
            previous_gate = trial.get("baseline_gate")
            if previous_gate is not None and previous_gate != gate:
                trial.setdefault("baseline_gate_history", []).append(previous_gate)
            trial.update(baseline=baseline, baseline_reuse=str(pairing_path(
                trial["game"], trial["seed"], ROOT).relative_to(ROOT)),
                         baseline_gate=gate, status="complete", stage="done", error=None)
            write_json(self.path, state)
        print(f"{trial['game']} full FLASH seed {trial['seed']}: reused baseline seed "
              f"{baseline['seed']} (score={baseline['metrics']['score']:g}); no baseline training.", flush=True)
        return True

    def reuse_baseline(self, trial):
        with self.locked():
            state = self.read()
            saved = next(item for item in state["trials"]
                         if (item["game"], item["round"]) == (trial["game"], trial["round"]))
            self._reuse_baseline(state, saved)
            return dict(saved)

    def _restore_baseline_bindings(self, state):
        from results.baseline_pairs import load_pairings

        bindings = load_pairings(ROOT)
        for trial in state["trials"]:
            binding = bindings.get(trial["game"], {}).get(str(trial["seed"]))
            if binding is not None and trial["full_flash"] is not None and trial["baseline"] is None:
                # A crash between binding publication and the state write must
                # neither lose a pair nor consume the baseline a second time.
                self._reuse_baseline(state, trial, binding["gate"])

    def claim(self):
        with self.locked():
            state = self.read()
            if state.get("reset_pending"):
                raise ValueError("Reset is incomplete; stop workers and rerun reproduction_cluster.py setup "
                                 + state["reset_pending"] + " --reset")
            if state["config"].get("server"):
                from reproduction_cluster import apply_assignment
                if apply_assignment(state):
                    write_json(self.path, state)
            self._restore_baseline_bindings(state)
            self._cleanup(state)
            # A lease is inherited by the child as well: killing only its worker
            # cannot let another GPU reuse this trial while training is alive.
            games = state["config"]["games"]
            progress = game_progress(state)
            counts = {game: item["successful_pairs"] for game, item in progress.items()}
            active_games = Counter()
            shared_games = state["config"].get("shared_games", [])
            pending = []
            for trial in state["trials"]:
                if trial["game"] in games and trial["status"] not in {"complete", "cancelled"}:
                    if trial.get("transition_pending"):
                        # A controller is ingesting a finished legacy child.
                        # Its parent may release the lease before ingestion.
                        active_games[trial["game"]] += 1
                        continue
                    reason = cancellation_reason(trial, progress[trial["game"]])
                    if reason:
                        lease = self._lease(trial)
                        if lease is not None:
                            try:
                                trial.update(status="cancelled", stage="cancelled", error=None,
                                             cancellation_reason=reason)
                                write_json(self.path, state)
                            finally:
                                lease.close()
                        continue
                    active_games[trial["game"]] += 1
                    pending.append(trial)
            unfinished = [game for game in games if counts[game] < KEEP]
            parallel_games = state["config"].get("shared_games", ("gopher", "krull"))
            # Finish each game's first three pairs before allocating fourth
            # pairs. Even interrupted fourth-pair work stays in the lower tier.
            for first_three in (True, False):
                tier = {game for game in games if (counts[game] < PRIORITY_PAIRS) == first_three}
                for trial in pending:
                    if trial["game"] not in tier:
                        continue
                    lease = self._lease(trial)
                    if lease is not None:
                        trial.update(status="running", worker=f"{socket.gethostname()}:{os.getpid()}", error=None)
                        write_json(self.path, state)
                        return "trial", state, dict(trial), lease
                # Before three completed pairs, any assigned game can occupy
                # additional GPUs. In-flight attempts are not successful pairs:
                # one lagging game can use all four GPUs instead of fourth pairs.
                ready = [game for game in unfinished if game in tier
                         and selected_pending_seeds(progress[game]) is None and (
                    first_three or active_games[game] < (
                        KEEP - counts[game] if game in parallel_games else 1))]
                if not ready:
                    continue
                # Spread work over idle/less occupied games before duplicating
                # a busy game, then use attempts and configured order as before.
                least_active = min(active_games[game] for game in ready)
                ready = [game for game in ready if active_games[game] == least_active]
                attempts = {game: len(game_trials(state, game)) for game in ready}
                ordered = sorted(ready, key=attempts.__getitem__)
                for game in ordered:
                    recovered = self._claim_historical(state, game)
                    if recovered is not None:
                        return recovered
                game = ordered[0]
                break
            else:
                return ("wait" if unfinished else "done"), state, None, None
            # Keep legacy artifact/lease round numbers unique, even for games
            # assigned late under the old server-wide round counter.
            round_number = 1 + max((trial["round"] for trial in game_trials(state, game)), default=0)
            seed = next_seed(state, game)
            trial = dict(game=game, seed=seed, round=round_number, status="running",
                         stage="warmup", full_flash=None, baseline=None, retained=True,
                         worker=f"{socket.gethostname()}:{os.getpid()}", error=None)
            if game in shared_games:
                trial["shared_trial"] = True
            lease = self._lease(trial)
            if lease is None:
                raise RuntimeError("New trial already has an active lease")
            state["round"] = max(state["round"], round_number)
            state["trials"].append(trial)
            write_json(self.path, state)
            return "trial", state, dict(trial), lease

    def update(self, trial, **changes):
        with self.locked():
            state = self.read()
            saved = next(item for item in state["trials"]
                         if (item["game"], item["round"]) == (trial["game"], trial["round"]))
            saved.update(changes)
            # A low-scoring fifth result can still qualify through the mean
            # gate. Keep its shared warmup until its baseline has finished.
            if saved["status"] in {"complete", "cancelled"}:
                for loser in retained_trials(state, trial["game"])[KEEP:]:
                    if not self._recoverable(state, loser):
                        loser.update(retained=False, cleanup_pending=True)
            # Persist the numeric result and deletion intent before removing data.
            write_json(self.path, state)
            self._cleanup(state)
            return dict(saved)

    def prepare_baseline(self, trial):
        with self.locked():
            state = self.read()
            saved = next(item for item in state["trials"]
                         if (item["game"], item["round"]) == (trial["game"], trial["round"]))
            old_gate = saved.get("baseline_gate")
            if old_gate is None or not old_gate["accepted"]:
                gate = baseline_gate(state, saved)
                if old_gate is not None and old_gate != gate:
                    saved.setdefault("baseline_gate_history", []).append(old_gate)
                saved["baseline_gate"] = gate
                write_json(self.path, state)
            return dict(saved)

    def check_cancellation(self, trial):
        state = self.read()
        saved = next(item for item in state["trials"]
                     if (item["game"], item["round"]) == (trial["game"], trial["round"]))
        if saved["status"] == "complete":
            return
        reason = (saved.get("cancellation_reason") if saved["status"] == "cancelled"
                  else cancellation_reason(saved, game_progress(state)[trial["game"]]))
        if reason:
            raise TrialCancelled(reason)


def stop_child(process, grace=10):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        # The leader may exit before an environment child that ignores TERM.
        # Reclaim the whole group before this GPU starts its next trial.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class TrainingProcessError(RuntimeError):
    def __init__(self, code, logfile):
        self.code = code
        super().__init__(f"Training exited with {code}; see {logfile}")


class TrialCancelled(RuntimeError):
    """Normal scheduling cancellation, distinct from a training failure."""


def run_command(command, environment, lease, logfile, campaign=None, trial=None):
    if campaign is not None:
        campaign.check_cancellation(trial)
    print(f"Running {' '.join(command)}\nLog: {logfile}", flush=True)
    with logfile.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lease.fileno(),))
        cancelled = False
        try:
            while True:
                try:
                    code = process.wait(timeout=1 if campaign is not None else None)
                    break
                except subprocess.TimeoutExpired:
                    campaign.check_cancellation(trial)
        except TrialCancelled:
            cancelled = True
            raise
        finally:
            # Reap environment processes even after the training leader exits.
            stop_child(process, grace=1 if cancelled else 10)
    if code:
        if campaign is not None:
            campaign.check_cancellation(trial)
        raise TrainingProcessError(code, logfile)


def read_evaluation(path, state, trial, enabled):
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("env_name") != "atari100k-" + trial["game"]
            or result.get("seed") != trial["seed"]
            or result.get("retrieval_enabled") is not enabled
            or result.get("episodes") != state["config"]["override_config"].get("final_eval_episodes", 100)):
        raise ValueError(f"Final evaluation identity/settings mismatch: {path}")
    checkpoint = Path(result["checkpoint"])
    if checkpoint.parent != path.parent or not checkpoint.is_file():
        raise ValueError(f"Final evaluation checkpoint is missing or outside the branch: {checkpoint}")
    compare_score(result["metrics"]["score"], state["config"]["references"][trial["game"]])
    return result


def run_independent_branch(campaign, state, trial, lease, enabled, environment):
    """Train from step zero, without resetting an episode to create a fork."""
    directory, run_name, paths = trial_paths(state, trial)
    branch = "full_flash" if enabled else "baseline"
    path = paths[branch]
    if path.exists():
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Refusing to replace a symlink branch: {path}")
        # A fresh run must not mix replay/checkpoints with a previous, partial
        # attempt. Keep that attempt inside this trial for normal top-4 cleanup.
        archive = directory / ("unfinished_" + branch + "_" + uuid.uuid4().hex)
        path.rename(archive)
        history = list(trial.get("independent_archives", [])) + [str(archive)]
        trial = campaign.update(trial, independent_archives=history)
    override = dict(state["config"]["override_config"], retrieval_enabled=enabled)
    environment = dict(environment, run_name=run_name + ("_O" if enabled else "_X"),
                       override_config=json.dumps(override))
    command = [state["config"]["python"], "-u", str(ROOT / "main.py"),
               "--seed", str(trial["seed"])]
    if state["config"]["no_wandb"]:
        command.append("--no-wandb")
    print(f"{trial['game']} seed {trial['seed']}: {branch} starts from step zero "
          "because no usable shared warmup exists.", flush=True)
    run_command(command, environment, lease, directory / (branch + ".log"), campaign, trial)


def validate_baseline_candidate(candidate, state, trial):
    checkpoint = Path(candidate["checkpoint"])
    override = state["config"]["override_config"]
    budget = override.get("epochs", 50) * override.get("epoch_length", 2000)
    if (candidate.get("version") != 1 or candidate.get("env_name") != "atari100k-" + trial["game"]
            or candidate.get("seed") != trial["seed"] or candidate.get("model_step") != budget
            or candidate.get("training_budget") != budget or candidate.get("uninterrupted") is not True
            or candidate.get("paused_environments") != 0 or checkpoint.resolve() != checkpoint
            or not checkpoint.is_relative_to(artifact_dir(state, trial)) or not checkpoint.is_file()):
        raise ValueError("Completed warmup is not a compatible full-budget baseline")
    return checkpoint


def evaluate_warmup_baseline(state, trial, lease, environment, campaign=None):
    """Evaluate existing baseline weights; perform no optimizer updates."""
    directory, run_name, paths = trial_paths(state, trial)
    candidate = trial["warmup_fallback"]["baseline_candidate"]
    source = validate_baseline_candidate(candidate, state, trial)
    output = paths["baseline"]
    if output.is_symlink() or output.resolve() != output:
        raise ValueError(f"Refusing to evaluate in a symlink branch: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / source.name
    if checkpoint.exists():
        if not checkpoint.samefile(source):
            raise ValueError(f"Baseline checkpoint does not match the completed warmup: {checkpoint}")
    else:
        # Both live inside one trial on the same filesystem. Keep the original
        # immutable weights without another 0.5-GiB copy or checkpoint rewrite.
        os.link(source, checkpoint)
    write_json(output / "training_source.json", candidate)
    override = dict(state["config"]["override_config"], retrieval_enabled=False)
    environment = dict(environment, run_name=run_name + "_X", override_config=json.dumps(override))
    command = [state["config"]["python"], "-u", str(ROOT / "main.py"),
               "--seed", str(trial["seed"]), "--mode", "final_evaluation", "--checkpoint", str(checkpoint)]
    if state["config"]["no_wandb"]:
        command.append("--no-wandb")
    print(f"{trial['game']} seed {trial['seed']}: evaluating the completed warmup as baseline; "
          "no baseline training is needed.", flush=True)
    run_command(command, environment, lease, directory / "baseline.log", campaign, trial)


def run_trial(campaign, state, trial, lease):
    if trial["status"] == "complete" and trial["baseline"] is not None:
        return
    from retrieval_runs import branch_commands, WARMUP_UNAVAILABLE_EXIT_CODE

    campaign.check_cancellation(trial)
    directory, run_name, paths = trial_paths(state, trial)
    shared = paths["warmup"] / "shared_warmup"
    environment = dict(os.environ, env_name="atari100k-" + trial["game"], run_name=run_name,
                       override_config=json.dumps(state["config"]["override_config"]),
                       WANDB_DIR=str(directory))
    python = state["config"]["python"]
    if trial.get("baseline_recovery") and trial["baseline"] is None:
        from reproduction_recovery import available_baseline_source
        source, reason = available_baseline_source(state, trial)
        if source is None:
            campaign.update(trial, status="complete", stage="done", error=None,
                            baseline_recovery_skipped=dict(reason=reason))
            print(f"{trial['game']} seed {trial['seed']}: saved baseline resources are unavailable: {reason}", flush=True)
            return
        if source["mode"] == "evaluate_warmup":
            fallback = dict(trial.get("warmup_fallback", {}), baseline_candidate=source["candidate"])
            trial = campaign.update(trial, warmup_fallback=fallback)
    if trial["full_flash"] is None:
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / MARKER
        if marker.exists() and json.loads(marker.read_text()) != identity(state, trial):
            raise ValueError(f"Trial ownership mismatch: {directory}")
        write_json(marker, identity(state, trial))
        if not trial.get("warmup_fallback") and not (shared / "run.json").is_file():
            campaign.update(trial, stage="warmup")
            command = [python, "-u", str(ROOT / "main.py"), "--seed", str(trial["seed"]),
                       "--warmup_only", "--load_last"]
            if state["config"]["no_wandb"]:
                command.append("--no-wandb")
            try:
                run_command(command, environment, lease, directory / "warmup.log", campaign, trial)
            except TrainingProcessError as error:
                if error.code != WARMUP_UNAVAILABLE_EXIT_CODE:
                    raise
                fallback = dict(mode="independent", reason=str(error), time=time.time())
                candidate_file = paths["warmup"] / "baseline_candidate.json"
                if candidate_file.is_file():
                    candidate = json.loads(candidate_file.read_text())
                    validate_baseline_candidate(candidate, state, trial)
                    fallback["baseline_candidate"] = candidate
                trial = campaign.update(trial, warmup_fallback=fallback)
        if not trial.get("warmup_fallback"):
            plan = json.loads((shared / "run.json").read_text(encoding="utf-8"))
            if (plan["env_name"] != environment["env_name"] or plan["run_name"] != run_name
                    or plan["override_config"].get("seed") != trial["seed"]
                    or not (shared / plan["checkpoint"]).is_file()):
                raise ValueError(f"Shared warmup identity/checkpoint mismatch: {shared}")
        campaign.update(trial, stage="full_flash")
        result_path = paths["full_flash"] / "final_evaluation.json"
        if not result_path.is_file():
            if trial.get("warmup_fallback"):
                run_independent_branch(campaign, state, trial, lease, True, environment)
            else:
                command, _ = branch_commands(shared)
                command[0] = python
                run_command(command, environment, lease, directory / "full_flash.log", campaign, trial)
        result = read_evaluation(result_path, state, trial, True)
        comparison = compare_score(result["metrics"]["score"], state["config"]["references"][trial["game"]])
        comparison["evaluation"] = result
        trial = campaign.update(trial, full_flash=comparison)
        print(f"{trial['game']} seed {trial['seed']}: full FLASH={comparison['score']}, "
              f"delta HNS={comparison['delta_hns']:+.6f}, passes={comparison['passes']}", flush=True)
    trial = campaign.prepare_baseline(trial)
    gate = trial["baseline_gate"]
    print(f"{trial['game']} seed {trial['seed']}: baseline gate accepted={gate['accepted']}, "
          f"individual={gate['individual_passes']}, paired mean={gate['mean_passes']} "
          f"(mean HNS={gate['mean_hns']:.6f}, previous pairs={len(gate['previous_pair_seeds'])})", flush=True)
    if gate["accepted"] and trial["baseline"] is None:
        trial = campaign.reuse_baseline(trial)
    if gate["accepted"] and trial["baseline"] is None:
        campaign.update(trial, stage="baseline")
        result_path = paths["baseline"] / "final_evaluation.json"
        if not result_path.is_file():
            if trial.get("warmup_fallback", {}).get("baseline_candidate"):
                evaluate_warmup_baseline(state, trial, lease, environment, campaign)
            elif trial.get("warmup_fallback"):
                run_independent_branch(campaign, state, trial, lease, False, environment)
            else:
                _, command = branch_commands(shared)
                command[0] = python
                run_command(command, environment, lease, directory / "baseline.log", campaign, trial)
        result = read_evaluation(result_path, state, trial, False)
        trial = campaign.update(trial, baseline=result)
    campaign.update(trial, stage="done", status="complete", error=None)


def show_status(state):
    print(f"Campaign: {state['search_id']} | round {state['round']}")
    counts = successful_pair_counts(state)
    for game in state["config"]["games"]:
        retained = retained_trials(state, game)
        count = counts[game]
        scope = (f" ({len(local_successful_trials(state, game))} local; includes last-sync remote results)"
                 if game in state["config"].get("shared_games", []) else "")
        scores = ", ".join(f"seed{trial['seed']}={trial['full_flash']['score']:g}" for trial in retained)
        print(f"{game}: {count}/{KEEP} successful pairs{scope}; local top {len(retained)}: {scores or '-'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="Shared by local GPU workers; defaults to the server's configured state, otherwise results/reproduction/state.json")
    parser.add_argument("--games", nargs="+", help="Initial game order (default: all 26 Atari100k games)")
    parser.add_argument("--start-seed", type=int, help="First round seed (default: 3090); each round adds 10")
    parser.add_argument("--no-wandb", action="store_true", default=None, help="Disable W&B for a new campaign")
    parser.add_argument("--status", action="store_true", help="Read progress without training or pruning")
    parser.add_argument("--once", action="store_true", help="Run at most one available trial, including any conditional baseline; exit without waiting if no trial is available")
    parser.add_argument("--supervised", action="store_true", help="With --once, return 20 when waiting and 21 when all games are complete")
    parser.add_argument("--server", choices=("a", "b", "c"), help="Bind this server to its game assignment; normally set once with reproduction_cluster.py setup")
    parser.add_argument("--require-server", action="store_true", help="Require cluster setup before starting a worker")
    parser.add_argument("--check-gpu", action="store_true", help="Verify selected GPU model and PyTorch CUDA before starting a worker")
    args = parser.parse_args(argv)
    if args.supervised and not args.once:
        parser.error("--supervised requires --once")
    from reproduction_cluster import attach_server, local_settings
    settings = local_settings()
    campaign = Campaign(args.state or settings.get("state") or DEFAULT_STATE)
    if args.status:
        state = campaign.read()
        if state["config"].get("server"):
            from reproduction_cluster import apply_assignment
            apply_assignment(state)
        show_status(state)
        return 0
    server = args.server
    if server is None and settings.get("state") == str(campaign.path):
        server = settings.get("server")
    if args.require_server and not server:
        raise ValueError("Server is not configured. Run python reproduction_cluster.py setup b on A6000 "
                         "(setup a for the 3090 server, setup c for the single-GPU server) before N_train.sh")
    if args.check_gpu:
        from gpu_runtime import check_gpu
        from reproduction_cluster import load_plan
        expected_model = load_plan()["servers"][server].get("gpu_model") if server else None
        device, _ = check_gpu(expected_model)
        # Bind by UUID so CUDA's enumeration order cannot select another card
        # when several physical GPUs have the same model name.
        os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    campaign.initialize(args.games, args.start_seed, args.no_wandb)
    if server:
        attach_server(campaign, server)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            kind, state, trial, lease = campaign.claim()
            if kind == "done":
                show_status(state)
                return 21 if args.supervised else 0
            if kind == "wait":
                if args.once:
                    print("No trial is currently available; --once exits without starting training.", flush=True)
                    return 20 if args.supervised else 0
                time.sleep(10)
                continue
            try:
                print(f"Round {trial['round']}: {trial['game']} seed {trial['seed']}", flush=True)
                run_trial(campaign, state, trial, lease)
            except TrialCancelled as error:
                campaign.update(trial, status="cancelled", stage="cancelled", error=None,
                                cancellation_reason=str(error))
                print(f"{trial['game']} seed {trial['seed']}: cancelled excess work; taking the next game.", flush=True)
            except BaseException as error:
                campaign.update(trial, status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                                error=str(error))
                raise
            finally:
                lease.close()
            if args.once:
                return 0
    except KeyboardInterrupt:
        print("Worker stopped; restart with the same --state to continue.", file=sys.stderr)
        return 130
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Reproduction worker failed: {error}", file=sys.stderr)
        raise SystemExit(1)
