#!/usr/bin/env python3
"""Round-robin full FLASH search with a paper-score gate and per-game top four."""

import argparse
import ast
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
                   and trial["retained"] and trial["full_flash"] is not None),
                  key=lambda trial: (-trial["full_flash"]["score"], trial["round"]))


def successful_trials(state, game):
    return [trial for trial in retained_trials(state, game)
            if trial["full_flash"]["passes"] and trial["baseline"] is not None
            and trial["status"] == "complete"]


def artifact_dir(state, trial):
    return ROOT / "callbacks" / "reproduction" / state["search_id"] / trial["game"] / f"round_{trial['round']:06d}"


def identity(state, trial):
    return {key: value for key, value in (
        ("search_id", state["search_id"]), ("game", trial["game"]),
        ("round", trial["round"]), ("seed", trial["seed"]))}


def trial_paths(state, trial):
    directory = artifact_dir(state, trial)
    run = directory.relative_to(ROOT / "callbacks") / f"seed{trial['seed']}"
    environment = "atari100k-" + trial["game"]
    branches = {name: ROOT / "callbacks" / (str(run) + suffix) / environment
                for name, suffix in (("warmup", "_warmup"), ("full_flash", "_O"), ("baseline", "_X"))}
    return directory, str(run), branches


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
        for trial in state["trials"]:
            if trial["game"] not in games or type(trial["round"]) is not int or trial["round"] < 1:
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
            if not trial.get("cleanup_pending"):
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
            self._cleanup(state)
            # A lease is inherited by the child as well: killing only its worker
            # cannot let another GPU reuse this trial while training is alive.
            active_games = set()
            for trial in state["trials"]:
                if trial["status"] != "complete":
                    active_games.add(trial["game"])
                    lease = self._lease(trial)
                    if lease is not None:
                        trial.update(status="running", worker=f"{socket.gethostname()}:{os.getpid()}", error=None)
                        write_json(self.path, state)
                        return "trial", state, dict(trial), lease
            games = state["config"]["games"]
            unfinished = [game for game in games if len(successful_trials(state, game)) < KEEP]
            if not unfinished:
                return "done", state, None, None
            ready = [game for game in unfinished if game not in active_games]
            if not ready:
                return "wait", state, None, None
            # Prefer games tried fewer times, breaking ties by the configured
            # game order. Busy games do not hold up the next pass for idle ones.
            attempts = {game: sum(trial["game"] == game for trial in state["trials"])
                        for game in ready}
            game = min(ready, key=attempts.__getitem__)
            # Keep legacy artifact/lease round numbers unique, even for games
            # assigned late under the old server-wide round counter.
            round_number = 1 + max((trial["round"] for trial in state["trials"]
                                    if trial["game"] == game), default=0)
            seed = (state["config"]["start_seed"]
                    + attempts[game] * state["config"].get("seed_step", 1))
            validate_seed(seed)
            trial = dict(game=game, seed=seed, round=round_number, status="running",
                         stage="warmup", full_flash=None, baseline=None, retained=True,
                         worker=f"{socket.gethostname()}:{os.getpid()}", error=None)
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
            for loser in retained_trials(state, trial["game"])[KEEP:]:
                loser.update(retained=False, cleanup_pending=True)
            # Persist the numeric result and deletion intent before removing data.
            write_json(self.path, state)
            self._cleanup(state)
            return dict(saved)


def stop_child(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class TrainingProcessError(RuntimeError):
    def __init__(self, code, logfile):
        self.code = code
        super().__init__(f"Training exited with {code}; see {logfile}")


def run_command(command, environment, lease, logfile):
    print(f"Running {' '.join(command)}\nLog: {logfile}", flush=True)
    with logfile.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lease.fileno(),))
        try:
            code = process.wait()
        finally:
            # Reap environment processes even after the training leader exits.
            stop_child(process)
    if code:
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
    run_command(command, environment, lease, directory / (branch + ".log"))


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


def evaluate_warmup_baseline(state, trial, lease, environment):
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
    run_command(command, environment, lease, directory / "baseline.log")


def run_trial(campaign, state, trial, lease):
    from retrieval_runs import branch_commands, WARMUP_UNAVAILABLE_EXIT_CODE

    directory, run_name, paths = trial_paths(state, trial)
    shared = paths["warmup"] / "shared_warmup"
    environment = dict(os.environ, env_name="atari100k-" + trial["game"], run_name=run_name,
                       override_config=json.dumps(state["config"]["override_config"]),
                       WANDB_DIR=str(directory))
    python = state["config"]["python"]
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
                run_command(command, environment, lease, directory / "warmup.log")
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
                run_command(command, environment, lease, directory / "full_flash.log")
        result = read_evaluation(result_path, state, trial, True)
        comparison = compare_score(result["metrics"]["score"], state["config"]["references"][trial["game"]])
        comparison["evaluation"] = result
        trial = campaign.update(trial, full_flash=comparison)
        print(f"{trial['game']} seed {trial['seed']}: full FLASH={comparison['score']}, "
              f"delta HNS={comparison['delta_hns']:+.6f}, passes={comparison['passes']}", flush=True)
    if trial["full_flash"]["passes"] and trial["retained"] and trial["baseline"] is None:
        campaign.update(trial, stage="baseline")
        result_path = paths["baseline"] / "final_evaluation.json"
        if not result_path.is_file():
            if trial.get("warmup_fallback", {}).get("baseline_candidate"):
                evaluate_warmup_baseline(state, trial, lease, environment)
            elif trial.get("warmup_fallback"):
                run_independent_branch(campaign, state, trial, lease, False, environment)
            else:
                _, command = branch_commands(shared)
                command[0] = python
                run_command(command, environment, lease, directory / "baseline.log")
        result = read_evaluation(result_path, state, trial, False)
        trial = campaign.update(trial, baseline=result)
    campaign.update(trial, stage="done", status="complete", error=None)


def show_status(state):
    print(f"Campaign: {state['search_id']} | round {state['round']}")
    for game in state["config"]["games"]:
        retained = retained_trials(state, game)
        count = len(successful_trials(state, game))
        scores = ", ".join(f"seed{trial['seed']}={trial['full_flash']['score']:g}" for trial in retained)
        print(f"{game}: {count}/{KEEP} successful pairs; top {len(retained)}: {scores or '-'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="Shared by local GPU workers; defaults to the server's configured state, otherwise results/reproduction/state.json")
    parser.add_argument("--games", nargs="+", help="Initial game order (default: all 26 Atari100k games)")
    parser.add_argument("--start-seed", type=int, help="First round seed (default: 3090); each round adds 10")
    parser.add_argument("--no-wandb", action="store_true", default=None, help="Disable W&B for a new campaign")
    parser.add_argument("--status", action="store_true", help="Read progress without training or pruning")
    parser.add_argument("--once", action="store_true", help="Run at most one available trial, including any conditional baseline; exit without waiting if no trial is available")
    parser.add_argument("--server", choices=("a", "b", "c"), help="Bind this server to its game assignment; normally set once with reproduction_cluster.py setup")
    parser.add_argument("--require-server", action="store_true", help="Require cluster setup before starting a worker")
    parser.add_argument("--check-gpu", action="store_true", help="Verify selected GPU model and PyTorch CUDA before starting a worker")
    args = parser.parse_args(argv)
    from reproduction_cluster import attach_server, local_settings
    settings = local_settings()
    campaign = Campaign(args.state or settings.get("state") or DEFAULT_STATE)
    if args.status:
        show_status(campaign.read())
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
                return 0
            if kind == "wait":
                if args.once:
                    print("No trial is currently available; --once exits without starting training.", flush=True)
                    return 0
                time.sleep(10)
                continue
            try:
                print(f"Round {trial['round']}: {trial['game']} seed {trial['seed']}", flush=True)
                run_trial(campaign, state, trial, lease)
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
