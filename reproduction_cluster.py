#!/usr/bin/env python3
"""Independent server campaigns, numeric history handoffs, and Git reports."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import uuid

import reproduce
from run_results import write_json


def plan_path():
    return reproduce.ROOT / "reproduction/servers.json"


def settings_path():
    return reproduce.ROOT / "reproduction.local.json"


def handoff_path(game):
    if game not in reproduce.references():
        raise ValueError(f"Unknown handoff game: {game}")
    return reproduce.ROOT / "reproduction/handoffs" / f"{game}.json"


def local_settings():
    path = settings_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_plan():
    plan = json.loads(plan_path().read_text(encoding="utf-8"))
    if plan.get("version") != 1 or not isinstance(plan.get("servers"), dict):
        raise ValueError("Unsupported reproduction server plan")
    assigned = []
    for name, entry in plan["servers"].items():
        if not re.fullmatch(r"[a-z0-9_-]+", name) or type(entry.get("gpus")) is not int or entry["gpus"] < 1:
            raise ValueError("Invalid server name or GPU count")
        if not isinstance(entry.get("games"), list) or not entry["games"]:
            raise ValueError(f"Server {name} needs at least one game")
        if entry.get("gpu_model") is not None and not isinstance(entry["gpu_model"], str):
            raise ValueError(f"Invalid GPU model for server {name}")
        assigned.extend(entry["games"])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(reproduce.references()):
        raise ValueError("Server plan must assign each of the 26 games exactly once")
    shared = plan.get("shared_games", [])
    if (not isinstance(shared, list) or any(not isinstance(game, str) for game in shared)
            or len(shared) != len(set(shared)) or set(shared) - set(assigned)):
        raise ValueError("Shared games must be distinct games from the server plan")
    for game, handoff in plan.get("handoffs", {}).items():
        if (game not in assigned or handoff.get("source") not in plan["servers"]
                or handoff.get("target") not in plan["servers"]
                or handoff["source"] == handoff["target"]
                or game not in plan["servers"][handoff["target"]]["games"]):
            raise ValueError(f"Invalid handoff: {game}")
    return plan


def assigned_games(plan, server):
    games = list(plan["servers"][server]["games"])
    return games + [game for game in plan.get("shared_games", []) if game not in games]


def game_owner(plan, game):
    return next(server for server, entry in plan["servers"].items() if game in entry["games"])


def shared_pair_counts(state):
    """Combine live local pairs and last-sync reports, excluding released copies."""
    shared = state["config"].get("shared_games", [])
    seeds = {game: {trial["seed"] for trial in reproduce.local_successful_trials(state, game)}
             for game in shared}
    if not shared:
        return {}
    plan = load_plan()
    legacy_counts = dict.fromkeys(shared, 0)
    for server in plan["servers"]:
        if server == state["config"]["server"]:
            continue
        path = reproduce.ROOT / "reproduction/reports" / f"{server}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("version") != 1 or report.get("server") != server:
            raise ValueError(f"Invalid server report: {path}")
        for game in shared:
            progress = report["games"].get(game, {})
            paired_seeds = progress.get("successful_pair_seeds")
            if paired_seeds is not None:
                if (not isinstance(paired_seeds, list)
                        or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in paired_seeds)):
                    raise ValueError(f"Invalid successful pair seeds in {path}: {game}")
                seeds[game].update(paired_seeds)
            elif server == game_owner(plan, game):
                # Old reports lack seed lists. Only the current owner's count
                # is authoritative; a former owner's report may repeat history.
                count = progress.get("successful_pairs", 0)
                if type(count) is not int or count < 0:
                    raise ValueError(f"Invalid successful pair count in {path}: {game}")
                legacy_counts[game] = count
    return {game: len(seeds[game]) + legacy_counts[game] for game in shared}


def apply_assignment(state, plan=None):
    """Adopt released numeric history, leaving original artifacts in place."""
    plan = load_plan() if plan is None else plan
    server = state["config"]["server"]
    if server not in plan["servers"]:
        raise ValueError(f"Server {server} is missing from the plan")
    games = assigned_games(plan, server)
    shared = list(plan.get("shared_games", []))
    offsets = {}
    for game in shared:
        owner = game_owner(plan, game)
        servers = [owner] + sorted(set(plan["servers"]) - {owner})
        if len(servers) > state["config"].get("seed_step", 1):
            raise ValueError("Seed step is too small for distinct shared-game server seeds")
        offsets[game] = servers.index(server)
    assignment = dict(games=games, shared_games=shared, shared_seed_offsets=offsets)
    handoffs = plan.get("handoffs", {})
    outgoing = {game for game, move in handoffs.items() if move["source"] == server}
    started = {trial["game"] for trial in state["trials"]}
    legacy_started = {trial["game"] for trial in state["trials"] if not trial.get("shared_trial")}
    owned = set(state["config"]["games"]) - set(state["config"].get("shared_games", []))
    unreleased = (outgoing & (owned | legacy_started)) - set(state.get("released_games", {}))
    if unreleased:
        raise ValueError("Release outgoing history first with python reproduction_cluster.py handoff: "
                         + ", ".join(sorted(unreleased)))
    if started - set(games) - set(state.get("released_games", {})):
        raise ValueError("Cannot transfer games without an explicit history handoff")
    changed = any(state["config"].get(key) != value for key, value in assignment.items())
    # Validate all incoming snapshots before changing local assignment/history.
    incoming = []
    for game, move in handoffs.items():
        if move["target"] != server:
            continue
        path = handoff_path(game)
        if not path.is_file():
            raise ValueError(f"{game} history is not ready. Run python reproduction_cluster.py handoff "
                             f"on {move['source']}, publish its handoff files, then pull here.")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        validate_handoff(snapshot, game, move)
        imported = state.get("imported_handoffs", {}).get(game)
        if imported == snapshot["id"]:
            continue
        if imported is not None:
            raise ValueError(f"Refusing to replace already imported {game} history")
        local_seeds = {trial["seed"] for trial in state["trials"] if trial["game"] == game}
        if local_seeds & {trial["seed"] for trial in snapshot["trials"]}:
            raise ValueError(f"Local {game} seeds overlap the source history; preserve both states for inspection")
        reference = state["config"]["references"].get(game, reproduce.references()[game])
        if reference != snapshot["reference"]:
            raise ValueError(f"Paper references differ for {game}")
        incoming.append((game, snapshot))
    for game, snapshot in incoming:
        state.setdefault("imported_history", {})[game] = snapshot["trials"]
        state.setdefault("imported_handoffs", {})[game] = snapshot["id"]
        reserved = set(state.get("reserved_seeds", {}).get(game, ())) | set(snapshot["reserved_seeds"])
        state.setdefault("reserved_seeds", {})[game] = sorted(reserved)
        changed = True
    # Keep the active trial table readable by workers already running older
    # code. They finish their current unrelated game without being interrupted.
    for game in state.get("released_games", {}):
        departing = [trial for trial in state["trials"] if trial["game"] == game
                     and (game not in shared or not trial.get("shared_trial"))]
        if departing:
            existing = state.setdefault("released_trials", {}).get(game, [])
            by_seed = {trial["seed"]: trial for trial in existing + departing}
            state["released_trials"][game] = list(by_seed.values())
            state["trials"] = [trial for trial in state["trials"] if trial not in departing]
            changed = True
    for game in list(state.get("imported_history", {})):
        if game not in games:
            continue
        move = handoffs[game]
        snapshot = json.loads(handoff_path(game).read_text(encoding="utf-8"))
        existing = {trial["seed"] for trial in state["trials"] if trial["game"] == game}
        for record in state["imported_history"][game]:
            if record["seed"] in existing:
                continue
            record = dict(record, imported_from=move["source"], artifact_search_id=snapshot["search_id"],
                          source_status=record["status"], status="complete",
                          retained=record["full_flash"] is not None)
            if record["source_status"] != "complete":
                record["baseline"] = None
            state["trials"].append(record)
        del state["imported_history"][game]
        changed = True
    state["config"].update(assignment)
    table = reproduce.references()
    for game in games:
        state["config"]["references"].setdefault(game, table[game])
    return changed


def validate_handoff(snapshot, game, move):
    if (snapshot.get("version") != 1 or snapshot.get("game") != game
            or snapshot.get("source") != move["source"] or snapshot.get("target") != move["target"]
            or not isinstance(snapshot.get("id"), str) or not snapshot["id"]
            or snapshot.get("reference") != reproduce.references()[game]
            or not isinstance(snapshot.get("search_id"), str)
            or not snapshot["search_id"].startswith("search_") or not snapshot["search_id"][7:].isalnum()):
        raise ValueError(f"Invalid numeric history handoff for {game}")
    seeds = set()
    for trial in snapshot["trials"]:
        seed = reproduce.validate_seed(trial["seed"])
        if (seed is None or seed in seeds or trial["game"] != game
                or type(trial["round"]) is not int or trial["round"] < 1
                or trial["status"] not in {"complete", "released"}):
            raise ValueError(f"Invalid or duplicate seed in {game} history")
        seeds.add(seed)
        for result in (trial["full_flash"], trial["baseline"]):
            if result is not None:
                score = result["score"] if "score" in result else result["metrics"]["score"]
                reproduce.compare_score(score, snapshot["reference"])
        if trial["baseline"] is not None and trial["full_flash"] is None:
            raise ValueError(f"Baseline without full FLASH in {game} history")
    for seed in snapshot["reserved_seeds"]:
        if reproduce.validate_seed(seed) is None:
            raise ValueError(f"Missing reserved seed in {game} history")


def release_history(campaign, settings):
    """Freeze outgoing scores/seeds; never move/delete checkpoints or replay."""
    from reproduction_seeds import occupied_seeds

    plan = load_plan()
    server = settings["server"]
    moves = {game: move for game, move in plan.get("handoffs", {}).items() if move["source"] == server}
    if not moves:
        print(f"Server {server} has no outgoing history handoffs.")
        return
    leases = []
    try:
        with campaign.locked():
            state = campaign.read()
            if state["config"].get("server") != server:
                raise ValueError("Campaign/server identity mismatch")
            for trial in state["trials"]:
                if (trial["game"] in moves and not (
                        trial["game"] in plan.get("shared_games", []) and trial.get("shared_trial"))):
                    lease = campaign._lease(trial)
                    if lease is None:
                        raise ValueError(f"{trial['game']} is still running. Finish or stop its worker before handoff.")
                    leases.append(lease)
            occupied = occupied_seeds(reproduce.ROOT)
            snapshots = {}
            for game, move in moves.items():
                released = state.get("released_games", {}).get(game)
                path = handoff_path(game)
                if released and path.is_file():
                    snapshot = json.loads(path.read_text(encoding="utf-8"))
                    validate_handoff(snapshot, game, move)
                    if snapshot["id"] != released or snapshot["search_id"] != state["search_id"]:
                        raise ValueError(f"Handoff does not belong to this campaign: {game}")
                    snapshots[game] = snapshot
                    continue
                records = []
                for trial in reproduce.game_trials(state, game):
                    full = trial["full_flash"]
                    baseline = trial["baseline"]
                    records.append(dict(
                        game=game, seed=trial["seed"], round=trial["round"],
                        status="complete" if trial["status"] == "complete" else "released",
                        retained=False,
                        full_flash=None if full is None else reproduce.compare_score(full["score"], state["config"]["references"][game]),
                        baseline=None if baseline is None else dict(metrics=dict(score=baseline["metrics"]["score"]))))
                snapshot = dict(version=1, id=released or uuid.uuid4().hex, game=game,
                                source=server, target=move["target"], search_id=state["search_id"],
                                reference=state["config"]["references"].get(game, reproduce.references()[game]),
                                trials=records, reserved_seeds=sorted(
                                    set(occupied.get(game, ())) | {trial["seed"] for trial in records}
                                    | set(state.get("reserved_seeds", {}).get(game, ()))))
                validate_handoff(snapshot, game, move)
                snapshots[game] = snapshot
            for game, snapshot in snapshots.items():
                state.setdefault("released_games", {})[game] = snapshot["id"]
            # Release locally before making any snapshot available for Git.
            write_json(campaign.path, state)
            for game, snapshot in snapshots.items():
                write_json(handoff_path(game), snapshot)
    finally:
        for lease in leases:
            lease.close()
    settings["pending_handoffs"] = sorted(set(settings.get("pending_handoffs", ())) | set(moves))
    write_json(settings_path(), settings)
    for game, snapshot in snapshots.items():
        print(f"Released {game}: {server} -> {snapshot['target']}; "
              f"{len(snapshot['trials'])} attempts, original artifacts preserved.")
    print("Publish reproduction/handoffs/*.json together with the updated code/plan. "
          "The receiver imports scores and reserves all prior seeds on restart.")


def attach_server(campaign, server):
    plan = load_plan()
    if server not in plan["servers"]:
        raise ValueError(f"Unknown server: {server}")
    with campaign.locked():
        state = campaign.read()
        if state["config"].get("server", server) != server:
            raise ValueError("This campaign is already bound to another server")
        state["config"]["server"] = server
        apply_assignment(state, plan)
        write_json(campaign.path, state)
        return state


def reset_campaign(campaign, server):
    """Explicit reset removes only this campaign's owned trial artifacts."""
    with campaign.locked():
        if not campaign.path.exists():
            return
        state = campaign.read()
        if state["config"].get("server", server) != server:
            raise ValueError("Cannot reset a campaign bound to another server")
        leases = []
        directories = []
        try:
            for trial in state["trials"]:
                lease = campaign._lease(trial)
                if lease is None:
                    raise ValueError("Stop all N_train.sh workers before setup --reset; a trial is still running")
                leases.append(lease)
                directory = reproduce.artifact_dir(state, trial)
                if directory.is_symlink() or directory.resolve() != directory:
                    raise ValueError(f"Refusing to reset a symlink path: {directory}")
                if directory.exists():
                    marker = directory / reproduce.MARKER
                    if not trial.get("reset_verified") and (marker.is_symlink() or not marker.is_file()
                            or json.loads(marker.read_text()) != reproduce.identity(state, trial)):
                        raise ValueError(f"Refusing to delete an unowned trial directory: {directory}")
                    trial["reset_verified"] = True
                    directories.append(directory)
            # Validate every lease/marker before deleting any artifacts.
            state["reset_pending"] = server
            write_json(campaign.path, state)
            for directory in directories:
                shutil.rmtree(directory)
                for parent in (directory.parent, directory.parent.parent):
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            campaign.path.unlink()
        finally:
            for lease in leases:
                lease.close()
    print(f"Reset local state and {len(directories)} owned trial artifact directories.")


def setup_server(campaign, server, reset=False):
    settings = local_settings()
    if settings.get("server", server) != server:
        raise ValueError("This machine is already configured as another server")
    if settings.get("state", str(campaign.path)) != str(campaign.path):
        raise ValueError("This server is already configured with another local state path")
    plan = load_plan()
    if server not in plan["servers"]:
        raise ValueError(f"Unknown server: {server}")
    if reset:
        reset_campaign(campaign, server)
    if not campaign.path.exists():
        campaign.initialize(plan["servers"][server]["games"])
    state = attach_server(campaign, server)
    settings.update(server=server, state=str(campaign.path))
    write_json(settings_path(), settings)
    print(f"Server {server}: {plan['servers'][server]['gpus']} GPUs, {len(state['config']['games'])} games")
    print("Games: " + ", ".join(state["config"]["games"]))
    return state


def configured_campaign():
    settings = local_settings()
    if not settings.get("server") or not settings.get("state"):
        raise ValueError("Run reproduction_cluster.py setup a (or b/c) once on this server")
    return reproduce.Campaign(settings["state"]), settings


def doctor(selected):
    from gpu_runtime import check_gpu, query_devices

    plan = load_plan()
    settings = local_settings()
    server = settings.get("server")
    print(f"Host: {socket.gethostname()}\nPython: {sys.executable}")
    print(f"Server role: {server or 'NOT CONFIGURED'}")
    devices = query_devices()
    for device in devices:
        print(f"Physical GPU {device['index']}: {device['name']} ({device['uuid']})")
    if server:
        print(f"Configured workers: {plan['servers'][server]['gpus']} (physical inventory: {len(devices)})")
    else:
        matches = [name for name, entry in plan["servers"].items() if entry.get("gpu_model")
                   and all(entry["gpu_model"].casefold() in device["name"].casefold() for device in devices)]
        if len(matches) == 1:
            print(f"Suggested setup: python reproduction_cluster.py setup {matches[0]}")
    expected = plan["servers"][server].get("gpu_model") if server else None
    check_gpu(expected, selected=selected, devices=devices)


def export_report(campaign, server):
    with campaign.locked():
        state = campaign.read()
        if state["config"].get("server") != server:
            raise ValueError("Campaign/server identity mismatch")
        if apply_assignment(state):
            write_json(campaign.path, state)
        games = {}
        for game in state["config"]["games"]:
            trials = sorted(reproduce.game_trials(state, game), key=lambda trial: (trial["round"], trial["seed"]))
            pairs = (reproduce.local_successful_trials(state, game)
                     if game in state["config"].get("shared_games", [])
                     else reproduce.successful_trials(state, game))
            top = sorted((trial for trial in trials if trial["full_flash"] is not None),
                         key=lambda trial: (-trial["full_flash"]["score"], trial["round"]))[:reproduce.KEEP]
            games[game] = dict(
                attempts=len(trials), completed=sum(
                    trial["status"] == "complete" and
                    (trial.get("source_status", "complete") == "complete" or trial["baseline"] is not None)
                    for trial in trials),
                successful_pairs=len(pairs), successful_pair_seeds=sorted({trial["seed"] for trial in pairs}),
                latest={key: trials[-1].get(key) for key in ("seed", "status", "stage")} if trials else None,
                top=[dict(seed=trial["seed"], score=trial["full_flash"]["score"],
                          delta_hns=trial["full_flash"]["delta_hns"],
                          baseline_complete=trial["baseline"] is not None) for trial in top])
        report = dict(version=1, server=server, search_id=state["search_id"], round=state["round"],
                      updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), games=games)
    path = reproduce.ROOT / "reproduction/reports" / f"{server}.json"
    write_json(path, report)
    return path


def show_reports():
    plan = load_plan()
    print("Server  GPUs  Games  Round  Complete trials  Successful pairs  Last sync (UTC)")
    for server, entry in plan["servers"].items():
        games = assigned_games(plan, server)
        path = reproduce.ROOT / "reproduction/reports" / f"{server}.json"
        if not path.exists():
            print(f"{server:6}  {entry['gpus']:4}  {len(games):5}  {'-':5}  no report")
            continue
        report = json.loads(path.read_text())
        current = [report["games"][game] for game in games if game in report["games"]]
        done = sum(game["completed"] for game in current)
        pairs = sum(game["successful_pairs"] for game in current)
        print(f"{server:6}  {entry['gpus']:4}  {len(games):5}  {report['round']:5}  "
              f"{done:15}  {pairs:16}  {report['updated_at']}")
        unstarted = [game for game in games
                     if game in report["games"] and report["games"][game]["attempts"] == 0]
        if unstarted:
            print("        Not started at last sync: " + ", ".join(unstarted))
    summary = dict(config=dict(server=None, shared_games=plan.get("shared_games", [])), trials=[])
    for game, count in shared_pair_counts(summary).items():
        print(f"{game} (shared, last sync): {count}/{reproduce.KEEP} successful pairs across all servers")


def move_game(campaign, settings, game, target):
    """Release locally before publishing a handoff. The receiver adopts on pull."""
    with campaign.locked():
        state = campaign.read()
        plan = load_plan()
        server = settings["server"]
        if state["config"].get("server") != server:
            raise ValueError("Campaign/server identity mismatch")
        if target not in plan["servers"] or target == server:
            raise ValueError("Select a different target server from the plan")
        if game in plan.get("shared_games", []):
            raise ValueError("Shared games already run on every server; keep their owner for stable seeds")
        source = plan["servers"][server]["games"]
        if game not in source:
            raise ValueError("Run move on the server that currently owns this game")
        if reproduce.game_trials(state, game):
            raise ValueError("This game has already started; keep its parameters/replay on its current server")
        if len(source) == 1:
            raise ValueError("Each server must retain at least one game")
        settings["pending_plan"] = True
        write_json(settings_path(), settings)
        source.remove(game)
        plan["servers"][target]["games"].append(game)
        write_json(plan_path(), plan)
        apply_assignment(state, plan)
        write_json(campaign.path, state)
    print(f"Released {game}: {server} -> {target}. Run sync here, then sync on {target}.")


def git(*arguments, capture=False, check=True):
    return subprocess.run(["git", *arguments], cwd=reproduce.ROOT, check=check,
                          text=True, stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def sync_git(campaign, settings):
    server = settings["server"]
    report = reproduce.ROOT / "reproduction/reports" / f"{server}.json"
    paths = [str(report.relative_to(reproduce.ROOT))]
    if settings.get("pending_plan"):
        paths.append(str(plan_path().relative_to(reproduce.ROOT)))
    for game in settings.get("pending_handoffs", ()):
        paths.append(str(handoff_path(game).relative_to(reproduce.ROOT)))
    # Commit only our report/explicit handoff. Leave unrelated work alone.
    dirty = set()
    for arguments in (("diff", "--name-only", "-z"), ("diff", "--cached", "--name-only", "-z")):
        dirty.update(filter(None, git(*arguments, capture=True).stdout.split("\0")))
    if dirty - set(paths):
        raise ValueError("Commit or stash other tracked changes before sync: " + ", ".join(sorted(dirty - set(paths))))
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", capture=True, check=False)
    if upstream.returncode:
        raise ValueError("Configure this branch's Git upstream before sync")
    export_report(campaign, server)
    git("add", "--", *paths)
    changed = git("diff", "--cached", "--quiet", "--", *paths, check=False)
    if changed.returncode == 1:
        git("commit", "--only", "-m", f"Update reproduction server {server}", "--", *paths)
    elif changed.returncode:
        raise RuntimeError("Could not inspect reproduction changes")
    for attempt in range(3):
        git("pull", "--rebase")
        pushed = git("push", check=False)
        if pushed.returncode == 0:
            settings.pop("pending_plan", None)
            settings.pop("pending_handoffs", None)
            write_json(settings_path(), settings)
            attach_server(campaign, server)
            return
        if attempt == 2:
            raise RuntimeError("Git push failed; local results/commit remain intact. Resolve the Git error and rerun sync.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="One-time local server identity; preserve existing local trials")
    setup.add_argument("server", choices=("a", "b", "c"))
    setup.add_argument("--state", type=Path, default=reproduce.DEFAULT_STATE)
    setup.add_argument("--reset", action="store_true", help="Delete this local campaign's state and owned artifacts before setup; requires stopped workers")
    sync = commands.add_parser("sync", help="Export, commit this server's report, pull/rebase, push, and show all reports")
    sync.add_argument("--local-only", action="store_true", help="Export and show reports without any Git/network operations")
    commands.add_parser("status", help="Show the most recently exchanged reports")
    diagnose = commands.add_parser("doctor", help="Check physical GPU models, server role, and selected PyTorch CUDA device without training")
    diagnose.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index or UUID (default: 0)")
    move = commands.add_parser("move", help="Hand off an unstarted game from this server")
    move.add_argument("game")
    move.add_argument("target", choices=("a", "b", "c"))
    commands.add_parser("handoff", help="Release the plan's outgoing games with all numeric history; preserve local artifacts")
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup_server(reproduce.Campaign(args.state), args.server, reset=args.reset)
        return 0
    if args.command == "doctor":
        doctor(args.gpu)
        return 0
    if args.command == "status":
        show_reports()
        return 0
    campaign, settings = configured_campaign()
    # Serialize daily sync/handoff on this machine, independently of live jobs.
    lock_path = campaign.path.parent / "cluster_sync.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "handoff":
            release_history(campaign, settings)
        elif args.command == "move":
            move_game(campaign, settings, args.game, args.target)
        elif args.local_only:
            export_report(campaign, settings["server"])
            show_reports()
        else:
            sync_git(campaign, settings)
            show_reports()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Cluster command failed: {error}", file=sys.stderr)
        raise SystemExit(1)
