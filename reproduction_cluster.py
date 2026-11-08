#!/usr/bin/env python3
"""Independent server campaigns, small Git reports, and unstarted-game handoffs."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

import reproduce
from run_results import write_json


def plan_path():
    return reproduce.ROOT / "reproduction/servers.json"


def settings_path():
    return reproduce.ROOT / "reproduction.local.json"


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
        assigned.extend(entry["games"])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(reproduce.references()):
        raise ValueError("Server plan must assign each of the 26 games exactly once")
    return plan


def apply_assignment(state, plan=None):
    """Called under the local campaign lock; a started game keeps its owner."""
    plan = load_plan() if plan is None else plan
    server = state["config"]["server"]
    if server not in plan["servers"]:
        raise ValueError(f"Server {server} is missing from the plan")
    games = plan["servers"][server]["games"]
    started = {trial["game"] for trial in state["trials"]}
    if started - set(games):
        raise ValueError("Cannot transfer games with training history: " + ", ".join(sorted(started - set(games))))
    if state["config"]["games"] == games:
        return False
    state["config"]["games"] = list(games)
    table = reproduce.references()
    for game in games:
        state["config"]["references"].setdefault(game, table[game])
    return True


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


def setup_server(campaign, server):
    settings = local_settings()
    if settings.get("server", server) != server:
        raise ValueError("This machine is already configured as another server")
    if settings.get("state", str(campaign.path)) != str(campaign.path):
        raise ValueError("This server is already configured with another local state path")
    plan = load_plan()
    if server not in plan["servers"]:
        raise ValueError(f"Unknown server: {server}")
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


def export_report(campaign, server):
    with campaign.locked():
        state = campaign.read()
        if state["config"].get("server") != server:
            raise ValueError("Campaign/server identity mismatch")
        if apply_assignment(state):
            write_json(campaign.path, state)
        games = {}
        for game in state["config"]["games"]:
            trials = [trial for trial in state["trials"] if trial["game"] == game]
            top = reproduce.retained_trials(state, game)
            games[game] = dict(
                attempts=len(trials), completed=sum(trial["status"] == "complete" for trial in trials),
                successful_pairs=len(reproduce.successful_trials(state, game)),
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
        path = reproduce.ROOT / "reproduction/reports" / f"{server}.json"
        if not path.exists():
            print(f"{server:6}  {entry['gpus']:4}  {len(entry['games']):5}  {'-':5}  no report")
            continue
        report = json.loads(path.read_text())
        current = [report["games"][game] for game in entry["games"] if game in report["games"]]
        done = sum(game["completed"] for game in current)
        pairs = sum(game["successful_pairs"] for game in current)
        print(f"{server:6}  {entry['gpus']:4}  {len(entry['games']):5}  {report['round']:5}  "
              f"{done:15}  {pairs:7}/{4 * len(entry['games']):<8}  {report['updated_at']}")
        unstarted = [game for game in entry["games"]
                     if game in report["games"] and report["games"][game]["attempts"] == 0]
        if unstarted:
            print("        Not started at last sync: " + ", ".join(unstarted))


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
        source = plan["servers"][server]["games"]
        if game not in source:
            raise ValueError("Run move on the server that currently owns this game")
        if any(trial["game"] == game for trial in state["trials"]):
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
    sync = commands.add_parser("sync", help="Export, commit this server's report, pull/rebase, push, and show all reports")
    sync.add_argument("--local-only", action="store_true", help="Export and show reports without any Git/network operations")
    commands.add_parser("status", help="Show the most recently exchanged reports")
    move = commands.add_parser("move", help="Hand off an unstarted game from this server")
    move.add_argument("game")
    move.add_argument("target", choices=("a", "b", "c"))
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup_server(reproduce.Campaign(args.state), args.server)
        return 0
    if args.command == "status":
        show_reports()
        return 0
    campaign, settings = configured_campaign()
    # Serialize daily sync/handoff on this machine, independently of live jobs.
    lock_path = campaign.path.parent / "cluster_sync.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "move":
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
