#!/usr/bin/env python3
"""Append missing Atari100k baseline jobs, filling one seed per game per round."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys

from classify_wandb_runs import HERE, SCORE_KEY, number


# ROM names used by nnet/envs/atari/atari_env.py (including games absent in Excel).
GAMES = (
    "alien", "amidar", "assault", "asterix", "bank_heist", "battle_zone",
    "boxing", "breakout", "chopper_command", "crazy_climber", "demon_attack",
    "freeway", "frostbite", "gopher", "hero", "jamesbond", "kangaroo", "krull",
    "kung_fu_master", "ms_pacman", "pong", "private_eye", "qbert", "road_runner",
    "seaquest", "up_n_down",
)
SEEDS = (9997, 9998, 9999)
GAME_ALIASES = {game.replace("_", ""): game for game in GAMES}


def game_from_environment(environment):
    environment = str(environment or "").strip().lower()
    if environment.startswith("atari100k-"):
        key = re.sub(r"[^a-z0-9]", "", environment.split("-", 1)[1])
        return GAME_ALIASES.get(key)
    return None


def read_coverage(xlsx_path):
    """Count finished finite scores and reserve running jobs; accept Unknown."""
    try:
        from openpyxl import load_workbook
    except ImportError as error:
        raise RuntimeError("Install openpyxl or use the twister Python environment") from error
    book = load_workbook(xlsx_path, read_only=True, data_only=True)
    completed, running = set(), set()
    try:
        if "Runs" not in book.sheetnames:
            raise ValueError("Workbook must contain the Runs sheet")
        rows = book["Runs"].iter_rows(values_only=True)
        headers = next(rows, ())
        required = {"Environment", "Seed", "State", "Retrieval Enable", SCORE_KEY}
        if required - set(headers):
            raise ValueError(f"Missing Runs columns: {', '.join(sorted(required - set(headers)))}")
        for values in rows:
            row = dict(zip(headers, values))
            game = game_from_environment(row["Environment"])
            seed = number(row["Seed"])
            retrieval = str(row["Retrieval Enable"]).strip().lower()
            if (game is None or seed not in SEEDS or retrieval not in {"false", "unknown"}
                    or row.get("Score Status") == "warmup"):
                continue
            key = (game, int(seed))
            state = str(row["State"]).strip().lower()
            if state == "finished" and number(row[SCORE_KEY]) is not None:
                completed.add(key)
            elif state == "running":
                running.add(key)
    finally:
        book.close()
    return completed, running - completed


def queued_jobs(text):
    """Recognize baseline training entries without executing any shell code."""
    jobs = set()
    for line in text.splitlines():
        try:
            tokens = shlex.split(line, comments=True)
            if "main.py" not in tokens:
                continue
            assignments = dict(token.split("=", 1) for token in tokens if "=" in token)
            game = game_from_environment(assignments.get("env_name"))
            config = json.loads(assignments.get("override_config", "{}"))
            seed = assignments.get("--seed")
            if "--seed" in tokens:
                seed = tokens[tokens.index("--seed") + 1]
            mode = assignments.get("--mode", assignments.get("-m", "training"))
            for flag in ("--mode", "-m"):
                if flag in tokens:
                    mode = tokens[tokens.index(flag) + 1]
            if (game and int(seed) in SEEDS and isinstance(config, dict)
                    and config.get("retrieval_enabled") is False and mode == "training"):
                jobs.add((game, int(seed)))
        except (ValueError, TypeError, IndexError):
            continue
    return jobs


def plan_jobs(completed, running, queued):
    """Reach coverage 1 for every game, then 2, then 3 (existing work counts)."""
    covered = set(completed) | set(running) | set(queued)
    jobs = []
    for target in range(1, len(SEEDS) + 1):
        for game in GAMES:
            seeds = {seed for name, seed in covered if name == game and seed in SEEDS}
            if len(seeds) >= target:
                continue
            seed = next(seed for seed in SEEDS if seed not in seeds)
            jobs.append((target, game, seed))
            covered.add((game, seed))
    return jobs


def command(game, seed):
    return (f"env_name=atari100k-{game} "
            "override_config='{\"retrieval_enabled\":false}' "
            f"python3 -u main.py --seed {seed}")


def append_jobs(xlsx_path, queue_path, dry_run=False):
    completed, running = read_coverage(xlsx_path)
    queue_path = Path(queue_path)

    def prepare():
        before = queue_path.read_text(encoding="utf-8") if queue_path.exists() else ""
        queued = queued_jobs(before)
        jobs = plan_jobs(completed, running, queued)
        return before, queued, jobs

    if dry_run:
        _, queued, jobs = prepare()
    else:
        # Queue workers use sed -i, so lock their stable .lock file, not the queue.
        with queue_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before, queued, jobs = prepare()
            if jobs:
                with queue_path.open("a", encoding="utf-8") as stream:
                    if before and not before.endswith("\n"):
                        stream.write("\n")
                    stream.write("".join(command(game, seed) + "\n" for _, game, seed in jobs))
    return completed, running, queued, jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=HERE / "converted_results.xlsx")
    parser.add_argument("--queue", type=Path, default=HERE.parent / "job_queue_titan.txt")
    parser.add_argument("--dry-run", action="store_true", help="Print new jobs without changing files")
    args = parser.parse_args(argv)
    try:
        completed, running, queued, jobs = append_jobs(args.xlsx, args.queue, args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Queue generation failed: {error}\n")
    stages = Counter(target for target, _, _ in jobs)
    print(f"완료 {len(completed)}개 / 실행 중 {len(running)}개 / 기존 대기 {len(queued)}개")
    print(f"{'추가 예정' if args.dry_run else '추가 완료'}: {len(jobs)}개 → {args.queue}")
    print(" / ".join(f"{target}개 시드 확보 단계: {stages[target]}개" for target in range(1, 4)))
    if args.dry_run:
        for _, game, seed in jobs:
            print(command(game, seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
