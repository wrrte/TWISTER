"""Run TWISTER's final Atari evaluation and save every completed episode score.

Uses the existing process evaluator without changing its policy, RNG handling,
environment resets, or episode selection. Episode numbers follow completion
order across parallel environments. No training or W&B upload is performed.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="atari100k-gopher")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--num-envs", type=positive_int, default=32,
                        help="parallel evaluation environments (at least 2)")
    parser.add_argument("--training-num-envs", type=positive_int, default=4,
                        help="original training environment count; determines evaluation seed offset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 4294967295")
    if not args.env_name.startswith("atari100k-"):
        parser.error("this script supports Atari environments")
    if args.num_envs < 2:
        parser.error("--num-envs must be at least 2 to use the process evaluator")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    for name in ("episode_scores.csv", "evaluation_summary.json"):
        if (args.output_dir / name).exists():
            parser.error(f"output already exists: {args.output_dir / name}; choose another --output-dir")
    return args


@contextmanager
def record_episodes(evaluation, csv_path, rows):
    """Observe the existing episode generator; restore it even on interruption."""
    original = evaluation._play_vector
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode", "score", "frames"))
        writer.writeheader()
        stream.flush()

        def recorded(*args, **kwargs):
            for output in original(*args, **kwargs):
                row = {
                    "episode": len(rows) + 1,
                    "score": float(output["score"]),
                    "frames": int(output["steps"]),
                }
                if not math.isfinite(row["score"]) or row["frames"] < 1:
                    raise ValueError(f"invalid episode result: {row}")
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                yield output

        evaluation._play_vector = recorded
        try:
            yield
        finally:
            evaluation._play_vector = original


def summarize(rows, metrics, expected_episodes):
    if len(rows) != expected_episodes:
        raise RuntimeError(f"expected {expected_episodes} completed episodes, got {len(rows)}")
    scores = [row["score"] for row in rows]
    means = {
        "score": statistics.mean(scores),
        "steps": statistics.mean(row["frames"] for row in rows),
    }
    for key, mean in means.items():
        if not math.isclose(mean, float(metrics[key]), rel_tol=1e-12, abs_tol=1e-8):
            raise RuntimeError(f"saved episode mean disagrees with evaluator metric {key}")
    return {
        "mean": means["score"],
        "std_sample": statistics.stdev(scores) if len(scores) > 1 else None,
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main():
    args = parse_args()
    # Keep --help and import-based validation independent of ML dependencies.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT))
    import torch
    from configuration import load_run_config
    from nnet.models import TWISTER
    from nnet.models import twister_evaluation as evaluation
    from retrieval_runs import close_environments
    from run_results import write_json

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU-enabled shell or explicitly pass --device cpu")
    torch.set_num_threads(1)
    config = load_run_config(seed=args.seed)
    config.update({
        "retrieval_enabled": False,
        "num_envs": args.training_num_envs,
        "eval_env_backend": "process",
        "final_eval_episodes": args.episodes,
        "final_eval_num_envs": args.num_envs,
        "load_replay_buffer_state_dict": False,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "episode_scores.csv"
    summary_path = args.output_dir / "evaluation_summary.json"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    model = TWISTER(env_name=args.env_name, override_config=config)
    rows = []
    try:
        # TWISTER.compile initializes model_step; it does not run training.
        model.compile()
        model.to(torch.device(args.device))
        model.load(str(args.checkpoint), evaluation_only=True)
        print(f"Recording {args.episodes} episodes to {csv_path}", flush=True)
        with record_episodes(evaluation, csv_path, rows):
            _, metrics = evaluation.evaluate_final(model, verbose_progress_bar=2)
        score_statistics = summarize(rows, metrics, args.episodes)
        write_json(summary_path, {
            "status": "complete",
            "env_name": args.env_name,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "model_step": int(model.model_step),
            "retrieval_enabled": False,
            "episodes": len(rows),
            "num_envs": min(args.num_envs, args.episodes),
            "training_num_envs": args.training_num_envs,
            "environment_seeds": [
                (args.seed + args.training_num_envs + i) % (2 ** 31 - 1)
                for i in range(min(args.num_envs, args.episodes))
            ],
            "episode_order": "completion order across parallel environments",
            "frames_definition": "evaluator steps metric: emulator frames, action repeat included",
            "time_limit_frames": model.config.time_limit_eval,
            "env_params": dict(model.config.env_params, **model.config.eval_env_params),
            "override_config": config,
            "device": str(model.device),
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "score_statistics": score_statistics,
            "episode_scores": [row["score"] for row in rows],
            "episode_results": rows,
        })
        print(f"Mean score: {score_statistics['mean']:g}")
        print(f"Individual scores: {csv_path}")
        print(f"Summary and all scores: {summary_path}")
    finally:
        close_environments(model)


if __name__ == "__main__":
    main()
