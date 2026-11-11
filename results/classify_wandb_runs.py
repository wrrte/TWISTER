#!/usr/bin/env python3
"""Export TWISTER's final evaluation scores using the read-only W&B Public API."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SCORE_KEY = "Evaluation-final/score"
FINAL_KEYS = [SCORE_KEY] + ["Evaluation-final/" + key for key in (
    "steps", "episodes", "num_envs", "model_step", "action_step", "checkpoint")]
RETRIEVAL_FIELDS = {
    "warmup_steps": "Warmup Steps",
    "target": "Retrieval Target",
    "z_score_threshold": "Z Score Threshold",
    "batch_size_reduction": "Batch Size Reduction",
    "hash_bits": "Hash Bits",
    "anchor_weight": "Anchor Weight",
}
FIELDS = [
    "Run Name", "Run ID", "State", "Game", "Environment", "Seed",
    "Retrieval Enable", "Config", *RETRIEVAL_FIELDS.values(),
    "Eval Return", *FINAL_KEYS, "Score Source", "Score Status", "Error",
    "Commit", "Created At", "Run URL",
]


def number(value):
    """Accept finite scores, including zero and negative episode returns."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def config_value(config, key):
    if key in config:
        return config[key]
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def metric_value(summary, key):
    value = summary.get(key)
    if value is None:
        group, name = key.split("/", 1)
        nested = summary.get(group)
        if isinstance(nested, dict):
            value = nested.get(name)
    # Some runs configure a 'last' summary reducer. Never substitute max/mean.
    return value.get("last") if isinstance(value, dict) else value


def parse_identity(name, config):
    env = config_value(config, "env_name") or ""
    if not env:
        match = re.search(r"(?:^|/)((?:atari100k|dmc)-[^/]+)(?:/|$)", name)
        env = match.group(1) if match else ""
    seed = config_value(config, "seed")
    if seed is None:
        match = re.search(r"(?:^|[_/])seed[_-]?(\d+)(?=[_/]|$)", name, re.I)
        seed = int(match.group(1)) if match else None
    retrieval = config_value(config, "retrieval_enabled")
    run_label = name.rstrip("/").split("/")[-2] if "/" in name else name
    if run_label.endswith("_warmup"):
        retrieval = "Both"
    elif run_label.endswith("_O"):
        retrieval = True
    elif run_label.endswith("_X"):
        retrieval = False
    if retrieval is True or str(retrieval).lower() == "true":
        retrieval = "True"
    elif retrieval is False or str(retrieval).lower() == "false":
        retrieval = "False"
    elif str(retrieval).lower() == "both":
        retrieval = "Both"
    else:
        retrieval = "Unknown"
    return str(env), seed, retrieval


def seed_from_metadata(run):
    metadata = run.metadata or {}
    args = metadata.get("args", [])
    for index, argument in enumerate(args):
        if argument == "--seed" and index + 1 < len(args):
            value = str(args[index + 1])
        elif str(argument).startswith("--seed="):
            value = str(argument).split("=", 1)[1]
        else:
            continue
        if value.isdigit():
            return int(value)
    return None


def load_run_overrides(path=None):
    """Load confirmed config/score corrections keyed by run ID or run name."""
    source = Path(path) if path is not None else HERE / "run_overrides.json"
    if not source.exists() and path is None:
        return {}
    overrides = json.loads(source.read_text(encoding="utf-8"))
    allowed = {"retrieval_enabled", "env_name", "seed", "score", "steps", "episodes",
               "num_envs", "model_step", "action_step", "checkpoint", "source"}
    if not isinstance(overrides, dict):
        raise ValueError("Run overrides must map run IDs or run names to corrections")
    for identity, values in overrides.items():
        if not isinstance(values, dict) or not values or set(values) - allowed:
            raise ValueError(f"Invalid override fields for {identity}")
        if "retrieval_enabled" in values and type(values["retrieval_enabled"]) is not bool:
            raise ValueError(f"{identity}: retrieval_enabled must be true/false")
        for key in ("score", "steps"):
            if key in values and number(values[key]) is None:
                raise ValueError(f"{identity}: {key} must be a finite number")
        for key in ("seed", "episodes", "num_envs", "model_step", "action_step"):
            if key in values and (type(values[key]) is not int or values[key] < 0
                                  or (key in ("episodes", "num_envs") and values[key] == 0)):
                raise ValueError(f"{identity}: invalid {key}")
        for key in ("env_name", "checkpoint", "source"):
            if key in values and (not isinstance(values[key], str) or not values[key].strip()):
                raise ValueError(f"{identity}: {key} must be nonempty text")
    return overrides


def apply_run_overrides(rows, overrides, add_missing=True):
    """Apply confirmed results once, including local runs absent from the CSV."""
    for identity, values in overrides.items():
        matches = [row for row in rows if identity in (row.get("Run ID"), row.get("Run Name"))]
        if not matches and add_missing and "score" in values:
            environment, seed, _ = parse_identity(identity, values)
            if not environment or seed is None or "retrieval_enabled" not in values:
                raise ValueError(f"{identity}: a missing run needs environment, seed and retrieval_enabled")
            row = {field: "N/A" for field in FIELDS}
            row.update({"Run ID": identity, "Run Name": identity, "Environment": environment,
                        "Game": environment.split("-", 1)[-1], "Seed": str(seed), "Run URL": ""})
            rows.append(row)
            matches = [row]
        for row in matches:
            if "env_name" in values:
                row["Environment"] = values["env_name"]
                row["Game"] = values["env_name"].split("-", 1)[-1]
            if "seed" in values:
                row["Seed"] = str(values["seed"])
            if "retrieval_enabled" in values:
                enabled = values["retrieval_enabled"]
                label = "Retrieval ON" if enabled else "Retrieval OFF"
                if enabled:
                    for prefix in ("Retrieval ON", "Retrieval Unknown"):
                        if row.get("Config", "").startswith(prefix):
                            label += row["Config"][len(prefix):]
                            break
                row["Retrieval Enable"] = str(enabled)
                row["Config"] = label
            if "score" in values:
                row[SCORE_KEY] = row["Eval Return"] = number(values["score"])
                if row.get("State") != "running":
                    row["State"] = "finished"
                row.update({"Score Status": "ok", "Error": "",
                            "Score Source": "override: " + values.get("source", "user-confirmed evaluation")})
                for key in ("steps", "episodes", "num_envs", "model_step", "action_step", "checkpoint"):
                    # Do not attribute metadata from an earlier evaluation to the new score.
                    row["Evaluation-final/" + key] = values.get(key, "N/A")
    # Running takes precedence over scores and shared-warmup classification,
    # including when converting a CSV exported before this status was introduced.
    for row in rows:
        if row.get("State") == "running":
            row["Score Status"] = "running"
    return rows


def classify_run(run, history_fallback=True, config_overrides=None):
    config = {**(run.config or {}), **(config_overrides or {})}
    summary = run.summary
    name = str(run.name or "")
    environment, seed, retrieval = parse_identity(name, config)
    if seed is None:
        try:
            seed = seed_from_metadata(run)
        except Exception as error:
            print(f"Warning: {run.id}: seed metadata unavailable ({type(error).__name__}).",
                  file=sys.stderr)
    retrieval_config = {
        key: config_value(config, "retrieval." + key)
        for key in RETRIEVAL_FIELDS
    }
    # Keep other recorded retrieval settings in the grouping label as well.
    recorded = config.get("retrieval")
    details = dict(recorded) if isinstance(recorded, dict) else {}
    for key, value in config.items():
        if key.startswith("retrieval."):
            details[key[len("retrieval."):]] = value
    label = {"False": "Retrieval OFF", "True": "Retrieval ON",
             "Both": "Shared warmup", "Unknown": "Retrieval Unknown"}[retrieval]
    if retrieval != "False" and details:
        label += " " + json.dumps(details, sort_keys=True, ensure_ascii=False)
    row = {
        "Run Name": name, "Run ID": run.id, "State": run.state,
        "Game": environment.split("-", 1)[-1], "Environment": environment,
        "Seed": seed if seed is not None else "N/A",
        "Retrieval Enable": retrieval, "Config": label,
        **{column: retrieval_config[key] for key, column in RETRIEVAL_FIELDS.items()},
        **{key: metric_value(summary, key) for key in FINAL_KEYS},
        "Score Source": "", "Score Status": "missing", "Error": "",
        "Commit": getattr(run, "commit", None), "Created At": run.created_at,
        "Run URL": run.url,
    }
    score = number((config_overrides or {}).get("score", row[SCORE_KEY]))
    if score is not None:
        row["Score Source"] = "summary"
    elif run.state == "running":
        # An active run has not reached its final result; do not scan history.
        row["Score Status"] = "running"
    elif retrieval == "Both":
        row["Score Status"] = "warmup"
    elif history_fallback:
        try:
            # scan_history does not sample; requesting only this key also handles
            # sparse logs where other final metrics were written separately.
            for entry in run.scan_history(keys=[SCORE_KEY], page_size=1000):
                candidate = number(entry.get(SCORE_KEY))
                if candidate is not None:
                    score = candidate
            if score is not None:
                row["Score Source"] = "history"
        except Exception as error:
            # A partially downloaded history cannot establish the final value.
            score = None
            row["Score Status"] = "error"
            row["Error"] = f"{type(error).__name__}: {error}"
    row[SCORE_KEY] = row["Eval Return"] = score if score is not None else "N/A"
    if score is not None:
        row["Score Status"] = "ok"
    apply_run_overrides([row], {run.id: config_overrides or {}}, add_missing=False)
    return {key: "N/A" if value is None else value for key, value in row.items()}


def classify_runs(runs, workers=100, history_fallback=True, run_overrides=None):
    """Fetch run details concurrently, retaining the input order in the CSV."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    overrides_by_id = run_overrides or {}
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Only this thread iterates W&B's paginator; each worker owns one run.
        futures = {
            executor.submit(
                classify_run, run, history_fallback=history_fallback,
                config_overrides=overrides_by_id.get(run.id, overrides_by_id.get(run.name)),
            ): index
            for index, run in enumerate(runs)
        }
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows[futures[future]] = row
            print(f"[{completed}/{total}] {row['Run Name']}: {row[SCORE_KEY]} "
                  f"({row['Score Status']}, {row['State']})", flush=True)
    return [rows[index] for index in range(total)]


def write_csv(rows, output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or "choemj-kaist")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT") or "nnet")
    parser.add_argument("--base-url", default=os.environ.get("WANDB_BASE_URL"),
                        help="W&B API endpoint; normally leave unset (not the Forge UI URL)")
    parser.add_argument("--output", type=Path, default=HERE / "wandb_runs_classification.csv")
    parser.add_argument("--xlsx", nargs="?", type=Path, const=HERE / "performance_twister.xlsx",
                        help="Also create Excel; optionally supply its path")
    parser.add_argument("--run-id", action="append", help="Export only these run IDs (repeatable)")
    parser.add_argument("--states", nargs="+", help="For example: --states finished running")
    parser.add_argument("--name-regex", default=r"^callbacks/.*(?:atari100k|dmc)-",
                        help="W&B displayName regex; use '' to include every project run")
    parser.add_argument("--no-history-fallback", action="store_true")
    parser.add_argument("--run-overrides", type=Path,
                        help="Run-ID/name config and score corrections; defaults to results/run_overrides.json")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=100,
                        help="Concurrent run lookups (default: 100; use 1 for sequential processing)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        import wandb
        if not hasattr(wandb, "Api"):
            raise ImportError("The installed wandb SDK could not be imported")
    except ImportError as error:
        parser.exit(1, f"{error}. Install: python -m pip install -r {HERE / 'requirements.txt'}\n")
    try:
        overrides_by_id = load_run_overrides(args.run_overrides)
        # Api reuses WANDB_API_KEY / saved login. No wandb.init or cloud writes.
        overrides = {"base_url": args.base_url} if args.base_url else {}
        api = wandb.Api(overrides=overrides, timeout=args.timeout)
        project = f"{args.entity}/{args.project}"
        filters = {}
        if args.name_regex:
            filters["displayName"] = {"$regex": args.name_regex}
        if args.states:
            filters["state"] = {"$in": args.states}
        if args.run_id:
            filters["name"] = {"$in": args.run_id}
        print(f"Reading {project} ({SCORE_KEY}); up to {args.workers} workers", flush=True)
        rows = classify_runs(
            api.runs(project, filters=filters, order="+created_at", per_page=100),
            workers=args.workers, history_fallback=not args.no_history_fallback,
            run_overrides=overrides_by_id,
        )
        if not rows:
            raise ValueError("No matching runs. Check entity/project, --run-id and --name-regex.")
        output = write_csv(rows, args.output)
        print(f"CSV: {output}\nScores: {dict(Counter(row['Score Status'] for row in rows))}")
        if args.xlsx is not None:
            from convert_csv_to_excel import convert_csv_to_excel
            convert_csv_to_excel(output, args.xlsx, args.run_overrides)
        return 1 if any(row["Score Status"] == "error" for row in rows) else 0
    except Exception as error:
        print(f"Export failed ({type(error).__name__}): {error}", file=sys.stderr)
        print("Check W&B credentials (wandb login), project access and WANDB_BASE_URL.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
