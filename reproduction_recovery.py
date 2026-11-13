"""Validate saved OFF starting points without creating a CUDA context."""

from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys


@lru_cache(maxsize=32)
def checkpoint_metadata(python, checkpoint, modified, size, project_root):
    # Read on CPU in a short-lived process. Use mmap when installed PyTorch
    # supports it; only small replay/RNG/optimizer metadata is returned.
    script = """
import inspect, json, pathlib, sys, torch
sys.path[:0] = [sys.argv[2], str(pathlib.Path(sys.argv[2]).parent)]
options = dict(map_location='cpu', weights_only=False)
if 'mmap' in inspect.signature(torch.load).parameters:
    options['mmap'] = True
saved = torch.load(sys.argv[1], **options)
replay = saved.get('replay_buffer_state_dict')
if (not isinstance(replay, dict) or 'buffer_keys' not in replay
        or saved.get('optimizer_state_dict') is None
        or 'model_state_dict' not in saved
        or not saved.get('retrieval_run_state', {}).get('rng')):
    raise ValueError('Checkpoint lacks model, optimizer, replay or RNG state')
print(json.dumps(dict(source_dirs=replay.get('source_dirs'),
                     source_files=replay.get('source_files'),
                     model_step=int(saved['model_step']))))
"""
    result = subprocess.run([python, "-c", script, checkpoint, project_root], capture_output=True,
                            text=True, timeout=60)
    if result.returncode:
        raise ValueError("Cannot inspect saved checkpoint: " + result.stderr.strip().splitlines()[-1])
    return json.loads(result.stdout.strip().splitlines()[-1])


def baseline_source(state, trial):
    import reproduce as r

    directory, run_name, paths = r.trial_paths(state, trial)
    marker = directory / r.MARKER
    if not marker.is_file() or json.loads(marker.read_text()) != r.identity(state, trial):
        raise ValueError("Original trial directory/ownership marker is missing")

    # A completed full-budget OFF warmup can be evaluated directly.
    candidate = trial.get("warmup_fallback", {}).get("baseline_candidate")
    candidate_file = paths["warmup"] / "baseline_candidate.json"
    if candidate is None and candidate_file.is_file():
        candidate = json.loads(candidate_file.read_text())
    if candidate is not None:
        checkpoint = r.validate_baseline_candidate(candidate, state, trial)
        return dict(mode="evaluate_warmup", checkpoint=str(checkpoint), candidate=candidate)
    if trial.get("warmup_fallback"):
        raise ValueError("No saved compatible OFF checkpoint; historical recovery does not retrain from step zero")

    shared = paths["warmup"] / "shared_warmup"
    plan = json.loads((shared / "run.json").read_text())
    if (plan.get("version") != 1 or plan.get("env_name") != "atari100k-" + trial["game"]
            or plan.get("run_name") != run_name
            or plan.get("override_config", {}).get("seed") != trial["seed"]
            or not isinstance(plan.get("cli"), dict)):
        raise ValueError("Saved warmup identity/branch settings do not match")
    config = Path(plan["config_file"])
    if not config.is_absolute():
        config = r.ROOT / config
    if not config.is_file():
        raise ValueError("Original model configuration is missing")
    checkpoint = (shared / plan["checkpoint"]).resolve()
    if checkpoint.parent != shared.resolve() or not checkpoint.is_file():
        raise ValueError("Shared warmup checkpoint is missing")
    stamp = checkpoint.stat()
    metadata = checkpoint_metadata(state["config"].get("python", sys.executable), str(checkpoint),
                                   stamp.st_mtime_ns, stamp.st_size, str(r.ROOT))
    if metadata["model_step"] != plan.get("warmup_model_steps"):
        raise ValueError("Warmup step metadata does not match its checkpoint")
    sources = metadata["source_files"] if metadata["source_dirs"] is not None else None
    if sources is None:
        directories = metadata["source_dirs"]
        if directories is None:
            directories = [str(paths["warmup"] / "ReplayBuffer")]
        sources = [str(path) for folder in directories for path in Path(folder).glob("*.torch")]
    if not sources or any(not Path(path).is_file() for path in sources):
        raise ValueError("Replay files required by the saved warmup are missing")
    return dict(mode="shared_warmup", checkpoint=str(checkpoint), shared=str(shared),
                replay_files=sources)


def available_baseline_source(state, trial):
    try:
        return baseline_source(state, trial), None
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        return None, str(error)
