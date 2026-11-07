"""Atomic local results, also available when W&B logging is disabled."""

import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_final_evaluation(callback_path, model, metrics, checkpoint):
    if callback_path is None:
        return
    write_json(Path(callback_path) / "final_evaluation.json", {
        "version": 1,
        "env_name": "-".join(model.config.env_name),
        "seed": model.config.seed,
        "retrieval_enabled": model.config.retrieval_enabled,
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "model_step": int(model.model_step),
        "action_step": int(model.action_step),
        "episodes": model.config.final_eval_episodes,
        "num_envs": min(model.config.final_eval_num_envs, model.config.final_eval_episodes),
        "metrics": {key: float(value) for key, value in metrics.items()},
    })
