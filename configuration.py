"""Editable run defaults with recursive command-line environment overrides."""

import copy
import json
import os
from pathlib import Path


DEFAULTS_PATH = Path(__file__).resolve().parent / "configs" / "defaults.json"
_resolved_config = None


def _json_object(text, origin):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{origin} must contain a JSON object")
    return value


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_seed(seed):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
        raise ValueError("seed must be null or an integer between 0 and 4294967295")
    return seed


def load_run_config(seed=None):
    defaults = _json_object(DEFAULTS_PATH.read_text(encoding="utf-8"), str(DEFAULTS_PATH))
    override = _json_object(os.environ.get("override_config", "{}"), "override_config")
    config = _merge(defaults, override)
    if seed is not None:
        config["seed"] = seed
    validate_seed(config.get("seed"))
    return config


def set_run_config(config):
    """Freeze effective settings before model construction, including saved branches."""
    global _resolved_config
    validate_seed(config.get("seed"))
    _resolved_config = copy.deepcopy(config)
    # Custom Python configs can continue reading the existing environment API.
    os.environ["override_config"] = json.dumps(config)


def get_run_config():
    # Direct imports of configs.twister also support defaults + sparse overrides.
    return copy.deepcopy(_resolved_config) if _resolved_config is not None else load_run_config()
