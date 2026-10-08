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


def load_run_config():
    defaults = _json_object(DEFAULTS_PATH.read_text(encoding="utf-8"), str(DEFAULTS_PATH))
    override = _json_object(os.environ.get("override_config", "{}"), "override_config")
    return _merge(defaults, override)


def set_run_config(config):
    """Freeze effective settings before model construction, including saved branches."""
    global _resolved_config
    _resolved_config = copy.deepcopy(config)
    # Custom Python configs can continue reading the existing environment API.
    os.environ["override_config"] = json.dumps(config)


def get_run_config():
    # Direct imports of configs.twister also support defaults + sparse overrides.
    return copy.deepcopy(_resolved_config) if _resolved_config is not None else load_run_config()
