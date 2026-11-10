"""Entry-point configuration and process launching for TWISTER's shared warmup."""

import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training_branches import launch_training_branches
from configuration import load_run_config, set_run_config, validate_seed


WARMUP_UNAVAILABLE_EXIT_CODE = 75


class WarmupBoundaryUnavailable(RuntimeError):
    """No shared episode boundary can be used within the training budget."""


class SharedWarmupComplete(BaseException):
    """A successful training boundary, deliberately not a training error."""

    def __init__(self, directory):
        self.directory = Path(directory)


def prepare_retrieval_run(args):
    branch = getattr(args, "retrieval_branch", None)
    shared = getattr(args, "shared_warmup", None)
    warmup_only = getattr(args, "warmup_only", False)
    if warmup_only and (branch or args.mode != "training"):
        raise ValueError('--warmup_only requires shared-warmup training without --retrieval_branch')
    if bool(branch) != bool(shared):
        raise ValueError("--retrieval_branch and --shared_warmup must be used together")
    if branch:
        if args.mode != "training" or args.checkpoint is not None or args.load_last:
            raise ValueError("Shared-warmup branches require training without --checkpoint/--load_last")
        directory = Path(shared).resolve()
        plan = json.loads((directory / "run.json").read_text())
        if plan.get("version") != 1:
            raise ValueError("Unsupported TWISTER shared-warmup version")
        override = dict(plan["override_config"], retrieval_enabled=branch == "on")
        seed = getattr(args, "seed", None)
        if seed is not None:
            validate_seed(seed)
            if seed != override.get("seed"):
                raise ValueError("--seed must match the saved shared-warmup seed; "
                                 "branches restore the checkpoint RNG state")
        set_run_config(override)
        os.environ["env_name"] = plan["env_name"]
        os.environ["run_name"] = plan["run_name"] + ("_O" if branch == "on" else "_X")
        args.config_file = plan["config_file"]
        args.checkpoint = str(directory / plan["checkpoint"])
        return plan

    override = load_run_config(seed=getattr(args, "seed", None))
    set_run_config(override)
    if not os.environ.get("run_name") and override.get("seed") is not None:
        game_name = os.environ["env_name"].split("-", 1)[-1]
        os.environ["run_name"] = f"{game_name}_seed{override['seed']}"
    if override.get("retrieval_enabled") != "Both":
        if warmup_only:
            raise ValueError('--warmup_only requires retrieval_enabled="Both"')
        return None
    if args.mode != "training":
        raise ValueError('retrieval_enabled="Both" is a training mode; evaluate an ON/OFF branch instead')
    run_name = os.environ.get("run_name") or "twister_retrieval"
    plan = dict(version=1, run_name=run_name, env_name=os.environ["env_name"],
                override_config=override, config_file=args.config_file,
                cli={key: value for key, value in vars(args).items()
                     if key not in ("checkpoint", "load_last", "retrieval_branch", "shared_warmup", "warmup_only")})
    os.environ["run_name"] = run_name + "_warmup"
    return plan


def branch_commands(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "run.json").read_text())
    command = [sys.executable, str(Path(__file__).resolve().with_name("main.py"))]
    for key, value in plan["cli"].items():
        if key == "wandb_relogin":
            # The parent already authenticated; children reuse its saved credentials.
            continue
        if key == "wandb" and value is False:
            command.append("--no-wandb")
        elif value is True:
            command.append("--" + key)
        elif value is not None and value is not False:
            command.extend(("--" + key, str(value)))
    command.extend(("--shared_warmup", str(directory)))
    return (command + ["--retrieval_branch", "on"], command + ["--retrieval_branch", "off"])


def close_environments(model):
    environments = list(model.env.envs) + ([model.env_eval] if model.env_eval is not None else [])
    for env in environments:
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            close = getattr(env, "close", None)
            if callable(close):
                close()
                break
            env = getattr(env, "env", None)


def fit_retrieval_run(model, args, fit_kwargs):
    run = model.retrieval_run
    if run is None:
        return model.fit(**fit_kwargs)
    if fit_kwargs.get("accumulated_steps", 1) != 1:
        raise ValueError("TWISTER shared-warmup runs require accumulated_steps=1")
    completed = None
    try:
        model.fit(**fit_kwargs)
        if run.shared:
            if not run.target_reached:
                raise ValueError("The warmup decision target exceeds the available training budget")
            run.export_baseline_candidate()
            raise WarmupBoundaryUnavailable(
                "Training budget ended before all warmup episodes finished. "
                "Run retrieval ON from the initial seed; a compatible completed warmup can serve as OFF.")
    except SharedWarmupComplete as event:
        completed = event.directory
    finally:
        close_environments(model)
    if completed is not None:
        if getattr(args, "warmup_only", False):
            return completed
        enabled, disabled = branch_commands(completed)
        launch_training_branches(completed, enabled, disabled)
        raise RuntimeError("The branch supervisor unexpectedly returned")
