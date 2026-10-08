"""Entry-point configuration and process launching for TWISTER's shared warmup."""

import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training_branches import launch_training_branches


class SharedWarmupComplete(BaseException):
    """A successful training boundary, deliberately not a training error."""

    def __init__(self, directory):
        self.directory = Path(directory)


def prepare_retrieval_run(args):
    branch = getattr(args, "retrieval_branch", None)
    shared = getattr(args, "shared_warmup", None)
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
        os.environ["override_config"] = json.dumps(override)
        os.environ["env_name"] = plan["env_name"]
        os.environ["run_name"] = plan["run_name"] + ("_O" if branch == "on" else "_X")
        args.config_file = plan["config_file"]
        args.checkpoint = str(directory / plan["checkpoint"])
        return plan

    override = json.loads(os.environ.get("override_config", "{}"))
    if override.get("retrieval_enabled") != "Both":
        return None
    if args.mode != "training":
        raise ValueError('retrieval_enabled="Both" is a training mode; evaluate an ON/OFF branch instead')
    run_name = os.environ.get("run_name") or "twister_retrieval"
    plan = dict(version=1, run_name=run_name, env_name=os.environ["env_name"],
                override_config=override, config_file=args.config_file,
                cli={key: value for key, value in vars(args).items()
                     if key not in ("checkpoint", "load_last", "retrieval_branch", "shared_warmup")})
    os.environ["run_name"] = run_name + "_warmup"
    return plan


def branch_commands(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "run.json").read_text())
    command = [sys.executable, str(Path(__file__).resolve().with_name("main.py"))]
    for key, value in plan["cli"].items():
        if value is True:
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
            raise RuntimeError("Training budget ended before all warmup episodes finished; "
                               "increase epochs or lower retrieval.warmup_steps. No branches were launched.")
        # Always retain the completed branch, even between periodic saves.
        model.save(str(Path(args.config.callback_path) /
                       f"checkpoints_epoch_{model.config.epochs}_step_{int(model.model_step)}.ckpt"))
    except SharedWarmupComplete as event:
        completed = event.directory
    finally:
        close_environments(model)
        if args.wandb:
            import wandb
            wandb.finish()
    if completed is not None:
        enabled, disabled = branch_commands(completed)
        launch_training_branches(completed, enabled, disabled)
        raise RuntimeError("The branch supervisor unexpectedly returned")
