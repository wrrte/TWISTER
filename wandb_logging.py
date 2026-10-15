"""Authenticate before model setup and make the W&B destination visible."""

import os
from pathlib import Path

import wandb


def _load_settings():
    # Read the SDK's settings without starting its background service before login.
    config_dir = Path(os.environ.get("WANDB_CONFIG_DIR", "~/.config/wandb")).expanduser()
    settings = wandb.Settings(settings_system=str(config_dir / "settings"))
    settings.update_from_system_settings()
    settings.update_from_env_vars(os.environ)
    return settings


def login_wandb(enabled=True, project=None, entity=None, relogin=False):
    if not enabled:
        print("W&B logging disabled (--no-wandb).", flush=True)
        return

    settings = _load_settings()
    project = project or settings.project or "nnet"
    entity = entity or settings.entity or "<account default>"
    print("W&B startup: mode={}, project={}, entity={}".format(
        settings.mode, project, entity), flush=True)
    if settings.mode in ("offline", "disabled"):
        print("W&B login skipped: mode={} does not upload logs.".format(settings.mode), flush=True)
        return

    print("W&B: verifying login before model initialization; "
          "credentials will be requested if needed.", flush=True)
    try:
        if not wandb.login(force=True, verify=True, relogin=True if relogin else None):
            raise RuntimeError("W&B login did not complete.")
    except Exception as error:
        raise RuntimeError(
            "W&B login failed before model initialization. Log in interactively "
            "or set WANDB_API_KEY for unattended jobs; use --no-wandb to train without W&B."
        ) from error
    print("W&B login verified.", flush=True)


def init_wandb_run(callback_path, project=None, entity=None):
    settings = _load_settings()
    run = wandb.init(
        project=project or settings.project or "nnet",
        entity=entity or settings.entity,
        sync_tensorboard=True,
        name=callback_path,
    )
    print("W&B run: mode={}, entity={}, project={}, name={}".format(
        run.settings.mode, run.entity, run.project, run.name), flush=True)
    if run.settings.mode == "online":
        print("W&B project URL: {}".format(run.project_url), flush=True)
        print("W&B run URL: {}".format(run.url), flush=True)
    elif run.settings.mode == "offline":
        print("W&B logs are not being uploaded (mode={}); local directory: {}".format(
            run.settings.mode, run.dir), flush=True)
    else:
        print("W&B logging disabled; W&B does not save or upload this run.", flush=True)
    return run
