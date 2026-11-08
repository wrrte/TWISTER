"""Read GPU identity and probe CUDA in a short-lived process before training."""

import csv
import io
import json
import os
import subprocess
import sys


def query_devices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("nvidia-smi GPU query timed out") from error
    if result.returncode:
        raise RuntimeError("GPU query failed: " + (result.stderr.strip() or result.stdout.strip()))
    devices = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) == 3:
            devices.append(dict(index=row[0].strip(), name=row[1].strip(), uuid=row[2].strip()))
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return devices


def select_device(devices, selected):
    if not selected or "," in selected:
        raise ValueError("Select exactly one GPU with CUDA_VISIBLE_DEVICES (the N_train.sh scripts do this)")
    matches = [device for device in devices if selected in (device["index"], device["uuid"])]
    if len(matches) != 1:
        raise ValueError(f"GPU {selected!r} was not found in nvidia-smi")
    return matches[0]


def probe_torch(selected):
    # The supervisor itself must not retain an initialized CUDA context while
    # its training subprocess uses the same GPU.
    command = [sys.executable, "-c", (
        "import json, torch; "
        "available=torch.cuda.is_available(); "
        "print(json.dumps(dict(version=torch.__version__, available=available, "
        "count=torch.cuda.device_count(), name=torch.cuda.get_device_name(0) if available else None)))"
    )]
    try:
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=selected),
                                capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("PyTorch CUDA check timed out") from error
    if result.returncode:
        raise RuntimeError("PyTorch CUDA check failed: " + result.stderr.strip())
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("PyTorch CUDA check returned an invalid result") from error


def check_gpu(expected_model=None, selected=None, devices=None):
    devices = query_devices() if devices is None else devices
    selected = os.environ.get("CUDA_VISIBLE_DEVICES") if selected is None else selected
    device = select_device(devices, selected)
    if expected_model and expected_model.casefold() not in device["name"].casefold():
        raise ValueError(f"Server expects {expected_model}, but GPU {selected} is {device['name']}")
    torch = probe_torch(device["uuid"])
    if not torch["available"] or torch["count"] != 1:
        raise RuntimeError("PyTorch cannot use exactly one CUDA GPU; training has not started")
    if torch["name"] != device["name"]:
        raise RuntimeError(f"GPU identity mismatch: nvidia-smi={device['name']}, PyTorch={torch['name']}")
    print(f"GPU check OK: physical GPU {device['index']} ({device['name']}) -> cuda:0; "
          f"PyTorch {torch['version']}", flush=True)
    return device, torch
