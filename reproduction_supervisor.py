#!/usr/bin/env python3
"""Refresh workers between trials, or adopt already stopped legacy workers."""

import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_results import write_json


ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "results/reproduction/supervisors"
WAIT = 20
DONE = 21


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message, flush=True)


def process(pid):
    """Include zombies: a stopped parent cannot reap its completed child."""
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    group=int(fields[2]), session=int(fields[3]), start=int(fields[19]),
                    exit_status=int(fields[49]))
    except FileNotFoundError:
        return None


def same_process(saved):
    current = process(saved["pid"])
    return current if current and current["start"] == saved["start"] else None


def ended(item):
    return item is None or item["state"] in {"Z", "X", "x"}


def command(pid):
    return [value.decode() for value in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if value]


def environment(pid):
    return dict(value.decode().split("=", 1) for value in
                (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in value)


def gpu_snapshot():
    def query(scope, fields):
        result = subprocess.run(["nvidia-smi", "--query-" + scope + "=" + fields,
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError("nvidia-smi failed: " + (result.stderr.strip() or result.stdout.strip()))
        return [[value.strip() for value in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

    devices = {}
    for index, name, uuid, memory in query("gpu", "index,name,uuid,memory.used"):
        devices[int(index)] = dict(index=int(index), name=name, uuid=uuid, memory=int(memory), pids=[])
    by_uuid = {item["uuid"]: item for item in devices.values()}
    for uuid, pid in query("compute-apps", "gpu_uuid,pid"):
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].append(int(pid))
    return devices


def parents_for(device):
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            pid = int(directory.name)
            argv = command(pid)
            positions = [i for i, value in enumerate(argv) if Path(value).name == "reproduce.py"]
            if not positions or directory.stat().st_uid != os.getuid():
                continue
            if (directory / "cwd").resolve() != ROOT.resolve():
                continue
            env = environment(pid)
            if env.get("CUDA_VISIBLE_DEVICES") not in {str(device["index"]), device["uuid"]}:
                continue
            item = process(pid)
            if item is not None:
                matches.append(dict(item, argv=argv[positions[0] + 1:],
                                    python=os.readlink(str(directory / "exe"))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def gpu_lock(device, inherited=None):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / (device["uuid"] + ".lock")
    if inherited is not None:
        lock = os.fdopen(inherited, "a")
        if os.fstat(lock.fileno()).st_ino != path.stat().st_ino:
            lock.close()
            raise ValueError("Inherited supervisor lock does not match the GPU")
    else:
        lock = path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"GPU {device['index']} already has a supervisor or transition in progress")
    return lock


class IdleWindow:
    def __init__(self, limit=1000, seconds=5, interval=1):
        self.limit, self.seconds = limit, seconds
        self.max_gap = max(2.5, interval * 2)
        self.since = self.last = None

    def reset(self):
        self.since = self.last = None

    def observe(self, memory, now):
        if memory >= self.limit or (self.last is not None and now - self.last > self.max_gap):
            self.reset()
        self.last = now
        if memory >= self.limit:
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.seconds


def state_path(arguments):
    import reproduction_cluster as cluster
    for index, argument in enumerate(arguments):
        if argument == "--state":
            return Path(arguments[index + 1]).expanduser().resolve()
        if argument.startswith("--state="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    return Path(cluster.local_settings().get("state", ROOT / "results/reproduction/state.json")).resolve()


def snapshot_job(device, parent):
    import reproduce as r
    if parent["state"] not in {"T", "t"} or "--supervised" in parent["argv"]:
        raise ValueError(f"GPU {device['index']}: expected an already stopped legacy parent")
    campaign = r.Campaign(state_path(parent["argv"]))
    state = campaign.read()
    trials = [item for item in state["trials"] if item.get("worker", "").rsplit(":", 1)[-1] == str(parent["pid"])
              and item["status"] not in {"complete", "cancelled"}]
    if len(trials) != 1:
        raise ValueError(f"GPU {device['index']}: cannot identify exactly one current trial")
    trial = trials[0]
    if trial["stage"] not in {"warmup", "full_flash", "baseline"}:
        raise ValueError("Unsupported transition stage: " + trial["stage"])
    children = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            item = process(int(directory.name))
            if item and item["ppid"] == parent["pid"]:
                children.append(item)
    if len(children) > 1:
        raise ValueError("Legacy worker has multiple direct children; refusing an ambiguous transition")
    child = children[0] if children else None
    if child and child["session"] != child["pid"]:
        raise ValueError("Training child does not have the expected separate session")
    if child and not ended(child) and not any(Path(value).name == "main.py" for value in command(child["pid"])):
        raise ValueError("The active child is not a main.py training process")
    return dict(version=1, gpu=device["index"], uuid=device["uuid"], parent=parent, child=child,
                state_path=str(campaign.path), search_id=state["search_id"], game=trial["game"],
                round=trial["round"], stage=trial["stage"], phase="waiting", created=time.time())


def job_trial(job):
    import reproduce as r
    campaign = r.Campaign(job["state_path"])
    state = campaign.read()
    if state["search_id"] != job["search_id"]:
        raise ValueError("Campaign changed since transition was scheduled")
    trial = next(item for item in state["trials"] if (item["game"], item["round"]) == (job["game"], job["round"]))
    directory = r.artifact_dir(state, trial)
    if json.loads((directory / r.MARKER).read_text()) != r.identity(state, trial):
        raise ValueError("Trial ownership marker does not match")
    return campaign, state, trial


def validate_completion(job):
    import reproduce as r
    child = same_process(job["child"]) if job["child"] else None
    if not ended(child):
        return None
    code = None
    if child:
        status = child["exit_status"]
        if not os.WIFEXITED(status):
            raise ValueError(f"Training child ended from signal {os.WTERMSIG(status)}")
        code = os.WEXITSTATUS(status)
        if code != 0 and not (code == 75 and job["stage"] == "warmup"):
            raise ValueError(f"Training child failed with exit code {code}; worker will not restart")
    campaign, state, trial = job_trial(job)
    _, name, paths = r.trial_paths(state, trial)
    if job["stage"] == "warmup":
        if code == 75:
            fallback = dict(mode="independent", reason="Suspended warmup exited with code 75", time=time.time())
            candidate = paths["warmup"] / "baseline_candidate.json"
            if candidate.is_file():
                fallback["baseline_candidate"] = json.loads(candidate.read_text())
                r.validate_baseline_candidate(fallback["baseline_candidate"], state, trial)
            return dict(warmup_fallback=fallback)
        shared = paths["warmup"] / "shared_warmup"
        plan = json.loads((shared / "run.json").read_text())
        checkpoint = (shared / plan["checkpoint"]).resolve()
        if (plan.get("version") != 1 or plan.get("env_name") != "atari100k-" + trial["game"]
                or plan.get("run_name") != name or plan.get("override_config", {}).get("seed") != trial["seed"]
                or checkpoint.parent != shared.resolve() or not checkpoint.is_file()):
            raise ValueError("Completed shared warmup is invalid")
        return {}
    enabled = job["stage"] == "full_flash"
    result = r.read_evaluation(paths[job["stage"]] / "final_evaluation.json", state, trial, enabled)
    score = result["metrics"]["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Completed evaluation score must be finite")
    if enabled:
        comparison = r.compare_score(score, state["config"]["references"][trial["game"]])
        return dict(full_flash=dict(comparison, evaluation=result))
    if trial["full_flash"] is None:
        raise ValueError("Completed baseline has no full FLASH result")
    return dict(baseline=result, status="complete", stage="done")


def signal_identity(saved, signum):
    # Bind signals to the process itself, even if a PID is recycled meanwhile.
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            descriptor = os.pidfd_open(saved["pid"])
        except ProcessLookupError:
            return
        try:
            if same_process(saved) is None:
                return
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
    elif same_process(saved) is not None:
        os.kill(saved["pid"], signum)


def retire_parent(job, timeout=30):
    parent = same_process(job["parent"])
    if ended(parent):
        return
    if parent["state"] not in {"T", "t"}:
        raise ValueError("Legacy parent was resumed outside the transition tool")
    if job["child"] and not ended(same_process(job["child"])):
        raise ValueError("Training child is still alive")
    # Queue termination while stopped, then let its existing handler clean up.
    # Never CONT alone: the old scheduler must not start another stage.
    signal_identity(job["parent"], signal.SIGTERM)
    signal_identity(job["parent"], signal.SIGCONT)
    deadline = time.monotonic() + timeout
    while not ended(same_process(job["parent"])):
        if time.monotonic() >= deadline:
            raise RuntimeError("Legacy parent did not exit; no replacement was started")
        time.sleep(0.1)


def ingest_completion(job, changes):
    campaign, _, trial = job_trial(job)
    lease = campaign._lease(trial)
    if lease is None:
        raise RuntimeError("Original training lease is still held; no replacement was started")
    try:
        campaign.update(trial, **dict(dict(status="interrupted", stage=job["stage"], error=None,
                                         transition_pending=False), **changes))
    finally:
        lease.close()


def mark_transition(job):
    campaign, _, _ = job_trial(job)
    with campaign.locked():
        state = campaign.read()
        trial = next(item for item in state["trials"]
                     if (item["game"], item["round"]) == (job["game"], job["round"]))
        if trial.get("worker", "").rsplit(":", 1)[-1] != str(job["parent"]["pid"]):
            raise ValueError("Another worker has already claimed the original trial")
        trial["transition_pending"] = True
        write_json(campaign.path, state)


def journal_path(gpu):
    return DIRECTORY / f"gpu{gpu}.transition.json"


def start_replacement(job, device, lock, env):
    arguments = [arg for arg in job["parent"]["argv"] if arg != "--once"]
    output = DIRECTORY / f"gpu{job['gpu']}.log"
    with output.open("a") as stream:
        child = subprocess.Popen([job["parent"]["python"], "-u", str(Path(__file__).resolve()),
                                  "run", "--lock-fd", str(lock.fileno()), "--", *arguments],
                                 cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=device["uuid"]),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock.fileno(),))
    return child.pid, str(output)


def adopt(gpus, limit=1000, seconds=5, interval=1, inspect=False):
    devices = gpu_snapshot()
    jobs, locks, envs = {}, {}, {}
    try:
        for gpu in gpus:
            device = devices[gpu]
            parents = parents_for(device)
            saved_path = journal_path(gpu)
            saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
            if saved and saved["phase"] == "started" and not ended(same_process(saved["replacement"])):
                log(f"GPU {gpu}: replacement supervisor already running, PID {saved['replacement']['pid']}")
                continue
            if len(parents) == 1:
                job = snapshot_job(device, parents[0])
            elif not parents and saved and (saved["phase"] in {"ready", "retiring", "retired"}
                                            or saved["phase"] == "error" and "completion" in saved):
                job = saved
            else:
                raise ValueError(f"GPU {gpu}: expected exactly one stopped parent; found {len(parents)}")
            if job["uuid"] != device["uuid"]:
                raise ValueError("GPU UUID changed since the transition was saved")
            jobs[gpu] = job
            log(f"GPU {gpu}: {job['game']} round {job['round']}, {job['stage']}; "
                f"parent {job['parent']['pid']}, child {job['child']['pid'] if job['child'] else 'already reaped'}")
            if not inspect:
                locks[gpu] = gpu_lock(device)
                envs[gpu] = environment(job["parent"]["pid"]) if same_process(job["parent"]) else dict(os.environ)
        if inspect:
            return 0
        for gpu, job in jobs.items():
            write_json(journal_path(gpu), job)
        windows = {gpu: IdleWindow(limit, seconds, interval) for gpu in jobs}
        failures = 0
        last_report = 0
        while jobs:
            try:
                devices = gpu_snapshot()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                for window in windows.values():
                    window.reset()
                log(f"GPU query unavailable; timers reset, no transition: {error}")
                time.sleep(interval)
                continue
            for gpu, job in list(jobs.items()):
                try:
                    device = devices[gpu]
                    if device["uuid"] != job["uuid"]:
                        raise ValueError("GPU identity changed")
                    parent = same_process(job["parent"])
                    if not ended(parent) and parent["state"] not in {"T", "t"}:
                        raise ValueError("Legacy parent is no longer stopped")
                    idle = windows[gpu].observe(device["memory"], time.monotonic())
                    if not idle or device["pids"]:
                        continue
                    if not job.get("ingested"):
                        changes = job.get("completion")
                        if changes is None:
                            changes = validate_completion(job)
                        if changes is None:
                            continue
                        job.update(phase="ready", completion=changes)
                        write_json(journal_path(gpu), job)
                        mark_transition(job)
                        log(f"GPU {gpu}: memory < {limit} MiB for {seconds:g}s; child result validated, retiring parent")
                        job["phase"] = "retiring"
                        write_json(journal_path(gpu), job)
                        retire_parent(job)
                        ingest_completion(job, changes)
                        job.update(phase="retired", ingested=True)
                        write_json(journal_path(gpu), job)
                    pid, output = start_replacement(job, device, locks[gpu], envs[gpu])
                    replacement = process(pid)
                    if replacement is None:
                        raise RuntimeError("Replacement supervisor exited immediately; see " + output)
                    job.update(phase="started", replacement=replacement, log=output, started=time.time())
                    write_json(journal_path(gpu), job)
                    log(f"GPU {gpu}: latest supervisor PID {pid}, log {output}")
                    locks.pop(gpu).close()
                    del jobs[gpu]
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    failures += 1
                    job.update(phase="error", error=str(error))
                    write_json(journal_path(gpu), job)
                    log(f"GPU {gpu}: transition stopped: {error}")
                    locks.pop(gpu).close()
                    del jobs[gpu]
            if time.monotonic() - last_report >= 30 and jobs:
                log("Waiting: " + "; ".join(f"GPU {gpu} {devices[gpu]['memory']} MiB, "
                    f"child {'finished' if ended(same_process(job['child'])) else 'running'}"
                    for gpu, job in jobs.items() if job["child"]))
                last_report = time.monotonic()
            if jobs:
                time.sleep(interval)
        return 1 if failures else 0
    finally:
        for lock in locks.values():
            lock.close()


def run(arguments, inherited=None):
    # Preserve the one-shot/status/help behavior of the existing train scripts.
    if any(argument in {"--status", "--help", "-h"} for argument in arguments):
        os.execv(sys.executable, [sys.executable, "-u", str(ROOT / "reproduce.py"), *arguments])
    one_shot = "--once" in arguments
    arguments = [argument for argument in arguments if argument != "--once"]
    from gpu_runtime import select_device
    devices = gpu_snapshot()
    device = select_device([dict(item, index=str(item["index"])) for item in devices.values()],
                           os.environ.get("CUDA_VISIBLE_DEVICES"))
    lock = gpu_lock(device, inherited)
    child = None

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if parents_for(device):
            raise RuntimeError("An existing worker still owns this GPU; use adopt for stopped parents")
        while True:
            log(f"GPU {device['index']}: starting a fresh one-trial worker")
            child = subprocess.Popen([sys.executable, "-u", str(ROOT / "reproduce.py"),
                                      *arguments, "--once", "--supervised"], cwd=ROOT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=device["uuid"]),
                                     start_new_session=True)
            code = child.wait()
            child = None
            if code == DONE:
                log(f"GPU {device['index']}: all assigned games complete")
                return 0
            if code == WAIT:
                if one_shot:
                    return 0
                time.sleep(10)
            elif code != 0:
                log(f"GPU {device['index']}: worker failed (exit {code}); automatic retries disabled")
                return code if code > 0 else 128 - code
            elif one_shot:
                return 0
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        return 130
    finally:
        lock.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("run", help="Run a fresh worker for each trial")
    worker.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    worker.add_argument("arguments", nargs=argparse.REMAINDER)
    transition = commands.add_parser("adopt", help="Finish children of already SIGSTOPed legacy workers, then replace")
    transition.add_argument("--gpus", type=int, nargs="+", required=True)
    transition.add_argument("--memory-limit-mib", type=int, default=1000)
    transition.add_argument("--idle-seconds", type=float, default=5)
    transition.add_argument("--poll-seconds", type=float, default=1)
    transition.add_argument("--inspect", action="store_true", help="Only discover stopped parents and their current trials")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments, args.lock_fd)
    if (len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0 or args.memory_limit_mib <= 0
            or not math.isfinite(args.idle_seconds) or args.idle_seconds < 5
            or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 2):
        parser.error("Use distinct GPU indices, a positive memory limit, idle >= 5s and 0 < poll <= 2s")
    return adopt(args.gpus, args.memory_limit_mib, args.idle_seconds, args.poll_seconds, args.inspect)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Transition monitor stopped; waiting legacy parents remain stopped")
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Supervisor failed: {error}", file=sys.stderr)
        raise SystemExit(1)
