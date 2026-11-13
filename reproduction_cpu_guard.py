#!/usr/bin/env python3
"""Pause slow local TWISTER training groups and resume them when CPU load falls."""

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

from reproduction_supervisor import command, environment, gpu_snapshot, process, same_process
from run_results import write_json


ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "results/reproduction/cpu_guard"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RATE = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>it/s|s/it)")
COUNTS = re.compile(r"(?P<step>\d+)/(?P<total>\d+)\s*\[")
EPOCH = re.compile(r"\bEpoch\s+\d+/\d+:")


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [CPU guard] " + message, flush=True)


class CpuMeter:
    """System-wide busy percentage, excluding idle and Linux I/O-wait time."""

    def __init__(self):
        self.previous = None

    def sample(self):
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        current = (sum(values), values[3] + values[4])
        previous, self.previous = self.previous, current
        if previous is None or current[0] <= previous[0]:
            return None
        return min(100.0, max(0.0, 100 * (1 - (current[1] - previous[1]) / (current[0] - previous[0]))))


class ProgressLog:
    """Read new tqdm records, including carriage returns and inverse rates."""

    def __init__(self, path):
        self.path = Path(path)
        self.rate = None
        self.training = True
        self.total = None
        self.pending = b""
        with self.path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 1024 * 1024))
            self.feed(stream.read())
            self.offset = stream.tell()
            self.inode = os.fstat(stream.fileno()).st_ino
        # Historical rates must never trigger a pause immediately on attachment.
        self.rate = None

    def feed(self, data):
        records = re.split(b"[\r\n]", self.pending + data)
        self.pending = records.pop()[-16384:]
        changed = False
        for record in records:
            text = ANSI.sub("", record.decode("utf-8", errors="replace"))
            if EPOCH.search(text):
                self.training, self.rate = True, None
                self.total = None
                changed = True
            counts = COUNTS.search(text)
            if not self.training or not counts:
                continue
            total = int(counts["total"])
            if self.total is None:
                self.total = total
            if total != self.total:
                # Step-triggered evaluation can appear inside a training epoch.
                # Ignore its rate without disabling the remaining train steps.
                self.rate = None
                changed = True
                continue
            if int(counts["step"]) >= total:
                self.training, self.rate = False, None
                changed = True
                continue
            matches = list(RATE.finditer(text))
            if matches:
                match = matches[-1]
                rate = float(match["rate"])
                self.rate = rate if match["unit"] == "it/s" else (1 / rate if rate else None)
                changed = True
        return changed

    def poll(self):
        with self.path.open("rb") as stream:
            status = os.fstat(stream.fileno())
            if status.st_ino != self.inode or status.st_size < self.offset:
                self.offset, self.pending, self.rate = 0, b"", None
                self.training = True
                self.total = None
            self.inode = status.st_ino
            # Bound a poll after a long suspension; only recent progress matters.
            if status.st_size - self.offset > 1024 * 1024:
                self.offset = status.st_size - 1024 * 1024
                self.pending = b""
            stream.seek(self.offset)
            data = stream.read(1024 * 1024)
            self.offset = stream.tell()
        return self.feed(data)


@dataclass
class Training:
    identity: dict
    gpu: str
    logfile: Path

    @property
    def key(self):
        return self.identity["pid"], self.identity["start"]


def script_is(argv, name):
    return any(Path(value).name == name for value in argv)


def training_process(pid, require_parent=True):
    """Only our training leaders in this checkout, each in its own session."""
    directory = Path("/proc") / str(pid)
    try:
        if directory.stat().st_uid != os.getuid() or (directory / "cwd").resolve() != ROOT:
            return None
        item = process(pid)
        if not item or item["state"] in {"Z", "X", "x"} or item["group"] != pid or item["session"] != pid:
            return None
        argv = command(pid)
        if not script_is(argv, "main.py"):
            return None
        for index, value in enumerate(argv):
            if value in {"-m", "--mode"} and (index + 1 == len(argv) or argv[index + 1] != "training"):
                return None
            if value.startswith("--mode=") and value != "--mode=training":
                return None
        if require_parent and not script_is(command(item["ppid"]), "reproduce.py"):
            return None
        logfile = (directory / "fd/1").resolve()
        if not logfile.is_relative_to(ROOT) or not logfile.is_file():
            return None
        return Training(item, environment(pid).get("CUDA_VISIBLE_DEVICES", "unknown"), logfile)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def discover(gpus=None):
    jobs = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            job = training_process(int(directory.name))
            if job is not None and (gpus is None or job.gpu in gpus):
                jobs.append(job)
    return jobs


def signal_group(saved, signum):
    current = same_process(saved)
    if not current or current["state"] in {"Z", "X", "x"}:
        return False
    if current["group"] != current["pid"] or current["session"] != current["pid"]:
        raise RuntimeError("Training no longer owns its separate process group")
    try:
        os.killpg(current["group"], signum)
    except ProcessLookupError:
        return False
    return True


@dataclass
class Options:
    min_rate: float = 1.30
    resume_cpu: float = 80.0
    grace_seconds: float = 60.0
    slow_seconds: float = 30.0
    settle_seconds: float = 30.0
    retry_seconds: float = 30.0
    jitter_min: float = 5.0
    jitter_max: float = 30.0
    max_running: int = None
    dry_run: bool = False


class Job:
    def __init__(self, training, now, options):
        self.training = training
        self.progress = ProgressLog(training.logfile)
        self.rate = self.slow_since = self.last_rate_at = None
        self.grace_until = now + options.grace_seconds
        self.paused = False
        self.paused_at = self.retry_at = None


class Guard:
    def __init__(self, options, journal, logger=log):
        self.options, self.journal, self.log = options, Path(journal), logger
        self.jobs = {}
        self.next_action = 0
        self.random = random.SystemRandom()

    def save(self):
        if not self.options.dry_run:
            write_json(self.journal, {"version": 1, "paused": [job.training.identity for job in self.jobs.values() if job.paused]})

    def update(self, trainings, now, cpu):
        current = {training.key: training for training in trainings}
        removed = set(self.jobs) - set(current)
        for key in removed:
            job = self.jobs[key]
            # A parent can exit while a paused child is still alive. Restore it
            # before forgetting it; an exited/reused leader fails the identity check.
            if job.paused and not self.options.dry_run:
                signal_group(job.training.identity, signal.SIGCONT)
            del self.jobs[key]
        if removed:
            self.save()
        for key, training in current.items():
            if key not in self.jobs:
                try:
                    self.jobs[key] = Job(training, now, self.options)
                except FileNotFoundError:
                    continue  # The completed trial may already have been pruned.
                self.log(f"Watching PID {training.identity['pid']} GPU {training.gpu}: {training.logfile}")
            job = self.jobs[key]
            job.training = training
            if job.paused or training.identity["state"] in {"T", "t"}:
                continue
            try:
                if job.progress.path != training.logfile:
                    job.progress = ProgressLog(training.logfile)
                    job.rate = job.slow_since = job.last_rate_at = None
                if job.progress.poll():
                    job.rate, job.last_rate_at = job.progress.rate, now
            except FileNotFoundError:
                job.rate = job.slow_since = job.last_rate_at = None
            fresh = job.last_rate_at is not None and now - job.last_rate_at <= 15
            if now < job.grace_until or not fresh or job.rate is None or job.rate > self.options.min_rate:
                job.slow_since = None
            elif job.slow_since is None:
                job.slow_since = now
        if now < self.next_action:
            return
        active = [job for job in self.jobs.values() if not job.paused and job.progress.training
                  and job.training.identity["state"] not in {"T", "t"}]
        slow = [job for job in active if job.slow_since is not None and now - job.slow_since >= self.options.slow_seconds]
        over_limit = self.options.max_running is not None and len(active) > self.options.max_running
        # Keep at least one managed training active, even if the game is slow
        # intrinsically. Serial decisions prevent every worker yielding at once.
        if len(active) > 1 and (slow or over_limit):
            candidates = slow or active
            job = min(candidates, key=lambda item: item.rate if item.rate is not None else math.inf)
            delay = self.random.uniform(self.options.jitter_min, self.options.jitter_max)
            job.paused, job.paused_at, job.retry_at = True, now, now + delay
            self.save()  # Record ownership before STOP, so a killed guard can recover.
            if not self.options.dry_run and not signal_group(job.training.identity, signal.SIGSTOP):
                job.paused = False
                self.save()
                return
            rate = f"{job.rate:.2f} it/s" if job.rate is not None else "no rate yet"
            self.log(f"{'Would pause' if self.options.dry_run else 'Paused'} PID {job.training.identity['pid']} "
                     f"GPU {job.training.gpu} ({rate}); first CPU check in {delay:.1f}s")
            self.next_action = now + delay
            for running in active:
                running.slow_since = None
            return
        # Jitter controls the first check; failed checks repeat every 30 seconds.
        # At most one resume per settle window, even when several jobs are ready.
        waiting = sorted((job for job in self.jobs.values() if job.paused and now >= job.retry_at),
                         key=lambda item: item.paused_at)
        for job in waiting:
            if cpu is None:
                continue
            if cpu > self.options.resume_cpu or (self.options.max_running is not None and len(active) >= self.options.max_running):
                job.retry_at = now + self.options.retry_seconds
                self.log(f"Waiting PID {job.training.identity['pid']}: CPU {cpu:.1f}%, active={len(active)}; "
                         f"retry in {self.options.retry_seconds:g}s")
                continue
            if not self.options.dry_run:
                signal_group(job.training.identity, signal.SIGCONT)
            job.paused, job.rate, job.slow_since, job.last_rate_at = False, None, None, None
            job.progress.rate = None
            job.grace_until = now + self.options.grace_seconds
            self.save()
            self.next_action = now + self.options.settle_seconds
            self.log(f"{'Would resume' if self.options.dry_run else 'Resumed'} PID {job.training.identity['pid']} "
                     f"GPU {job.training.gpu}: CPU {cpu:.1f}%")
            break

    def close(self):
        errors = []
        for job in self.jobs.values():
            if job.paused:
                try:
                    if not self.options.dry_run:
                        signal_group(job.training.identity, signal.SIGCONT)
                except (OSError, RuntimeError) as error:
                    errors.append(str(error))
                    continue
                job.paused = False
                self.log(f"Released PID {job.training.identity['pid']} on guard exit")
        self.save()
        if errors:
            raise RuntimeError("Could not resume every group; run --recover: " + "; ".join(errors))


def recover(journal):
    if not journal.exists():
        return
    saved = json.loads(journal.read_text())
    if saved.get("version") != 1:
        raise ValueError("Unknown CPU guard journal version")
    for identity in saved["paused"]:
        job = training_process(identity["pid"], require_parent=False)
        if job and job.identity["start"] == identity["start"]:
            signal_group(identity, signal.SIGCONT)
            log(f"Recovered previously paused PID {identity['pid']}")
    write_json(journal, {"version": 1, "paused": []})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", help="Physical GPU indices to manage (default: all our TWISTER reproduction workers)")
    parser.add_argument("--min-rate", type=float, default=1.30, help="Pause at or below this many iterations/s (default: 1.30)")
    parser.add_argument("--resume-cpu", type=float, default=80.0, help="System CPU percentage permitting a resume (default: 80)")
    parser.add_argument("--grace-seconds", type=float, default=60, help="Ignore slow rates after attachment/resume (default: 60)")
    parser.add_argument("--slow-seconds", type=float, default=30, help="Require sustained slow progress (default: 30)")
    parser.add_argument("--max-running", type=int, help="Optional additional cap on managed active trainings")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions without sending signals")
    parser.add_argument("--once", action="store_true", help="Discover/sample once and exit; useful with --dry-run")
    parser.add_argument("--recover", action="store_true", help="Only resume groups left paused by a previously killed guard")
    args = parser.parse_args(argv)
    if (not all(math.isfinite(value) for value in (args.min_rate, args.resume_cpu, args.grace_seconds, args.slow_seconds))
            or args.min_rate <= 0 or not 0 <= args.resume_cpu <= 100
            or args.grace_seconds < 0 or args.slow_seconds < 0
            or (args.max_running is not None and args.max_running < 1)
            or (args.gpus is not None and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)))
            or (args.recover and args.dry_run)):
        parser.error("Use finite positive rate, CPU 0..100, nonnegative durations and distinct nonnegative GPUs; --recover cannot be dry-run")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A CPU guard is already running for this checkout")
        journal = DIRECTORY / "paused.json"
        if not args.dry_run:
            recover(journal)
        if args.recover:
            return 0
        gpus = None
        if args.gpus is not None:
            devices = gpu_snapshot()
            if any(index not in devices for index in args.gpus):
                parser.error("Unknown GPU index")
            gpus = {str(index) for index in args.gpus} | {devices[index]["uuid"] for index in args.gpus}
        options = Options(min_rate=args.min_rate, resume_cpu=args.resume_cpu,
                          grace_seconds=args.grace_seconds, slow_seconds=args.slow_seconds,
                          max_running=args.max_running, dry_run=args.dry_run)
        guard, meter = Guard(options, journal), CpuMeter()

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        log(f"Started: slow <= {options.min_rate:g} it/s for {options.slow_seconds:g}s, "
            f"resume CPU <= {options.resume_cpu:g}%, dry_run={options.dry_run}")
        try:
            meter.sample()
            time.sleep(1)
            last_report = None
            while True:
                now, cpu = time.monotonic(), meter.sample()
                guard.update(discover(gpus), now, cpu)
                if last_report is None or now - last_report >= 30:
                    paused = sum(job.paused for job in guard.jobs.values())
                    cpu_text = f"{cpu:.1f}%" if cpu is not None else "unavailable"
                    log(f"CPU {cpu_text}; watching={len(guard.jobs)}, guard-paused={paused}")
                    last_report = now
                if args.once:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            log("Stopping; resuming groups paused by this guard")
        finally:
            guard.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
