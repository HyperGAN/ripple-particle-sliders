"""Bound one newly spawned owned CPU probe; never signal a foreign process."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cap-gib', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('explicit owned command required')
    args.out.mkdir(parents=True, exist_ok=False)
    cap, reserve, deadline = args.cap_gib*1024**3, 47*1024**3, 600
    available = psutil.virtual_memory().available
    report = dict(schema='ripple-owned-cpu-guard-v1', cap_bytes=cap,
        reserve_bytes=reserve, deadline_seconds=deadline, initial_available_bytes=available,
        gpu_started=False, foreign_processes_signaled=False, command=command)
    if available < reserve+cap:
        report.update(status='REFUSED_BEFORE_SPAWN', reason='fresh full host window unavailable')
        (args.out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report)); raise SystemExit(2)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    start, peak, minimum, reason = time.monotonic(), 0, available, None
    with (args.out/'worker.log').open('x') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        process = psutil.Process(child.pid); birth = process.create_time()
        while child.poll() is None:
            try:
                if process.create_time() != birth:
                    reason = 'owned process identity changed'; break
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
            except psutil.NoSuchProcess:
                continue
            available = psutil.virtual_memory().available
            peak, minimum = max(peak, rss), min(minimum, available)
            if rss > cap: reason = 'own RSS cap exceeded'
            elif available < reserve+max(0,cap-rss): reason = 'future host window no longer fits'
            elif time.monotonic()-start > deadline: reason = 'own deadline exceeded'
            if reason: break
            time.sleep(.2)
        if reason and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
        code = child.wait()
    report.update(status='COMPLETED' if code == 0 and reason is None else 'FAILED',
        reason=reason, exit_code=code, child_reaped=True, peak_owned_rss_bytes=peak,
        minimum_available_bytes=minimum, elapsed_seconds=time.monotonic()-start)
    (args.out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report)); raise SystemExit(0 if report['status']=='COMPLETED' else 1)


if __name__ == '__main__': main()
