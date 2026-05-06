"""Host CPU and memory stats — read straight from /proc.

Docker doesn't isolate /proc by default, so this works from inside the
backend container without any extra mount or privilege. Memory is a one-
shot read of /proc/meminfo; CPU% requires two samples of /proc/stat to
compute the busy/total delta.

A single SystemMonitor instance carries the previous /proc/stat snapshot
between calls so the percentage is meaningful — first call returns 0.0,
subsequent calls reflect the busy ratio over the interval.
"""

from __future__ import annotations

from threading import Lock
from typing import Any


def _read_meminfo() -> dict[str, int]:
    """Return /proc/meminfo as a {key: kB} dict."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                # Values look like "1234 kB" — we strip the unit.
                val = rest.strip().split()
                if val:
                    try:
                        out[k.strip()] = int(val[0])
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _read_cpu_total_busy() -> tuple[int, int]:
    """Return (total_jiffies, busy_jiffies) from /proc/stat's aggregate cpu line.

    /proc/stat's first line is `cpu user nice system idle iowait irq softirq …`.
    busy = everything that isn't idle/iowait; total = sum of all fields.
    """
    try:
        with open("/proc/stat") as f:
            head = f.readline()
    except OSError:
        return 0, 0
    parts = head.split()
    if not parts or parts[0] != "cpu":
        return 0, 0
    nums = [int(x) for x in parts[1:] if x.isdigit()]
    total = sum(nums)
    # Fields: user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    idle = nums[3] if len(nums) > 3 else 0
    iowait = nums[4] if len(nums) > 4 else 0
    busy = total - (idle + iowait)
    return total, busy


class SystemMonitor:
    def __init__(self) -> None:
        self._lock = Lock()
        self._prev_total: int | None = None
        self._prev_busy: int | None = None

    def read(self) -> dict[str, Any]:
        meminfo = _read_meminfo()
        total_kb = meminfo.get("MemTotal", 0)
        # MemAvailable is the kernel's estimate of "what a process could allocate
        # without triggering swap" — closer to user expectations than free+buffers.
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        used_kb = max(0, total_kb - avail_kb)
        mem_percent = round(100.0 * used_kb / total_kb, 1) if total_kb else 0.0

        with self._lock:
            total, busy = _read_cpu_total_busy()
            cpu_percent = 0.0
            if self._prev_total is not None and self._prev_busy is not None:
                d_total = max(1, total - self._prev_total)
                d_busy = max(0, busy - self._prev_busy)
                cpu_percent = round(100.0 * d_busy / d_total, 1)
            self._prev_total, self._prev_busy = total, busy

        return {
            "available": True,
            "cpu_percent": cpu_percent,
            "mem_used_bytes": used_kb * 1024,
            "mem_total_bytes": total_kb * 1024,
            "mem_percent": mem_percent,
        }


def sample_container_peak_memory(container, stop_event, on_sample=None) -> int:
    """Sidecar sampler — polls a Docker container's memory until stop_event.

    Returns the peak memory_stats.usage seen, in bytes. Designed to run in a
    worker thread alongside an autotune bench step; the caller passes
    threading.Event() and sets it when the bench exits.
    """
    peak = 0
    while not stop_event.is_set():
        try:
            stats = container.stats(stream=False)
            mem = stats.get("memory_stats", {}).get("usage")
            if isinstance(mem, int) and mem > peak:
                peak = mem
                if on_sample:
                    try: on_sample(peak)
                    except Exception: pass
        except Exception:
            pass
        # Sample at 1 Hz — finer is wasteful (docker stats is ~100ms per call).
        stop_event.wait(1.0)
    return peak
