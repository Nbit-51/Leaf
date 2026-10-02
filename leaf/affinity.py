"""Optional process-local CPU pinning for reproducible comparisons."""
from __future__ import annotations
import os


def pin_cpu(cpu: int | None) -> None:
    if cpu is None:
        return
    if cpu < 0 or cpu >= (os.cpu_count() or 1):
        raise ValueError("CPU index is outside this machine's logical CPUs")
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if cpu >= ctypes.sizeof(ctypes.c_size_t) * 8:
            raise ValueError("CPU pinning beyond one Windows processor group is not supported")
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise OSError(ctypes.get_last_error(), "Unable to set process CPU affinity")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    else:
        raise ValueError("CPU pinning is not supported by this operating system")
