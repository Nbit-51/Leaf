"""Optional process-local CPU pinning for reproducible comparisons."""
from __future__ import annotations
import os


def validate_worker_cpus(cpus: list[int] | None, threads: int, cpu: int | None = None) -> list[int] | None:
    """Validate an explicit one-thread-per-CPU policy without changing affinity."""
    if cpus is None:
        return None
    if cpu is not None:
        raise ValueError("--worker-cpus and --cpu are mutually exclusive")
    if (type(threads) is not int or not 1 <= threads <= 64 or
            not isinstance(cpus, (list, tuple)) or not cpus or len(cpus) != threads or
            any(type(value) is not int or value < 0 for value in cpus) or len(set(cpus)) != len(cpus)):
        raise ValueError("--worker-cpus requires one distinct nonnegative CPU index per inference thread")
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.GetProcessAffinityMask.restype = ctypes.c_int
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), 'Cannot query process CPU affinity')
        allowed = {value for value in range(ctypes.sizeof(ctypes.c_size_t)*8) if process.value & (1<<value)}
    elif hasattr(os, 'sched_getaffinity'):
        allowed = os.sched_getaffinity(0)
    else:
        raise ValueError('--worker-cpus is supported on Windows and Linux')
    if not set(cpus) <= allowed:
        raise ValueError('--worker-cpus contains an index outside the process allowed CPU set')
    return list(cpus)


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
