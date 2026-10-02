"""Scoped, explicitly requested benchmark-only Windows power controls."""
from contextlib import contextmanager
import ctypes
import os
import subprocess


class _PowerThrottlingState(ctypes.Structure):
    # Windows ULONG is 32 bits even on a 64-bit process. Do not use c_ulong,
    # whose width differs on non-Windows hosts running the mocked unit tests.
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32),
                ("StateMask", ctypes.c_uint32)]


def set_child_high_qos(process: subprocess.Popen) -> dict:
    """Opt one newly spawned benchmark child into verified Windows HighQoS.

    This is not a CLI default, priority adjustment, or system power-plan change.
    The original masks are queried and only EXECUTION_SPEED throttling is
    disabled; other power-throttling mechanisms retain their previous state.
    Unsupported/failed requests raise rather than silently qualifying a run.
    The policy ends with this child process. On failure, the spawning caller
    must terminate/wait its child as in any failed benchmark setup.

    Microsoft documents HighQoS as ControlMask=EXECUTION_SPEED, StateMask=0:
    https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
    """
    if os.name != "nt":
        raise RuntimeError("Child HighQoS is Windows-only; no process policy was applied")
    if not isinstance(process, subprocess.Popen) or not getattr(process, "_child_created", False):
        raise ValueError("HighQoS requires a newly spawned subprocess.Popen child")
    pid, handle = getattr(process, "pid", None), getattr(process, "_handle", None)
    pointer_limit = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid == os.getpid() or
            not isinstance(handle, int) or isinstance(handle, bool) or not 0 < handle < pointer_limit or
            getattr(handle, "closed", False)):
        raise ValueError("HighQoS requires a live, valid child process handle and PID")
    if process.poll() is not None:
        raise ValueError("HighQoS cannot be applied to an exited child process")
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get_information = kernel.GetProcessInformation
        set_information = kernel.SetProcessInformation
        get_process_id = kernel.GetProcessId
    except (AttributeError, OSError) as error:
        raise RuntimeError("Windows child HighQoS process-information APIs are unavailable") from error
    for function in (get_information, set_information):
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        function.restype = ctypes.c_int
    get_process_id.argtypes = [ctypes.c_void_p]
    get_process_id.restype = ctypes.c_uint32
    child_handle = ctypes.c_void_p(handle)
    actual_pid = get_process_id(child_handle)
    if not actual_pid:
        raise OSError(ctypes.get_last_error(), "Cannot verify HighQoS child process identity")
    if actual_pid != pid:
        raise ValueError("HighQoS Popen handle does not identify the requested child PID")
    information_class, version, execution_speed = 4, 1, 0x1

    def query(stage):
        state = _PowerThrottlingState(version, 0, 0)
        if not get_information(child_handle, information_class, ctypes.byref(state), ctypes.sizeof(state)):
            raise OSError(ctypes.get_last_error(), f"Cannot query child power throttling {stage} HighQoS")
        if state.Version != version:
            raise RuntimeError("Unsupported Windows child power-throttling structure version")
        return state

    before = query("before")
    requested_control = before.ControlMask | execution_speed
    requested_state = before.StateMask & ~execution_speed
    requested = _PowerThrottlingState(version, requested_control, requested_state)
    if not set_information(child_handle, information_class, ctypes.byref(requested), ctypes.sizeof(requested)):
        raise OSError(ctypes.get_last_error(), "Cannot apply child Windows HighQoS")
    after = query("after")
    if after.ControlMask != requested_control or after.StateMask != requested_state:
        raise RuntimeError("Windows child HighQoS verification failed; power-throttling masks differ")

    def record(state):
        return {"version": int(state.Version), "control_mask": int(state.ControlMask), "state_mask": int(state.StateMask)}

    return {"policy": "windows-high-qos", "applied": True, "verified": True, "process_id": pid,
            "scope": "newly-spawned-child-process-lifetime", "process_information_class": information_class,
            "before": record(before), "after": record(after), "priority_changed": False,
            "system_power_plan_changed": False}


@contextmanager
def keep_awake():
    """Thread-scoped guard; never alter the user's persistent power plan.

    Display sleep and explicit user-initiated sleep remain unaffected. Other
    platforms are a no-op; their sleep policy must be managed by the caller.
    Windows automatically drops execution requests when the process exits.
    """
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    execution_state = kernel.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint32]
    execution_state.restype = ctypes.c_uint32
    previous = execution_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if not previous:
        raise OSError(ctypes.get_last_error(), "Cannot prevent automatic sleep during the benchmark")
    try:
        yield
    finally:
        if not execution_state(previous):
            raise OSError(ctypes.get_last_error(), "Cannot restore benchmark execution state")
