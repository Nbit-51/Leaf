"""Scoped, explicitly requested benchmark-only Windows power controls."""
from contextlib import contextmanager
import ctypes
import os
import subprocess
import threading
import time


WINDOWS_POWER_MODE_NAMES = (
    "battery-saver", "better-battery", "balanced", "high-performance",
    "max-performance", "game-mode", "mixed-reality",
)
_POWER_MODE_INITIAL_TIMEOUT_SECONDS = 3.0
# A failed unregistration must not free an address Windows may still call.
_POWER_MODE_ORPHANED_CALLBACKS = []


@contextmanager
def benchmark_power_mode_guard(record: dict, *, required_mode: str | None = None):
    """Audit Windows' effective mode without changing the system power policy.

    AC power and GetSystemPowerStatus's saver flag do not describe the effective
    performance mode. Save into the caller's initially empty record even when
    setup fails. Saving modes require an explicit matching required_mode;
    unknown modes and *any* mode change invalidate the whole window, including
    a change back to the starting mode. This is an additional
    environmental guard, not proof of a fixed CPU frequency or PPM profile.
    Correctness checks and ordinary inference should not use this timing guard.
    """
    if record:
        raise ValueError("Power-mode audit requires a fresh empty record")
    if required_mode is not None and required_mode not in WINDOWS_POWER_MODE_NAMES:
        raise ValueError("Required benchmark mode must be a supported effective power mode")
    record.update(api_version=2, passed=False, notifications=[], system_power_plan_changed=False,
                  required_mode=required_mode, saving_mode_explicitly_selected=required_mode in WINDOWS_POWER_MODE_NAMES[:2])
    if os.name != "nt":
        raise RuntimeError("Effective benchmark power-mode auditing requires Windows")
    try:
        library = ctypes.WinDLL("powrprof", use_last_error=True)
        register = library.PowerRegisterForEffectivePowerModeNotifications
        unregister = library.PowerUnregisterFromEffectivePowerModeNotifications
    except (AttributeError, OSError) as error:
        raise RuntimeError("Windows effective power-mode notifications are unavailable") from error
    callback_type = ctypes.WINFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)
    register.argtypes = [ctypes.c_uint32, callback_type, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    register.restype = ctypes.c_int32
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int32
    ready, lock = threading.Event(), threading.Lock()
    origin = time.monotonic()

    @callback_type
    def changed(mode, context):
        number = int(mode)
        name = WINDOWS_POWER_MODE_NAMES[number] if 0 <= number < len(WINDOWS_POWER_MODE_NAMES) else "unknown"
        with lock:
            record["notifications"].append(dict(mode=number, name=name, elapsed_seconds=time.monotonic()-origin))
        ready.set()

    handle = ctypes.c_void_p()
    result = register(2, changed, None, ctypes.byref(handle))
    if result:
        record["register_hresult"] = int(result)
        raise OSError(result, "Cannot register effective benchmark power-mode notifications")
    if not handle.value:
        _POWER_MODE_ORPHANED_CALLBACKS.append((library, changed, handle))
        raise RuntimeError("Windows returned no effective power-mode registration handle")
    body_failed = True

    def validate():
        with lock:
            events = list(record["notifications"])
        record["initial_mode"] = events[0]["name"] if events else None
        record["final_mode"] = events[-1]["name"] if events else None
        if not events or any(x["mode"] not in range(len(WINDOWS_POWER_MODE_NAMES)) for x in events):
            return "Timing requires a known effective power mode; unknown mode observed"
        if required_mode is None and any(x['mode'] < 2 for x in events):
            return f"Timing requires a known performance power mode; saving/unknown mode observed: {record['final_mode']}"
        if len({x["mode"] for x in events}) != 1:
            return "Effective power mode changed during timing; no comparison accepted"
        if required_mode is not None and events[0]["name"] != required_mode:
            return f"Timing requires effective power mode {required_mode}; observed {events[0]['name']}"
        return None

    try:
        if not ready.wait(_POWER_MODE_INITIAL_TIMEOUT_SECONDS):
            raise RuntimeError("Windows supplied no initial effective power-mode notification")
        error = validate()
        if error:
            record["error"] = error
            raise RuntimeError(error)
        yield record
        body_failed = False
    finally:
        # Windows waits for outstanding callbacks here. Keep changed alive until
        # unregistration completes; do not unregister from the callback itself.
        result = unregister(handle)
        if result:
            _POWER_MODE_ORPHANED_CALLBACKS.append((library, changed, handle))
            record["unregister_hresult"] = int(result)
            raise OSError(result, "Cannot unregister effective benchmark power-mode notifications")
        error = validate()
        record["passed"] = not body_failed and error is None
        if error:
            record["error"] = error
            if not body_failed:
                raise RuntimeError(error)


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def require_benchmark_ac() -> dict:
    """Reject offline/unknown Windows AC state; no settings are changed."""
    if os.name != "nt":
        raise RuntimeError("Windows benchmark power check requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetSystemPowerStatus
    query.argtypes = [ctypes.POINTER(_SystemPowerStatus)]
    query.restype = ctypes.c_int
    state = _SystemPowerStatus()
    if not query(ctypes.byref(state)):
        raise OSError(ctypes.get_last_error(), "Cannot verify benchmark AC power")
    if state.ACLineStatus != 1:
        raise RuntimeError("Benchmark requires verified AC power; offline or unknown power invalidates timing")
    return {"ac_line_status": int(state.ACLineStatus), "battery_saver": bool(state.SystemStatusFlag)}


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
def keep_awake(*, display: bool = False):
    """Thread-scoped guard; never alter the user's persistent power plan.

    Display sleep remains unaffected unless the benchmark explicitly requests
    display=True (useful on Modern Standby hosts). Explicit user-initiated sleep
    remains unaffected. Other platforms are a no-op; their sleep policy must be
    managed by the caller.
    Windows automatically drops execution requests when the process exits.
    """
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    execution_state = kernel.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint32]
    execution_state.restype = ctypes.c_uint32
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, optionally ES_DISPLAY_REQUIRED.
    previous = execution_state(0x80000001 | (0x2 if display else 0))
    if not previous:
        raise OSError(ctypes.get_last_error(), "Cannot prevent automatic sleep during the benchmark")
    try:
        yield
    finally:
        if not execution_state(previous):
            raise OSError(ctypes.get_last_error(), "Cannot restore benchmark execution state")
