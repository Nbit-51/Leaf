from types import SimpleNamespace
import ctypes
import json

import pytest

from leaf import power


def fake_windows(monkeypatch, results):
    calls = []
    def state(flags):
        calls.append(flags)
        return next(results)
    monkeypatch.setattr(power, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(power, "ctypes", SimpleNamespace(
        WinDLL=lambda *args, **kwargs: SimpleNamespace(SetThreadExecutionState=state),
        c_uint32=int, get_last_error=lambda: 5))
    return calls


def test_non_windows_does_not_call_system_api(monkeypatch):
    monkeypatch.setattr(power, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(power, "ctypes", None)
    with power.keep_awake():
        pass


@pytest.mark.parametrize("throws", [False, True])
def test_windows_restores_previous_thread_state_even_on_failure(monkeypatch, throws):
    calls = fake_windows(monkeypatch, iter([0x80000002, 0x80000001]))
    try:
        with power.keep_awake():
            assert calls == [0x80000001]
            if throws:
                raise ValueError("benchmark failed")
    except ValueError:
        assert throws
    assert calls == [0x80000001, 0x80000002]


def test_keep_awake_failure_does_not_run_benchmark(monkeypatch):
    calls = fake_windows(monkeypatch, iter([0]))
    with pytest.raises(OSError, match="Cannot prevent"):
        with power.keep_awake():
            pytest.fail("benchmark should not run")
    assert calls == [0x80000001]


class FakeHandle(int):
    closed = False


class FakeChild:
    def __init__(self):
        self._child_created = True
        self._handle = FakeHandle(0x1234)
        self.pid = 4321
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture
def high_qos_windows(monkeypatch):
    calls = []
    settings = {"snapshots": [(1, 4, 5), (1, 5, 4)], "get_results": [1, 1],
                "set_result": 1, "process_id": 4321}

    def get_information(handle, kind, pointer, size):
        index = sum(call[0] == "get" for call in calls)
        calls.append(("get", handle.value, kind, size))
        state = pointer._obj
        assert state.Version == 1
        state.Version, state.ControlMask, state.StateMask = settings["snapshots"][index]
        return settings["get_results"][index]

    def set_information(handle, kind, pointer, size):
        state = pointer._obj
        calls.append(("set", handle.value, kind, size, state.Version, state.ControlMask, state.StateMask))
        return settings["set_result"]

    def get_process_id(handle):
        calls.append(("pid", handle.value))
        return settings["process_id"]

    kernel = SimpleNamespace(GetProcessInformation=get_information, SetProcessInformation=set_information,
                             GetProcessId=get_process_id)
    dll_calls = []
    def dll(name, **kwargs):
        dll_calls.append((name, kwargs))
        return kernel
    monkeypatch.setattr(power, "os", SimpleNamespace(name="nt", getpid=lambda: 9999))
    monkeypatch.setattr(power, "subprocess", SimpleNamespace(Popen=FakeChild))
    monkeypatch.setattr(power.ctypes, "WinDLL", dll, raising=False)
    monkeypatch.setattr(power.ctypes, "get_last_error", lambda: 5, raising=False)
    return FakeChild(), calls, settings, kernel, dll_calls


def test_high_qos_preserves_other_flags_and_verifies_exact_child(high_qos_windows):
    child, calls, _, kernel, dll_calls = high_qos_windows
    result = power.set_child_high_qos(child)
    assert calls == [("pid", 0x1234), ("get", 0x1234, 4, 12),
                     ("set", 0x1234, 4, 12, 1, 5, 4), ("get", 0x1234, 4, 12)]
    assert dll_calls == [("kernel32", {"use_last_error": True})]
    assert kernel.GetProcessInformation.argtypes == [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    assert kernel.SetProcessInformation.restype is ctypes.c_int
    assert kernel.GetProcessId.restype is ctypes.c_uint32
    assert result["applied"] is True and result["verified"] is True
    assert result["before"] == {"version": 1, "control_mask": 4, "state_mask": 5}
    assert result["after"] == {"version": 1, "control_mask": 5, "state_mask": 4}
    assert result["priority_changed"] is False and result["system_power_plan_changed"] is False
    assert result["process_id"] == child.pid
    assert json.loads(json.dumps(result)) == result


def test_already_high_qos_still_requires_verified_api_success(high_qos_windows):
    child, calls, settings, _, _ = high_qos_windows
    settings["snapshots"] = [(1, 1, 0), (1, 1, 0)]
    assert power.set_child_high_qos(child)["applied"] is True
    assert calls[2] == ("set", 0x1234, 4, 12, 1, 1, 0)


def test_non_windows_high_qos_is_not_a_successful_noop(monkeypatch):
    monkeypatch.setattr(power, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(power, "ctypes", None)
    with pytest.raises(RuntimeError, match="Windows-only"):
        power.set_child_high_qos(None)


@pytest.mark.parametrize("handle", [None, True, "4660", 0, -1,
    (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1, 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)])
def test_invalid_handles_never_call_windows_api(high_qos_windows, handle):
    child, calls, _, _, dll_calls = high_qos_windows
    child._handle = handle
    with pytest.raises(ValueError, match="handle"):
        power.set_child_high_qos(child)
    assert calls == dll_calls == []


def test_closed_handle_is_rejected_before_windows_api(high_qos_windows):
    child, calls, _, _, dll_calls = high_qos_windows
    child._handle.closed = True
    with pytest.raises(ValueError, match="handle"):
        power.set_child_high_qos(child)
    assert calls == dll_calls == []


@pytest.mark.parametrize("pid", [None, True, "4321", 0, -1, 9999])
def test_invalid_or_parent_pid_is_rejected(high_qos_windows, pid):
    child, calls, _, _, dll_calls = high_qos_windows
    child.pid = pid
    with pytest.raises(ValueError, match="PID"):
        power.set_child_high_qos(child)
    assert calls == dll_calls == []


def test_missing_spawned_child_contract_cannot_qualify(high_qos_windows):
    child, calls, _, _, _ = high_qos_windows
    child._child_created = False
    with pytest.raises(ValueError, match="newly spawned"):
        power.set_child_high_qos(child)
    with pytest.raises(ValueError, match="newly spawned"):
        power.set_child_high_qos(SimpleNamespace(_child_created=True, _handle=FakeHandle(0x1234), pid=4321))
    assert calls == []


def test_exited_child_is_not_reconfigured(high_qos_windows):
    child, calls, _, _, _ = high_qos_windows
    child.returncode = 0
    with pytest.raises(ValueError, match="exited"):
        power.set_child_high_qos(child)
    assert calls == []


@pytest.mark.parametrize("process_id,error", [(0, OSError), (7777, ValueError)])
def test_handle_identity_mismatch_fails_before_query_or_write(high_qos_windows, process_id, error):
    child, calls, settings, _, _ = high_qos_windows
    settings["process_id"] = process_id
    with pytest.raises(error):
        power.set_child_high_qos(child)
    assert calls == [("pid", 0x1234)]


@pytest.mark.parametrize("missing", ["GetProcessInformation", "SetProcessInformation", "GetProcessId"])
def test_unsupported_api_does_not_silently_qualify(high_qos_windows, missing):
    child, calls, _, kernel, _ = high_qos_windows
    delattr(kernel, missing)
    with pytest.raises(RuntimeError, match="unavailable"):
        power.set_child_high_qos(child)
    assert calls == []


def test_unavailable_windows_library_is_not_a_successful_noop(high_qos_windows, monkeypatch):
    child, calls, _, _, _ = high_qos_windows
    def unavailable(*args, **kwargs):
        raise OSError("unsupported Windows library")
    monkeypatch.setattr(power.ctypes, "WinDLL", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        power.set_child_high_qos(child)
    assert calls == []


@pytest.mark.parametrize("stage", ["before", "set", "after"])
def test_api_failure_is_closed_and_reports_stage(high_qos_windows, stage):
    child, calls, settings, _, _ = high_qos_windows
    if stage == "set":
        settings["set_result"] = 0
    else:
        settings["get_results"][stage == "after"] = 0
    with pytest.raises(OSError) as raised:
        power.set_child_high_qos(child)
    assert raised.value.errno == 5
    if stage == "before":
        assert not any(call[0] == "set" for call in calls)
    assert stage in str(raised.value).lower() if stage != "set" else "apply" in str(raised.value).lower()


@pytest.mark.parametrize("after", [(1, 4, 4), (1, 5, 5), (1, 1, 0), (2, 5, 4)])
def test_unverified_masks_or_version_do_not_qualify(high_qos_windows, after):
    child, _, settings, _, _ = high_qos_windows
    settings["snapshots"][1] = after
    with pytest.raises(RuntimeError, match="verification|version"):
        power.set_child_high_qos(child)


def test_unsupported_snapshot_version_never_writes(high_qos_windows):
    child, calls, settings, _, _ = high_qos_windows
    settings["snapshots"][0] = (2, 4, 5)
    with pytest.raises(RuntimeError, match="version"):
        power.set_child_high_qos(child)
    assert not any(call[0] == "set" for call in calls)
