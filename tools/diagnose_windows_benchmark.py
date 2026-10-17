"""Windows-only, monitored native timing diagnostics; never a promotion gate.

The monitor runs on a different physical core. One-second telemetry includes
load and warmup, has nonzero overhead, and is not per-inference profiling.
No global power settings or unrelated application's state is changed. Optional
Above Normal priority applies only to the child created for this diagnostic.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake, set_child_high_qos
from tools.decoder_validation import benchmark_stability
from tools.validate_decoder import digest, native_policy, utc_now, write_record


def validate_diagnostic_request(metrics: dict, runs: int, expected_cpu: int | None = None) -> None:
    """Validate diagnostic samples without qualifying them as performance evidence."""
    if not isinstance(metrics, dict):
        raise ValueError("Diagnostic metrics must be an object")
    for name, expected in (("threads", 1), ("activation_bits", 32)):
        if type(metrics.get(name)) is not int or metrics[name] != expected:
            raise ValueError(f"Diagnostic {name} differs from requested workload")
    names = ["prefill_samples_ms", "decode_samples_ms"]
    if metrics.get("diagnostic_timing_build"):
        names += ["thread_cpu_prefill_ms", "thread_cpu_decode_ms",
                  "thread_cycles_prefill", "thread_cycles_decode"]
        if metrics.get("diagnostic_counter_version", 1) >= 2:
            names += ["process_page_faults_prefill", "process_page_faults_decode"]
        if expected_cpu is not None and metrics.get("diagnostic_counter_version", 1) < 3:
            raise ValueError("Actual CPU verification requires diagnostic counter version 3")
        if metrics.get("diagnostic_counter_version", 1) >= 3:
            names += [f"logical_cpu_{boundary}_{phase}" for boundary in ("start", "end")
                      for phase in ("prefill", "decode")]
        if metrics.get("diagnostic_counter_version", 1) >= 4:
            names += [f"thread_{counter}_{phase}" for counter in ("context_switches", "wait_reason_mask")
                      for phase in ("prefill", "decode")]
    elif expected_cpu is not None:
        raise ValueError("Actual CPU verification requires an instrumented build")
    for name in names:
        values = metrics.get(name)
        if (not isinstance(values, list) or len(values) != runs or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f"Diagnostic {name} must contain {runs} finite nonnegative samples")
        if name.startswith("logical_cpu_"):
            if any(type(v) is not int for v in values):
                raise ValueError(f"Diagnostic {name} must contain integer CPU identifiers")
            if expected_cpu is not None and any(v != expected_cpu for v in values):
                raise ValueError(f"Diagnostic {name} differs from requested CPU {expected_cpu}")


def spawn_pinned(command, *, parent, cpu: int, **kwargs):
    """Inherit the target affinity at creation, restoring the monitor on failure too."""
    previous = parent.cpu_affinity()
    try:
        parent.cpu_affinity([cpu])
        return subprocess.Popen(command, **kwargs)
    finally:
        parent.cpu_affinity(previous)


def parse_cpu_sets(data: bytes) -> list[dict]:
    result, offset = [], 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated CPU-set header")
        size, kind = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError("Invalid CPU-set size")
        if kind == 0:
            if size < 32:
                raise ValueError("Truncated CPU-set information")
            fields = struct.unpack_from("<IH6B", data, offset + 8)
            result.append(dict(zip(("id", "group", "logical_cpu", "core", "last_level_cache",
                                    "numa", "efficiency_class", "flags"), fields)))
        offset += size
    return result


def cpu_sets() -> list[dict]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = ctypes.c_int
    needed = ctypes.c_uint32()
    fn(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        raise OSError(ctypes.get_last_error(), "Cannot size CPU-set query")
    buf = ctypes.create_string_buffer(needed.value)
    if not fn(buf, len(buf), ctypes.byref(needed), None, 0):
        raise OSError(ctypes.get_last_error(), "Cannot query CPU sets")
    return parse_cpu_sets(buf.raw[:needed.value])


class CounterValue(ctypes.Structure):
    _fields_ = [("status", ctypes.c_uint32), ("value", ctypes.c_double)]


class CounterItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("formatted", CounterValue)]


class Counters:
    def __init__(self, cpu: int):
        self.dll = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self.dll.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.dll.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(CounterValue)]
        self.dll.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        self.dll.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        for name in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterValue", "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(self.dll, name).restype = ctypes.c_uint32
        if self.dll.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
            raise RuntimeError("Cannot open PDH query")
        self.handles, self.unavailable = {}, {}
        paths = {"cpu_performance_percent": fr"\Processor Information(0,{cpu})\% Processor Performance",
                 "cpu_frequency_mhz_reported": fr"\Processor Information(0,{cpu})\Processor Frequency",
                 "cpu_dpc_percent": fr"\Processor({cpu})\% DPC Time",
                 "cpu_interrupt_percent": fr"\Processor({cpu})\% Interrupt Time",
                 "processor_queue_length": r"\System\Processor Queue Length",
                 "page_reads_per_second": r"\Memory\Page Reads/sec",
                 "process_cpu": r"\Process(*)\% Processor Time",
                 "process_pid": r"\Process(*)\ID Process"}
        for name, path in paths.items():
            handle = ctypes.c_void_p()
            status = self.dll.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle))
            if status:
                self.unavailable[name] = int(status)
            else:
                self.handles[name] = handle
        self.dll.PdhCollectQueryData(self.query)

    def read(self):
        status = self.dll.PdhCollectQueryData(self.query)
        result = {"collect_status": int(status)}
        for name, handle in self.handles.items():
            if name.startswith("process_"):
                continue
            value = CounterValue()
            status = self.dll.PdhGetFormattedCounterValue(handle, 0x200 | 0x8000, None, ctypes.byref(value))
            result[name] = value.value if not status and value.status in (0, 1) and math.isfinite(value.value) else None
        usage = self.array("process_cpu")
        pids = self.array("process_pid")
        result["process_activity"] = sorted([
            {"name": name, "pid": int(pids[name]), "one_core_percent": value}
            for name, value in usage.items() if name in pids and pids[name] > 0 and name != "_Total" and value > 0
        ], key=lambda x: x["one_core_percent"], reverse=True)
        return result

    def array(self, name):
        if name not in self.handles:
            return {}
        length, count = ctypes.c_uint32(), ctypes.c_uint32()
        handle = self.handles[name]
        fmt = 0x200 | 0x8000  # double, no cap at 100% for multicore processes
        self.dll.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(length), ctypes.byref(count), None)
        if not length.value:
            return {}
        buffer = ctypes.create_string_buffer(length.value)
        status = self.dll.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(length), ctypes.byref(count), buffer)
        if status:
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(CounterItem))
        return {items[i].name: items[i].formatted.value for i in range(count.value)
                if items[i].formatted.status in (0, 1) and math.isfinite(items[i].formatted.value)}

    def close(self):
        self.dll.PdhCloseQuery(self.query)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("executable", "artifact", "tokens", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--monitor-cpu", type=int, default=27)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--high-qos-abba", action="store_true")
    parser.add_argument("--above-normal", action="store_true",
                        help="match the scoped child priority used by current acceptance runs")
    args = parser.parse_args()
    if os.name != "nt" or args.runs < 5 or args.warmup < 0:
        parser.error("Requires Windows, at least five samples and nonnegative warmup")
    if args.output.exists():
        parser.error("Use a new result path; retain earlier diagnostics")
    runtime = json.loads(subprocess.check_output([str(args.executable.resolve()), "--runtime-info"], text=True))
    if (runtime.get("diagnostic_start_gate_supported") is not True or
            runtime.get("diagnostic_counter_version", 0) < 3):
        parser.error("Rebuild with LEAF_DIAGNOSTIC_TIMING for start-gate and actual-CPU verification")
    topology = cpu_sets()
    cpus = {c["logical_cpu"]: c for c in topology if c["group"] == 0}
    if args.cpu not in cpus or args.monitor_cpu not in cpus or cpus[args.cpu]["core"] == cpus[args.monitor_cpu]["core"]:
        parser.error("Benchmark and monitor require distinct physical cores in group zero")
    ids = json.loads(args.tokens.read_text())["benchmark_ids"]
    if len(ids) < 2 or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
        parser.error("Invalid benchmark tokens")
    inputs = {str(p): digest(p) for p in (args.executable, args.artifact, args.tokens, Path(__file__))}
    me = psutil.Process()
    original_affinity = me.cpu_affinity()
    # The child inherits the benchmark CPU at creation; Python monitors elsewhere.
    me.cpu_affinity([args.monitor_cpu])
    counters = Counters(args.cpu)
    record = {"benchmark": "windows-native-environment-diagnostic-v1", "measured_at_utc": utc_now(),
              "platform": platform.platform(), "input_sha256": inputs, "cpu_sets": topology,
              "cpu": args.cpu, "monitor_cpu": args.monitor_cpu, "native_policy": native_policy(),
              "active_power_scheme": subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip(),
              "processor_power_settings": subprocess.check_output(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR"], text=True),
              "runs": args.runs, "warmup": args.warmup, "prefill_tokens": len(ids) - 1,
              "windows_process_priority": "above_normal" if args.above_normal else "normal",
              "child_start_policy": "inherit-target-affinity-then-wait-for-monitor-gate",
              "diagnostic_runtime": runtime,
              "process_sampler": "PDH wildcard counters", "monitor_pid": os.getpid(),
              "counter_unavailable": counters.unavailable, "passes": [], "automatic_selection_authorized": False,
              "scope": "Instrumented diagnostic, not acceptance timing. 1 Hz telemetry includes load/warmup. No thermal/power-limit sensors; process totals cannot locate competitors on a core. Inaccessible/short-lived processes may be missed."}
    order = [False, True, True, False] if args.high_qos_abba else [False] * 4
    try:
        with keep_awake(), tempfile.TemporaryDirectory(prefix="leaf_diag_") as temp:
            root = Path(temp)
            request = root / "tokens.bin"
            request.write_bytes(struct.pack("<II", 1, len(ids)) + struct.pack(f"<{len(ids)}I", *ids))
            for index, high_qos in enumerate(order):
                print(f"Diagnostic {index+1}/4: HighQoS={high_qos}", flush=True)
                command = [str(args.executable.resolve()), str(args.artifact.resolve()), str(request),
                           str(root / "logits.bin"), str(root / "metrics.json"), "bench", "1", str(args.runs), str(args.warmup), "1", "0", "1", "32"]
                env = dict(os.environ)
                env.pop("LEAF_DECODER_PROFILE", None)
                gate = root / f"start-{index}.ready"
                env["LEAF_DIAGNOSTIC_START_GATE"] = str(gate)
                samples = []
                start = time.perf_counter()
                flags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if args.above_normal else 0
                with spawn_pinned(command, parent=me, cpu=args.cpu, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env, creationflags=flags) as child:
                    try:
                        process = psutil.Process(child.pid)
                        if args.above_normal and process.nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                            raise RuntimeError("Native child priority did not match request")
                        if process.cpu_affinity() != [args.cpu]:
                            raise RuntimeError("Native child affinity did not match request")
                        qos = set_child_high_qos(child) if high_qos else None
                        process.cpu_percent()
                        psutil.cpu_percent(percpu=True)
                        counters.read()  # Prime this pass; ignore the initial interval.
                        measured_at = utc_now()
                        start = time.perf_counter()
                        gate.touch()
                        last = time.perf_counter()
                        while child.poll() is None:
                            try:
                                child.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                pass
                            now = time.perf_counter()
                            sample_start = time.perf_counter()
                            battery = psutil.sensors_battery()
                            values = {"elapsed_seconds": now - start, "interval_seconds": now - last,
                                      "cpu_busy_percent": psutil.cpu_percent(percpu=True),
                                      "available_memory_bytes": psutil.virtual_memory().available,
                                      "power_plugged": battery.power_plugged if battery else None,
                                      "counters": counters.read()}
                            try:
                                values["child_one_core_percent"] = process.cpu_percent()
                                values["child_affinity"] = process.cpu_affinity()
                            except psutil.NoSuchProcess:
                                values["child_exited"] = True
                            values["collection_seconds"] = time.perf_counter() - sample_start
                            samples.append(values)
                            last = now
                        stdout, stderr = child.communicate()
                        if child.returncode:
                            raise RuntimeError(f"Native child failed: {stderr}")
                    except BaseException:
                        if child.poll() is None:
                            child.kill()
                        child.communicate()
                        raise
                metrics = json.loads((root / "metrics.json").read_text())
                item = {"index": index, "high_qos": high_qos, "child_pid": child.pid, "qos": qos,
                        "measured_at_utc": measured_at, "finished_at_utc": utc_now(),
                        "elapsed_seconds": time.perf_counter() - start, "latency": metrics,
                        "stability": benchmark_stability(metrics), "telemetry": samples,
                        "request_validated": False}
                record["passes"].append(item)
                validate_diagnostic_request(metrics, args.runs, expected_cpu=args.cpu)
                item["request_validated"] = True
                item["actual_cpu_boundaries_match_request"] = True
                write_record(args.output, dict(record, complete=False))
                print(json.dumps({"prefill": metrics["prefill_p50_ms"], "decode": metrics["decode_p50_ms"],
                                  "stable": item["stability"]["passed"]}), flush=True)
    finally:
        counters.close()
        me.cpu_affinity(original_affinity)
        # Preserve partial diagnostics on failure, without converting failure to success.
        record["complete"] = len(record["passes"]) == 4 and all(p["request_validated"] for p in record["passes"])
        record["inputs_unchanged"] = all(digest(Path(p)) == h for p, h in inputs.items())
        write_record(args.output, record)
    if not record["inputs_unchanged"]:
        raise RuntimeError("Diagnostic input changed")


if __name__ == "__main__":
    main()
