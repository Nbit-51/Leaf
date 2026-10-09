import struct

import pytest

from tools.diagnose_windows_benchmark import parse_cpu_sets, validate_diagnostic_request
from tools.benchmark_decoder_comparison import validate_timing_request
from tools.summarize_windows_diagnostics import distribution, summarize


def cpu_record(logical=2, core=2, efficiency=1, size=32):
    return struct.pack("<IIIH6BIQ", size, 0, 258, 0, logical, core, 0, 0, efficiency, 1, 0, 0) + b'\0' * (size - 32)


def test_cpu_sets_honor_variable_size_and_shared_core_topology():
    result = parse_cpu_sets(cpu_record(size=40) + struct.pack('<II', 8, 9) + cpu_record(3))
    assert [r['logical_cpu'] for r in result] == [2, 3]
    assert result[0]['core'] == result[1]['core'] == 2
    assert result[0]['efficiency_class'] == 1


@pytest.mark.parametrize('data', [b'\0', struct.pack('<II', 0, 0), struct.pack('<II', 40, 0), struct.pack('<II', 8, 0)])
def test_cpu_sets_reject_malformed_sizes(data):
    with pytest.raises(ValueError):
        parse_cpu_sets(data)


def test_missing_telemetry_is_not_reported_as_zero():
    assert distribution([None, None]) == {"count": 0}
    assert distribution([None, 2, 4]) == {"count": 2, "minimum": 2, "median": 3, "maximum": 4}


def instrumented_metrics():
    return dict(threads=1, activation_bits=32, diagnostic_timing_build=True,
                prefill_samples_ms=[100.0] * 5, decode_samples_ms=[20.0] * 5,
                thread_cpu_prefill_ms=[93.75] * 5, thread_cpu_decode_ms=[15.625] * 5,
                thread_cycles_prefill=[1000] * 5, thread_cycles_decode=[200] * 5)


def test_instrumented_samples_are_diagnostics_only_and_keep_their_marker():
    metrics = instrumented_metrics()
    validate_diagnostic_request(metrics, 5)
    assert metrics['diagnostic_timing_build'] is True
    with pytest.raises(ValueError, match='cannot qualify performance'):
        validate_timing_request(metrics, runs=5, threads=1, activation_bits=32)


def test_page_fault_counter_version_requires_matching_samples():
    metrics = instrumented_metrics()
    metrics.update(diagnostic_counter_version=2, process_page_faults_prefill=[0] * 5,
                   process_page_faults_decode=[0] * 5)
    validate_diagnostic_request(metrics, 5)
    metrics['process_page_faults_decode'] = [0] * 4
    with pytest.raises(ValueError, match='process_page_faults_decode'):
        validate_diagnostic_request(metrics, 5)


@pytest.mark.parametrize('key,value', [('threads', 2), ('threads', True), ('activation_bits', 8),
    ('prefill_samples_ms', [100] * 4), ('thread_cpu_decode_ms', None),
    ('thread_cycles_prefill', [float('nan')] * 5), ('thread_cpu_prefill_ms', [-1] * 5)])
def test_diagnostic_requests_still_reject_mismatched_or_invalid_samples(key, value):
    metrics = instrumented_metrics()
    metrics[key] = value
    with pytest.raises(ValueError):
        validate_diagnostic_request(metrics, 5)


def test_published_summary_omits_process_names_and_pids():
    import json
    topology = parse_cpu_sets(cpu_record() + cpu_record(3))
    sample = {"interval_seconds": 2, "cpu_busy_percent": [0, 0, 100, 0], "available_memory_bytes": 100,
              "power_plugged": True, "counters": {"process_activity": [
                  {"name": "private-app-name", "pid": 23456, "one_core_percent": 50},
                  {"name": "Code#3", "pid": 34567, "one_core_percent": 25}]}}
    record = {"cpu": 2, "monitor_cpu": 27, "cpu_sets": topology, "complete": False, "inputs_unchanged": True,
              "measured_at_utc": "test", "runs": 31, "warmup": 10, "native_policy": {}, "active_power_scheme": "test",
              "processor_power_settings": "test", "counter_unavailable": {}, "scope": "diagnostic",
              "passes": [{"index": 0, "high_qos": False, "latency": {}, "stability": {}, "telemetry": [sample]}]}
    result = summarize(record)
    text = json.dumps(result)
    assert all(secret not in text for secret in ("private-app-name", "Code#3", "23456", "34567"))
    assert result["passes"][0]["estimated_process_cpu_seconds_by_category"] == {"other_processes": 1, "vs_code": 0.5}
