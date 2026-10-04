import struct

import pytest

from tools.diagnose_windows_benchmark import parse_cpu_sets
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
