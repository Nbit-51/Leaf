import pytest
from tools.diagnose_embedding_inputs import summarize


def sample():
    return dict(diagnostic_timing_build=True, samples_ms=[[10, 10, 100]],
                thread_cpu_ms=[[10, 10, 10]], thread_cycles=[[100, 100, 100]],
                process_page_faults=[[0, 0, 0]])


def test_stall_is_retained_with_its_cpu_and_fault_counters():
    result = summarize(sample(), 3)[0]
    assert result["slowest"]["samples_ms"] == 100
    assert result["slowest"]["thread_cpu_ms"] == 10
    assert result["faults_total"] == 0
    assert result["wall_p90_over_p10"] > 1.25


@pytest.mark.parametrize("name", ["samples_ms", "thread_cpu_ms", "thread_cycles", "process_page_faults"])
def test_missing_samples_cannot_be_summarized(name):
    metrics = sample()
    metrics[name] = [[1, 2]]
    with pytest.raises(ValueError):
        summarize(metrics, 3)


def test_uninstrumented_output_is_rejected():
    metrics = sample()
    metrics.pop("diagnostic_timing_build")
    with pytest.raises(ValueError, match="instrumented"):
        summarize(metrics, 3)
