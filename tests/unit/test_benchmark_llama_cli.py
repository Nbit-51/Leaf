import pytest

from tools.benchmark_llama_cli import TIMERS, continuation, llama_work, summarize


def test_continuation_removes_only_cli_framing():
    assert continuation(' token \r\n\r\n', 'leaf') == ' token \n'
    assert continuation(' token \r\n\r\n\r\n', 'llama') == ' token \n'
    with pytest.raises(ValueError, match='framing'):
        continuation('token\n', 'llama')


def test_llama_work_keeps_prompt_and_decode_counts_distinct():
    report = ('0.012 I common_perf_print: prompt eval time = 110.0 ms / 63 tokens\n'
              '0.013 I common_perf_print:        eval time = 77.0 ms / 7 runs\n')
    assert llama_work(report) == (63, 7)
    with pytest.raises(ValueError, match='Missing or ambiguous'):
        llama_work(report + report)
    with pytest.raises(ValueError, match='Missing or ambiguous'):
        llama_work(report.splitlines()[0])


def test_comparison_requires_balanced_pairs_and_preserves_unstable_samples():
    samples = []
    for pair in range(1, 7):
        for engine in (('leaf', 'llama') if pair % 2 else ('llama', 'leaf')):
            time = 90.0 if engine == 'leaf' else 100.0
            samples.append(dict(pair=pair, engine=engine, **{timer: time for timer in TIMERS}))
    assert summarize(samples, 6)[TIMERS[0]]['median_paired_leaf_over_llama'] == 0.9
    samples[0][TIMERS[0]] = 300.0
    result = summarize(samples, 6)[TIMERS[0]]
    assert not result['stable']
    assert result['stability']['leaf']['sample_count'] == 6
    with pytest.raises(ValueError, match='Incomplete or out-of-order'):
        summarize(samples[:-1], 6)
