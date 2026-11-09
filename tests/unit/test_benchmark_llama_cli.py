import pytest

from tools.benchmark_llama_cli import TIMERS, continuation, llama_work, summarize, verify_quality_report


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


@pytest.mark.parametrize('invalid', [None, 'passed', 'fp32_math', 'threads', 'storage', 'gates', 'artifact_sha256',
                                    'executable_sha256', 'tokens_sha256', 'source_sha256'])
def test_accuracy_gate_requires_exact_measured_inputs(invalid):
    report = dict(format='leaf-output-quality-v1', complete=True, passed=True, precision='FP32',
                  fp32_math='accurate', threads=1, storage=dict(effective='fp32'), artifact_sha256='weights',
                  executable_sha256='runtime', tokens_sha256='fixture', source_sha256={'config.json': 'config'},
                  gates=dict(min_next_token_agreement=0.999, max_perplexity_ratio=1.001,
                             logit_atol=2e-4, logit_rtol=2e-4, exact_generation_required=True))
    args = dict(artifact_sha='weights', binary_sha='runtime', tokens_sha='fixture',
                snapshot_hashes={'config.json': 'config'})
    if invalid is None:
        verify_quality_report(report, **args)
    else:
        report[invalid] = {}
        with pytest.raises(ValueError, match='passing strict FP32 quality report'):
            verify_quality_report(report, **args)


def test_converter_stages_parameters_without_legacy_attention_buffers(tmp_path):
    torch = pytest.importorskip('torch')
    from safetensors.torch import load_file, save_file
    from tools.prepare_ollama_comparison import stage_conversion_snapshot

    snapshot, staging = tmp_path / 'snapshot', tmp_path / 'staging'
    snapshot.mkdir()
    staging.mkdir()
    (snapshot / 'config.json').write_text('{"model_type":"gpt2"}', encoding='utf-8')
    parameters = {'h.0.attn.c_attn.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
                  'h.0.attn.c_attn.bias': torch.ones(3)}
    original = dict(parameters, **{'h.0.attn.bias': torch.ones((1, 1, 2, 2), dtype=torch.bool)})
    save_file(original, snapshot / 'model.safetensors')
    assert stage_conversion_snapshot(snapshot, staging) == staging
    staged = load_file(staging / 'model.safetensors')
    assert staged.keys() == parameters.keys()
    assert all(torch.equal(staged[name], value) for name, value in parameters.items())
    assert load_file(snapshot / 'model.safetensors').keys() == original.keys()
