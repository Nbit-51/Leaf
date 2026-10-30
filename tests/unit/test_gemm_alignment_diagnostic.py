"""Alignment diagnostics retain all observations and reject incomplete native proof."""
import copy

import pytest

from tools.benchmark_gemm_alignment import summarize_records


ROUNDS, WARMUP = 3, 2


def fixture():
    order = [16*(3-p if r % 2 else p) for r in range(ROUNDS)
             for p in (0, 1, 2, 3, 3, 2, 1, 0)]
    cases = []
    for m in (63, 31):
        for n, k in ((2560, 960), (960, 2560), (4864, 896), (896, 4864)):
            for copies in (1, 4):
                cases.append(dict(type='case', complete=True, tokens=m, rows=n, columns=k,
                    weight_copies=copies, rounds=ROUNDS, warmup_sweeps=WARMUP,
                    packing_included=False, fp32_math='fast', whole_model_speed_claim=False,
                    raw_entry='single_noinline_runtime_shape', bit_exact_values=4*m*n*copies,
                    scalar_oracle_positions=36*copies, oracle_kind="hardware_scalar_fma_direct_input",
                    issued_fma_instructions_per_matrix=((m+15)//16)*2*n*k,
                    sample_order=order.copy(), original_packed_mod64=16,
                    packed_bytes=((m+15)//16)*k*16*4, weight_bytes=copies*n*k*4,
                    fma_ns_before=[0.25]*3, fma_ns_after=[0.30]*3,
                    phases=[dict(packed_mod64=p, ms_per_matrix=[9.0 if p in (0, 32) else 10.0]*(2*ROUNDS))
                            for p in (0, 16, 32, 48)]))
    return cases+[dict(type='summary', complete=True, cases=16,
                       timing_qualified=False, acceptance_timing=False)]


def test_complete_proof_summarizes_known_ratio_without_changing_raw_samples():
    records = fixture()
    saved = copy.deepcopy(records)
    result = summarize_records(records, ROUNDS, WARMUP)
    assert records == saved
    assert len(result) == 16
    first = result[0]
    assert first['aligned_over_split_pooled_ratio'] == pytest.approx(0.9)
    assert first['aligned_over_split_paired_ratios'] == pytest.approx([0.9, 0.9])
    assert first['all_phases_stable'] is True
    assert first['fma_control_before_after_ns'] == [0.25, 0.30]
    assert first['fma_control_larger_over_smaller'] == pytest.approx(1.2)
    expected = records[0]['issued_fma_instructions_per_matrix']*0.275*1e-6/9.0
    assert first['fraction_of_adjacent_fma_control'][0] == pytest.approx(expected)
    assert first['phases'][16]['stability']['sample_count'] == 2*ROUNDS
    assert first['original_packed_mod64'] == 16


def test_valid_but_unstable_sample_is_retained_and_fails_spread():
    records = fixture()
    records[0]['phases'][0]['ms_per_matrix'][-2:] = [90.0, 90.0]
    saved = copy.deepcopy(records)
    result = summarize_records(records, ROUNDS, WARMUP)
    assert records == saved
    assert result[0]['all_phases_stable'] is False
    assert result[0]['phases'][0]['stability']['sample_count'] == 2*ROUNDS
    assert result[0]['phases'][0]['stability']['p90_ms'] == 90.0


@pytest.mark.parametrize('fault', ['missing_case', 'duplicate_case', 'missing_summary',
    'incomplete_summary', 'summary_claim', 'early_summary', 'unexpected_shape'])
def test_missing_duplicated_or_invalid_cases_never_become_complete(fault):
    records = fixture()
    if fault == 'missing_case':
        del records[0]
    elif fault == 'duplicate_case':
        records[1] = copy.deepcopy(records[0])
    elif fault == 'missing_summary':
        records.pop()
    elif fault == 'incomplete_summary':
        records[-1]['complete'] = False
    elif fault == 'summary_claim':
        records[-1]['timing_qualified'] = True
    elif fault == 'early_summary':
        records[0], records[-1] = records[-1], records[0]
    else:
        records[0]['columns'] = 768
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)


@pytest.mark.parametrize(('field', 'value'), [
    ('type', 'summary'), ('complete', False), ('rounds', ROUNDS+1),
    ('warmup_sweeps', WARMUP+1), ('packing_included', True), ('fp32_math', 'accurate'),
    ('whole_model_speed_claim', True), ('raw_entry', 'different_kernel'),
    ('bit_exact_values', 1), ('scalar_oracle_positions', 35),
    ('issued_fma_instructions_per_matrix', 1),
])
def test_native_request_and_exactness_evidence_are_required(field, value):
    records = fixture()
    records[0][field] = value
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'reordered', 'short_samples',
                                  'extra_samples', 'missing_samples', 'order'])
def test_all_four_phases_have_complete_declared_ordered_samples(fault):
    records = fixture()
    phases = records[0]['phases']
    if fault == 'missing':
        phases.pop()
    elif fault == 'duplicate':
        phases[1]['packed_mod64'] = 0
    elif fault == 'reordered':
        phases.reverse()
    elif fault == 'short_samples':
        phases[0]['ms_per_matrix'].pop()
    elif fault == 'extra_samples':
        phases[0]['ms_per_matrix'].append(9.0)
    elif fault == 'missing_samples':
        del phases[0]['ms_per_matrix']
    else:
        records[0]['sample_order'][0] = 16
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0, -1, True, False, '9', None])
def test_invalid_sample_is_rejected_without_filtering(value):
    records = fixture()
    records[0]['phases'][0]['ms_per_matrix'][0] = value
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)


@pytest.mark.parametrize('value', [None, [], [0.25]*2, [0.25]*4, [float('nan')]*3,
    [float('inf')]*3, [0]*3, [-0.1]*3, [True]*3, ['0.25']*3])
@pytest.mark.parametrize('side', ['before', 'after'])
def test_invalid_adjacent_fma_control_is_rejected(side, value):
    records = fixture()
    records[0]['fma_ns_'+side] = value
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)


@pytest.mark.parametrize('location', ['weight_copies', 'phase', 'sample_order', 'summary_complete',
                                      'summary_claim'])
def test_boolean_metadata_cannot_impersonate_integer_phase_or_flag(location):
    records = fixture()
    if location == 'weight_copies':
        records[0]['weight_copies'] = True
    elif location == 'phase':
        records[0]['phases'][0]['packed_mod64'] = False
    elif location == 'sample_order':
        records[0]['sample_order'][0] = False
    elif location == 'summary_complete':
        records[-1]['complete'] = 1
    else:
        records[-1]['timing_qualified'] = 0
    with pytest.raises(ValueError):
        summarize_records(records, ROUNDS, WARMUP)
