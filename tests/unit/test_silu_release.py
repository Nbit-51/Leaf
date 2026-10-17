import pytest

from tools.verify_silu_release import check_release_dispatch


@pytest.mark.parametrize("bits,scalar,tokenwise", [(32, False, False), (8, False, False),
    (8, True, False), (8, False, True), (32,True,False), (32,False,True), (4,False,False)])
def test_release_dispatch_matches_fp32_decode_and_w8a8_prefill(bits, scalar, tokenwise):
    enabled = bits in (8,32) and not scalar
    metrics = dict(avx2=not scalar, vector_silu_gate_enabled=enabled,
                   vector_silu_gate_calls=22 if enabled and (bits == 32 or not tokenwise) else 0)
    check_release_dispatch(metrics, bits, scalar=scalar, tokenwise=tokenwise)
    metrics['vector_silu_gate_calls'] = 22 if metrics['vector_silu_gate_calls'] == 0 else 0
    with pytest.raises(ValueError, match='dispatch'):
        check_release_dispatch(metrics, bits, scalar=scalar, tokenwise=tokenwise)


@pytest.mark.parametrize('marker', ['experimental_silu_gate_build', 'diagnostic_timing_build'])
def test_release_does_not_accept_instrumented_or_experimental_build(marker):
    with pytest.raises(ValueError, match='Release executable'):
        check_release_dispatch({marker: True}, 8)
