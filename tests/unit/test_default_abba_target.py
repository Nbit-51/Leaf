"""The decode target must preserve independent improvement/stability gates."""
from contextlib import nullcontext
import json
import sys

import pytest

from tools import benchmark_default_abba as runner


@pytest.mark.parametrize('target,prefill,decode,passed', [
    (None, 95.0, 10.1, True),       # Original default retains its original limits.
    (None, 101.0, 9.0, False),      # A decode gain cannot pass the prefill target.
    ('decode', 101.0, 9.0, True),
    ('decode', 103.0, 9.0, False),  # Reject prefill regression despite faster decode.
    ('decode', 99.0, 9.9, False),   # Require a real decode gain, not mere non-regression.
])
def test_phase_gate_is_selected_before_samples(tmp_path, monkeypatch, target, prefill, decode, passed):
    before, after = tmp_path / 'before.exe', tmp_path / 'after.exe'
    before.write_bytes(b'before')
    after.write_bytes(b'after')
    (tmp_path / 'decoder-32.leaf').write_bytes(b'weights')
    (tmp_path / 'tokens.json').write_text(json.dumps({'benchmark_ids': [1, 2, 3]}))
    output = tmp_path / 'result.json'
    argv = ['benchmark_default_abba', '--before', str(before), '--after', str(after),
            '--workdir', str(tmp_path), '--output', str(output)]
    if target:
        argv.extend(['--target-phase', target])
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(runner, 'os', type('Environment', (), {'name': 'nt', 'environ': {}}))
    monkeypatch.setattr(runner, 'pin_cpu', lambda cpu: None)
    monkeypatch.setattr(runner, 'require_benchmark_ac', lambda: {'ac_line_status': 1})
    monkeypatch.setattr(runner, 'keep_awake', nullcontext)
    monkeypatch.setattr(runner, 'validate_timing_request', lambda *args, **kwargs: None)

    def native(binary, *args, **kwargs):
        # Every distribution is stable; only the declared ratio criteria decide.
        p, d = (100.0, 10.0) if binary == before else (prefill, decode)
        return None, dict(prefill_p50_ms=p, decode_p50_ms=d,
                          prefill_samples_ms=[p] * 31, decode_samples_ms=[d] * 31,
                          optimized_fp32_active=True)
    monkeypatch.setattr(runner, 'run_native', native)
    runner.main()
    result = json.loads(output.read_text())
    assert result['complete'] and result['gate']['every_pass_stable']
    assert result['gate']['passed'] is passed
    assert result['target_phase'] == (target or 'prefill')
    assert result['ratio_limits'][result['target_phase']] == 0.98
