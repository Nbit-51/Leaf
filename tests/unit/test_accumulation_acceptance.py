"""Research timing must stay bound to complete strict trained-quality evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('accumulation_acceptance',
    ROOT / 'benchmark/experiments/linear-accumulation/acceptance.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize('damage', [None, 'incomplete', 'relaxed_tolerance', 'binary',
                                   'artifact', 'generation', 'prefix', 'binding'])
def test_accumulation_timing_requires_bound_quality(tmp_path, monkeypatch, damage):
    before, after = tmp_path / 'before.exe', tmp_path / 'after.exe'
    before.write_bytes(b'before'); after.write_bytes(b'after')
    work = tmp_path / 'work'; work.mkdir()
    artifact = work / 'decoder-32.leaf'; artifact.write_bytes(b'weights')
    (work / 'tokens.json').write_text(json.dumps({'benchmark_ids': list(range(64))}))
    tokens = tmp_path / 'tokens.json'
    tokens.write_text(json.dumps({'sequences': [list(range(128))] * 8}))
    quality = dict(complete=True, passed=True, precision='FP32', requested_blocks=8,
        sequence_length=128, executable_sha256=runner.digest(after),
        artifact_sha256=runner.digest(artifact), tokens_sha256=runner.digest(tokens),
        generations=[dict(exact_match=True)] * 6,
        gates=dict(min_next_token_agreement=0.999, max_perplexity_ratio=1.001,
                   logit_atol=2e-4, logit_rtol=2e-4, exact_generation_required=True))
    if damage == 'incomplete': quality['complete'] = False
    if damage == 'relaxed_tolerance': quality['gates']['logit_atol'] = 2e-3
    if damage == 'binary': quality['executable_sha256'] = 'wrong'
    if damage == 'artifact': quality['artifact_sha256'] = 'wrong'
    if damage == 'generation': quality['generations'] = [dict(exact_match=False)] * 6
    if damage == 'prefix': (work / 'tokens.json').write_text(json.dumps({'benchmark_ids': [99] * 64}))
    if damage == 'binding':
        binding = tmp_path / 'binding.json'; binding.write_text(json.dumps(dict(complete=False, passed=False)))
        quality['exact_rebinding'] = dict(binding_report=str(binding))
    report = tmp_path / 'quality.json'; report.write_text(json.dumps(quality))
    output = tmp_path / 'acceptance.json'
    monkeypatch.setattr(sys, 'argv', ['acceptance', '--before', str(before), '--after', str(after),
        '--quality', str(report), '--workdir', str(work), '--output', str(output)])
    invoked = []

    def bench(*args, **kwargs):
        invoked.append(list(sys.argv))
        output.write_text(json.dumps(dict(complete=True, gate=dict(passed=True), hashes={})))
    monkeypatch.setattr(runner.runpy, 'run_path', bench)
    if damage:
        with pytest.raises(ValueError): runner.main()
        assert not invoked and not output.exists()
    else:
        runner.main()
        result = json.loads(output.read_text())
        assert result['bound_accumulation_quality']['passed']
        assert len(invoked) == 1
        for flag, value in [('--target-phase', 'prefill'), ('--pairs', '6'), ('--runs', '31'), ('--warmup', '10')]:
            assert invoked[0][invoked[0].index(flag) + 1] == value
