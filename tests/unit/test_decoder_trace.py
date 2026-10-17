"""Reference hooks must preserve inference and respect the rotary token axis."""
import importlib

import numpy as np
import pytest
import torch
from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM)

from tools.trace_decoder_logits import STAGES, capture_reference, compare, first_row


def test_trace_comparison_keeps_strict_near_zero_limit():
    expected = np.array([0.0, 3.0], dtype=np.float32)
    assert compare(np.array([0.00019, 3.0]), expected)['allclose']
    assert compare(np.array([0.00021, 3.0]), expected)['tolerance_violations'] == 1
    with pytest.raises(ValueError, match='Invalid trace tensor'):
        compare([float('nan')], [0.0])


def test_rotary_first_row_includes_every_head_at_position_zero():
    tensor = np.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
    np.testing.assert_array_equal(first_row(tensor, rotary=True), tensor[:, :, 0, :])
    assert first_row(tensor, rotary=True).size == 8


def tiny_model(family):
    if family == 'gpt_neox':
        config = GPTNeoXConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                              num_hidden_layers=2, vocab_size=32, max_position_embeddings=32,
                              rotary_pct=0.5)
        cls = GPTNeoXForCausalLM
    else:
        config_cls, cls = ((LlamaConfig, LlamaForCausalLM) if family == 'llama'
                           else (Qwen2Config, Qwen2ForCausalLM))
        config = config_cls(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                            num_key_value_heads=1, num_hidden_layers=2, vocab_size=32,
                            max_position_embeddings=32)
    config._attn_implementation = 'eager'
    return cls(config).eval()


@pytest.mark.parametrize('family', ['llama', 'qwen2', 'gpt_neox'])
def test_all_trace_stages_match_uninstrumented_reference(family):
    torch.set_num_threads(1)
    torch.manual_seed(713)
    model = tiny_model(family)
    row = [1, 5, 7]
    module = importlib.import_module(f'transformers.models.{family}.modeling_{family}')
    original = module.eager_attention_forward
    with torch.inference_mode():
        expected = model(torch.tensor([row]), use_cache=False).logits.numpy().copy()
    captured = {}
    result, layers = capture_reference(model, row, lambda name, value: captured.__setitem__(name, value))
    assert (result, layers) == (family, 2)
    assert set(captured) == {'embedding', 'final_norm', 'logits'} | {
        f'model.layers.{index}.{stage}' for index in range(2) for stage in STAGES}
    np.testing.assert_array_equal(captured['logits'], expected)
    assert captured['model.layers.0.q_rotary'].shape[2] == len(row)
    assert module.eager_attention_forward is original
    assert all(not item._forward_hooks and not item._forward_pre_hooks for item in model.modules())


def test_trace_restores_reference_after_failed_forward(monkeypatch):
    model = tiny_model('llama')
    module = importlib.import_module('transformers.models.llama.modeling_llama')
    original = module.eager_attention_forward

    def fail(*args, **kwargs):
        raise RuntimeError('interrupted reference')
    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='interrupted reference'):
        capture_reference(model, [1, 2], lambda *args: None)
    assert module.eager_attention_forward is original
    assert all(not item._forward_hooks and not item._forward_pre_hooks for item in model.modules())
