import json

import numpy as np
import pytest

from tools.decoder_plan import config_values, make_plan, validate_plan
from tools.export_decoder import HEADER, DESCRIPTOR, channel_factors, export_decoder, quantize, read_tensor, tensor_index, transform_array


def test_unknown_architecture_requires_explicit_plan():
    with pytest.raises(ValueError, match="Supply --plan"):
        make_plan({"model_type": "custom_decoder"})


def test_scaled_rope_is_rejected_instead_of_dropped():
    with pytest.raises(ValueError, match="Scaled RoPE"):
        make_plan({"model_type": "llama", "rope_parameters": {"rope_type": "linear", "factor": 2}})


def test_dense_mistral_plan_preserves_gqa_and_rope():
    source = dict(model_type="mistral", hidden_size=64, intermediate_size=128,
                  num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                  vocab_size=128, sliding_window=None, rope_theta=1000000.0,
                  rms_norm_eps=1e-5)
    plan = make_plan(source)
    validate_plan(plan)
    assert plan["config"]["kv_heads"] == 2
    assert plan["config"]["theta"] == 1000000.0
    assert plan["config"]["epsilon"] == 1e-5
    assert plan["config"]["gated"] == 1
    assert plan["config"]["activation"] == 0
    assert not any(name.endswith(".bias") for name in plan["tensors"])


def test_mistral_sliding_window_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="Sliding-window"):
        make_plan(dict(model_type="mistral", hidden_size=64, intermediate_size=128,
                       num_attention_heads=4, num_hidden_layers=2, vocab_size=128,
                       sliding_window=8))


@pytest.mark.parametrize("kind", ["gemma", "gemma2", "gemma3_text", "ministral3"])
def test_unimplemented_model_families_cannot_reuse_llama_semantics(kind):
    with pytest.raises(ValueError, match="No import adapter"):
        make_plan(dict(model_type=kind, hidden_size=64, intermediate_size=128,
                       num_attention_heads=4, num_hidden_layers=2, vocab_size=128))


def test_tensor_layout_transform_preserves_interleaved_qkv():
    array = np.arange(2 * 3 * 4 * 8).reshape(24, 8)
    actual = transform_array(array, [{"reshape": [2, 3, 4, 8]},
                                    {"slice": [1, 1, 2]}, {"reshape": [8, 8]}])
    np.testing.assert_array_equal(actual, array.reshape(2, 3, 4, 8)[:, 1].reshape(8, 8))


@pytest.mark.parametrize("bits", [8, 4])
def test_symmetric_quantization_handles_zero_rows_and_both_signs(bits):
    source = np.array([[0] * 8, [-7, -4, -2, -1, 1, 2, 4, 7]], dtype=np.float32)
    packed, scales = quantize(source, bits, 8)
    if bits == 4:
        unpacked = np.empty(source.shape, dtype=np.int8)
        unpacked[:, ::2] = packed & 15
        unpacked[:, 1::2] = packed >> 4
        unpacked[unpacked >= 8] -= 16
    else:
        unpacked = packed
    restored = unpacked.astype(np.float32) * scales
    assert np.isfinite(scales).all() and (scales > 0).all()
    np.testing.assert_allclose(restored, source, atol=0.03)


def test_bfloat16_source_is_read_without_pytorch(tmp_path):
    import struct
    source = np.array([1, -2, 0.5, 0], dtype="<f4")
    data = (source.view("<u4") >> 16).astype("<u2").tobytes()
    header = json.dumps({"tensor": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(data)]}}).encode()
    filename = tmp_path / "weights.safetensors"
    filename.write_bytes(struct.pack("<Q", len(header)) + header + data)
    base, index = tensor_index(filename)
    np.testing.assert_array_equal(read_tensor(filename, base, index["tensor"]), source.reshape(2, 2))


def test_invalid_weight_range_is_rejected(tmp_path):
    filename = tmp_path / "truncated.safetensors"
    filename.write_bytes(b"1234")
    with pytest.raises(ValueError, match="header"):
        tensor_index(filename)


def test_plan_geometry_cannot_silently_repeat_wrong_heads():
    config = dict(hidden=64, heads=4, head_dim=16, kv_heads=3)
    with pytest.raises(ValueError, match="head geometry"):
        config_values(config)


def test_channel_smoothing_preserves_unquantized_linear_outputs():
    rng = np.random.default_rng(42)
    weight = rng.normal(size=(12, 16)).astype(np.float32)
    inputs = rng.normal(size=(7, 16)).astype(np.float32)
    maximum = np.max(np.abs(inputs), axis=0)
    maximum[0] = 0  # dead calibration channels must not produce infinity
    factors = channel_factors(weight, maximum, 0.5)
    np.testing.assert_allclose((inputs / factors) @ (weight * factors).T,
                               inputs @ weight.T, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("maximum", [np.array([np.nan] * 8), np.array([-1.] * 8)])
def test_bad_calibration_is_rejected(maximum):
    with pytest.raises(ValueError, match="calibration"):
        channel_factors(np.ones((8, 8)), maximum, 0.5)


def small_plan():
    return make_plan(dict(model_type="llama", hidden_size=16, intermediate_size=32,
                          num_attention_heads=2, num_hidden_layers=1, vocab_size=32))


def test_custom_plan_cannot_silently_discard_unknown_semantics():
    plan = small_plan()
    plan["config"]["sliding_window"] = 8
    with pytest.raises(ValueError, match="Unsupported plan configuration"):
        validate_plan(plan)


def test_custom_plan_cannot_silently_discard_unknown_operator_weights():
    plan = small_plan()
    plan["tensors"]["model.layers.0.self_attn.q_norm.weight"] = {"source": "qnorm.weight"}
    with pytest.raises(ValueError, match="Unsupported plan tensor"):
        validate_plan(plan)


def test_multiple_eos_is_not_silently_collapsed():
    with pytest.raises(ValueError, match="Multiple EOS"):
        make_plan(dict(model_type="llama", hidden_size=16, intermediate_size=32,
                       num_attention_heads=2, num_hidden_layers=1, vocab_size=32, eos_token_id=[1, 2]))


def test_no_eos_is_not_converted_to_token_zero():
    config = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_hidden_layers=1, vocab_size=32, eos_token_id=None)
    plan = make_plan(config)
    assert plan["config"]["eos"] == 0xffffffff
    validate_plan(plan)


def test_nonrotary_odd_head_geometry_is_supported():
    config = dict(model_type="gpt2", n_embd=30, n_head=10, n_layer=1, vocab_size=32)
    validate_plan(make_plan(config))


def test_export_preserves_tied_weight_storage(tmp_path):
    import struct
    from tools.decoder_plan import tensor_shapes
    source = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_hidden_layers=1, vocab_size=32, tie_word_embeddings=True)
    (tmp_path / "config.json").write_text(json.dumps(source))
    plan = make_plan(source)
    entries, chunks, offset = {}, [], 0
    for name, shape in tensor_shapes(plan["config"]).items():
        if name == "lm_head.weight":
            continue
        data = np.ones(shape, dtype="<f4").tobytes()
        entries[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    header = json.dumps(entries).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"".join(chunks))
    artifact = tmp_path / "tied.leaf"
    record = export_decoder(tmp_path, artifact)
    blob = artifact.read_bytes()
    position, descriptors = HEADER.size, {}
    for _ in range(record["tensor_count"]):
        length = struct.unpack_from("<I", blob, position)[0]
        position += 4
        name = blob[position:position + length].decode()
        position += length
        descriptors[name] = DESCRIPTOR.unpack_from(blob, position)
        position += DESCRIPTOR.size
    assert descriptors["model.embed_tokens.weight"] == descriptors["lm_head.weight"]
    assert record["unique_payloads"] == record["tensor_count"] - 1


def test_gpt2_adapter_declares_base_checkpoint_aliases():
    plan = make_plan(dict(model_type="gpt2", n_embd=16, n_head=2, n_layer=1, vocab_size=32))
    validate_plan(plan)
    assert plan["tensors"]["model.embed_tokens.weight"]["source"] == "transformer.wte.weight"
    assert plan["tensors"]["model.embed_tokens.weight"]["source_aliases"] == ["wte.weight"]
    assert plan["tensors"]["model.layers.0.self_attn.q_proj.weight"]["source_aliases"] == ["h.0.attn.c_attn.weight"]


@pytest.mark.parametrize("aliases", [None, [], "weight", [0], [""]])
def test_bad_declarative_aliases_are_rejected(aliases):
    plan = small_plan()
    plan["tensors"]["model.embed_tokens.weight"]["source_aliases"] = aliases
    with pytest.raises(ValueError, match="source aliases"):
        validate_plan(plan)


def test_explicit_alias_exports_identical_payload_and_rejects_ambiguity(tmp_path):
    import struct
    from tools.decoder_plan import tensor_shapes
    config = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_hidden_layers=1, vocab_size=32)
    (tmp_path / "config.json").write_text(json.dumps(config))
    plan = make_plan(config)
    entries, payloads, offset = {}, [], 0
    for name, shape in tensor_shapes(plan["config"]).items():
        data = np.ones(shape, dtype="<f4").tobytes()
        entries[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        payloads.append(data); offset += len(data)
    def save():
        header = json.dumps(entries).encode()
        (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"".join(payloads))
    save()
    reference = tmp_path / "reference.leaf"
    export_decoder(tmp_path, reference)
    original = "model.embed_tokens.weight"
    entries["external.embedding"] = entries.pop(original)
    save()
    plan["tensors"][original]["source_aliases"] = ["external.embedding"]
    filename = tmp_path / "custom-plan.json"
    filename.write_text(json.dumps(plan))
    candidate = tmp_path / "aliased.leaf"
    export_decoder(tmp_path, candidate, plan_path=filename)
    assert candidate.read_bytes() == reference.read_bytes()
    entries["another.embedding"] = entries["external.embedding"]
    save()
    plan["tensors"][original]["source_aliases"].append("another.embedding")
    filename.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Ambiguous source aliases"):
        export_decoder(tmp_path, candidate, plan_path=filename)
