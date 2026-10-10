import pytest
from tools.export_embedding import embedding_config


def config():
    return dict(
        model_type="gemma3_text",
        use_bidirectional_attention=True,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=96,
        head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=16,
    )


def test_window_is_bidirectional_radius_and_head_width_is_independent():
    dims, constants, types = embedding_config(config())
    assert dims[-1] == 8
    assert dims[0] != dims[2] * dims[7]
    assert constants[-1] == 0.25
    assert types == ["sliding_attention", "full_attention"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_bidirectional_attention", False),
        ("attention_bias", True),
        ("hidden_activation", "silu"),
        ("attn_logit_softcapping", 30),
        ("rope_scaling", {"factor": 2}),
        ("layer_types", ["sliding_attention"]),
        ("num_key_value_heads", 3),
        ("sliding_window", 0),
        ("rope_parameters", {"full_attention": {"rope_type": "linear"}}),
    ],
)
def test_unsupported_semantics_are_not_silently_dropped(field, value):
    source = config()
    source[field] = value
    with pytest.raises(ValueError):
        embedding_config(source)
