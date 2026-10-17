"""Declarative decoder plans: model names belong to import adapters only.

The native engine consumes dimensions and block semantics, never a model ID.
Custom plans use the same tensor mappings and safe array transforms as built-in
adapters. Unsupported semantics must be rejected rather than silently dropped.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ACTIVATIONS = {"silu": 0, "gelu": 1, "gelu_new": 2, "gelu_fast": 2, "relu": 3}
MAX_EOS_TOKENS = 1024


def _parse_eos(value, vocab: int) -> list[int]:
    if value is None:
        return []
    tokens = value if isinstance(value, list) else [value]
    if not tokens or len(tokens) > MAX_EOS_TOKENS:
        raise ValueError(f"EOS list must contain 1..{MAX_EOS_TOKENS} tokens")
    if any(type(token) is not int or not 0 <= token < vocab for token in tokens):
        raise ValueError("Malformed or out-of-vocabulary EOS token")
    if len(set(tokens)) != len(tokens):
        raise ValueError("EOS list contains duplicate tokens")
    return list(tokens)


def eos_token_ids(config: dict) -> list[int]:
    """Return the complete termination policy, preserving token zero and no EOS."""
    primary = config["eos"]
    if "eos_tokens" not in config:
        return [] if type(primary) is int and primary == 0xffffffff else _parse_eos(primary, config["vocab"])
    tokens = _parse_eos(config["eos_tokens"], config["vocab"])
    if len(tokens) < 2 or type(primary) is not int or primary != tokens[0]:
        raise ValueError("Multi-EOS plan must preserve its first EOS token")
    return tokens


def _set_eos(config: dict, tokens: list[int]) -> None:
    config["eos"] = tokens[0] if tokens else 0xffffffff
    config.pop("eos_tokens", None)
    if len(tokens) > 1:
        config["eos_tokens"] = list(tokens)


def validate_generation_config(generation: dict, plan: dict) -> None:
    """Fail closed when snapshot termination differs from the effective plan.

    Leaf's generation policy is explicitly greedy with its own max-token
    budget; this guard does not promise to honor other Hugging Face defaults.
    """
    validate_plan(plan)
    if not isinstance(generation, dict):
        raise ValueError("generation_config.json must contain an object")
    if "eos_token_id" not in generation:
        return
    tokens = _parse_eos(generation["eos_token_id"], plan["config"]["vocab"])
    if tokens != eos_token_ids(plan["config"]):
        raise ValueError("generation_config.json EOS differs from the decoder plan")


def validate_snapshot_generation(snapshot: Path, plan: dict) -> None:
    validate_plan(plan)
    filename = snapshot / "generation_config.json"
    if filename.is_file():
        validate_generation_config(json.loads(filename.read_text(encoding="utf-8")), plan)


def load_snapshot_plan(snapshot: Path, plan_path: Path | None = None) -> dict:
    """Resolve model metadata and generation EOS once for every execution path.

    Built-in plans honor an explicit generation_config.json EOS override.
    Custom plans remain authoritative and must match that override. Other Hub
    generation defaults are outside Leaf's explicit greedy policy.
    """
    if plan_path is not None:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = make_plan(json.loads((snapshot / "config.json").read_text(encoding="utf-8")))
        generation_file = snapshot / "generation_config.json"
        if generation_file.is_file():
            generation = json.loads(generation_file.read_text(encoding="utf-8"))
            if not isinstance(generation, dict):
                raise ValueError("generation_config.json must contain an object")
            if "eos_token_id" in generation:
                _set_eos(plan["config"], _parse_eos(generation["eos_token_id"], plan["config"]["vocab"]))
    validate_snapshot_generation(snapshot, plan)
    return plan


def greedy_reference_tokens(model, prompt_ids: list[int], plan: dict, max_new_tokens: int) -> list[int]:
    """Full-prefix argmax oracle with no inherited Hub generation defaults.

    Transformers can fill unset fields of a fresh GenerationConfig from the
    model's defaults, including EOS, forced tokens and logit processors. Direct
    forwards keep this reference explicitly greedy and independent of those
    settings. The native path uses its KV cache and the same stop-token list.
    """
    import torch
    validate_plan(plan)
    if type(max_new_tokens) is not int or max_new_tokens <= 0 or not prompt_ids:
        raise ValueError("Greedy reference requires a prompt and positive token budget")
    if len(prompt_ids) + max_new_tokens - 1 > plan["config"]["max_positions"]:
        raise ValueError("Generation check exceeds model context")
    stop_tokens = set(eos_token_ids(plan["config"]))
    ids, generated = list(prompt_ids), []
    for _ in range(max_new_tokens):
        output = model(torch.tensor([ids]), use_cache=False)
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        del output
        if token in stop_tokens:
            break
        ids.append(token)
    return generated


def generation_policy_identity(plan: dict, max_new_tokens: int) -> dict:
    return dict(format="leaf-greedy-generation-v1", eos_token_ids=eos_token_ids(plan["config"]),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                repetition_penalty=1.0, use_cache=False)


def make_plan(source: dict) -> dict:
    kind = source.get("model_type")
    sliding_window = 0
    rope = source.get("rope_parameters") or {}
    if source.get("qk_layernorm") or source.get("use_qk_norm"):
        raise ValueError("Query/key normalization needs additional native operators")
    if rope and rope.get("rope_type", "default") != "default":
        raise ValueError("Scaled RoPE is not implemented; refusing to change model semantics")
    if kind == "gpt2":
        h, heads = source["n_embd"], source["n_head"]
        layers, f = source["n_layer"], source.get("n_inner") or 4 * h
        maximum = source.get("n_positions", 1024)
        norm, act, position, gated, parallel = 1, source.get("activation_function", "gelu_new"), 1, 0, 0
        eps, theta, kv, offset, final_norm = source.get("layer_norm_epsilon", 1e-5), 10000, heads, 0, 1
        if (source.get("reorder_and_upcast_attn") or source.get("scale_attn_by_inverse_layer_idx") or
                source.get("scale_attn_weights", True) is False):
            raise ValueError("This GPT2 attention scaling is not implemented by the current native operators")
    elif kind == "gpt_neox":
        h, heads, layers, f = (source[k] for k in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size"))
        maximum = source.get("max_position_embeddings", 2048)
        norm, act, position, gated, parallel = 1, source.get("hidden_act", "gelu"), 0, 0, int(source.get("use_parallel_residual", True))
        eps, theta, kv, offset, final_norm = source.get("layer_norm_eps", 1e-5), rope.get("rope_theta", source.get("rotary_emb_base", 10000)), heads, 0, 1
    elif kind == "opt":
        h, heads, layers, f = (source[k] for k in ("hidden_size", "num_attention_heads", "num_hidden_layers", "ffn_dim"))
        if source.get("word_embed_proj_dim", h) != h or not source.get("do_layer_norm_before", True):
            raise ValueError("OPT projection/post-norm variants require additional plan operators")
        if not source.get("layer_norm_elementwise_affine", True):
            raise ValueError("Non-affine OPT normalization needs additional native operators")
        maximum = source.get("max_position_embeddings", 2048)
        norm, act, position, gated, parallel = 1, source.get("activation_function", "relu"), 1, 0, 0
        eps, theta, kv, offset = source.get("layer_norm_eps", 1e-5), 10000, heads, 2
        final_norm = int(not source.get("_remove_final_layer_norm", False))
    elif kind in {"llama", "qwen2", "mistral"}:
        h, heads, layers, f = (source[k] for k in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size"))
        maximum = source.get("max_position_embeddings", 2048)
        norm, act, position, gated, parallel = 0, source.get("hidden_act", "silu"), 0, 1, 0
        eps, theta = source.get("rms_norm_eps", 1e-6), rope.get("rope_theta", source.get("rope_theta", 10000))
        kv, offset, final_norm = source.get("num_key_value_heads", heads), 0, 1
        if kind == "mistral" and source.get("sliding_window") is not None:
            sliding_window = source["sliding_window"]
            if type(sliding_window) is not int or not 0 < sliding_window <= 0xffffffff:
                raise ValueError("Mistral sliding_window must be a positive integer")
            if source.get("layer_types") is not None and source["layer_types"] != ["sliding_attention"] * layers:
                raise ValueError("Mixed per-layer sliding-window attention is not implemented")
        elif source.get("sliding_window") and source.get("use_sliding_window"):
            raise ValueError("Sliding-window attention is not implemented by the current native operators")
    else:
        raise ValueError(f"No import adapter for {kind!r}. Supply --plan with block semantics and tensor mappings.")
    if source.get("rope_scaling"):
        raise ValueError("Scaled RoPE is not implemented; refusing to change model semantics")
    if act not in ACTIVATIONS:
        raise ValueError(f"Unsupported feed-forward activation {act!r}")
    eos_tokens = _parse_eos(source.get("eos_token_id", 2), source["vocab_size"])
    eos = eos_tokens[0] if eos_tokens else 0xffffffff
    dim = source.get("head_dim", h // heads)
    rotary_dim = int(dim * rope.get("partial_rotary_factor", source.get("rotary_pct", 1.0))) if position == 0 else 0
    config = dict(hidden=h, intermediate=f, heads=heads, kv_heads=kv, layers=layers,
                  vocab=source["vocab_size"], max_positions=maximum, eos=eos,
                  head_dim=dim, theta=theta, epsilon=eps, norm=norm,
                  activation=ACTIVATIONS[act], position=position, gated=gated,
                  parallel_residual=parallel, rotary_dim=rotary_dim,
                  position_offset=offset, final_norm=final_norm,
                  embedding_scale=h ** 0.5 if source.get("scale_embedding", False) else 1.0)
    if sliding_window:
        config["sliding_window"] = sliding_window
    _set_eos(config, eos_tokens)
    tensors = {}

    def add(name, original=None, transforms=None, input_module=None):
        tensors[name] = {"source": original or name, "transforms": transforms or []}
        if input_module:
            tensors[name]["input_module"] = input_module

    if kind in {"llama", "qwen2", "mistral"}:
        add("model.embed_tokens.weight")
        add("lm_head.weight", "model.embed_tokens.weight" if source.get("tie_word_embeddings") else None)
        add("model.norm.weight")
        for i in range(layers):
            prefix = f"model.layers.{i}."
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                         "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                         "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                add(prefix + name)
            if source.get("attention_bias", kind == "qwen2"):
                for p in ("q", "k", "v") + (("o",) if kind != "qwen2" else ()):
                    add(prefix + f"self_attn.{p}_proj.bias")
            if source.get("mlp_bias", False):
                for p in ("gate", "up", "down"):
                    add(prefix + f"mlp.{p}_proj.bias")
    elif kind == "gpt2":
        add("model.embed_tokens.weight", "transformer.wte.weight")
        add("model.position_embeddings.weight", "transformer.wpe.weight")
        add("lm_head.weight", "transformer.wte.weight" if source.get("tie_word_embeddings", True) else "lm_head.weight")
        add("model.norm.weight", "transformer.ln_f.weight"); add("model.norm.bias", "transformer.ln_f.bias")
        for i in range(layers):
            prefix, original = f"model.layers.{i}.", f"transformer.h.{i}."
            for norm_name, norm_source in (("input_layernorm", "ln_1"), ("post_attention_layernorm", "ln_2")):
                for suffix in ("weight", "bias"):
                    add(prefix + norm_name + "." + suffix, original + norm_source + "." + suffix)
            for index, projection in enumerate(("q", "k", "v")):
                add(prefix + f"self_attn.{projection}_proj.weight", original + "attn.c_attn.weight",
                    [{"slice": [1, index * h, (index + 1) * h]}, {"transpose": [1, 0]}])
                add(prefix + f"self_attn.{projection}_proj.bias", original + "attn.c_attn.bias",
                    [{"slice": [0, index * h, (index + 1) * h]}])
            for target, origin in (("self_attn.o_proj", "attn.c_proj"), ("mlp.up_proj", "mlp.c_fc"), ("mlp.down_proj", "mlp.c_proj")):
                add(prefix + target + ".weight", original + origin + ".weight", [{"transpose": [1, 0]}])
                add(prefix + target + ".bias", original + origin + ".bias")
    elif kind == "gpt_neox":
        add("model.embed_tokens.weight", "gpt_neox.embed_in.weight")
        add("lm_head.weight", "gpt_neox.embed_in.weight" if source.get("tie_word_embeddings") else "embed_out.weight",
            input_module="embed_out")
        for suffix in ("weight", "bias"):
            add("model.norm." + suffix, "gpt_neox.final_layer_norm." + suffix)
        for i in range(layers):
            prefix, original = f"model.layers.{i}.", f"gpt_neox.layers.{i}."
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                for suffix in ("weight", "bias"):
                    add(prefix + norm_name + "." + suffix, original + norm_name + "." + suffix)
            for index, projection in enumerate(("q", "k", "v")):
                for suffix in ("weight", "bias"):
                    if suffix == "bias" and not source.get("attention_bias", True):
                        continue
                    shape = [heads, 3, dim, h] if suffix == "weight" else [heads, 3, dim]
                    target = [h, h] if suffix == "weight" else [h]
                    add(prefix + f"self_attn.{projection}_proj." + suffix, original + "attention.query_key_value." + suffix,
                        [{"reshape": shape}, {"slice": [1, index, index + 1]}, {"reshape": target}])
            for target, origin in (("self_attn.o_proj", "attention.dense"), ("mlp.up_proj", "mlp.dense_h_to_4h"), ("mlp.down_proj", "mlp.dense_4h_to_h")):
                for suffix in ("weight", "bias"):
                    if target == "self_attn.o_proj" and suffix == "bias" and not source.get("attention_bias", True):
                        continue
                    add(prefix + target + "." + suffix, original + origin + "." + suffix)
    elif kind == "opt":
        original = "model.decoder."
        add("model.embed_tokens.weight", original + "embed_tokens.weight")
        add("lm_head.weight", original + "embed_tokens.weight" if source.get("tie_word_embeddings", True) else "lm_head.weight")
        add("model.position_embeddings.weight", original + "embed_positions.weight")
        if final_norm:
            for suffix in ("weight", "bias"):
                add("model.norm." + suffix, original + "final_layer_norm." + suffix)
        for i in range(layers):
            prefix, origin = f"model.layers.{i}.", original + f"layers.{i}."
            for target, src in (("input_layernorm", "self_attn_layer_norm"), ("post_attention_layernorm", "final_layer_norm"),
                                ("self_attn.q_proj", "self_attn.q_proj"), ("self_attn.k_proj", "self_attn.k_proj"),
                                ("self_attn.v_proj", "self_attn.v_proj"), ("self_attn.o_proj", "self_attn.out_proj"),
                                ("mlp.up_proj", "fc1"), ("mlp.down_proj", "fc2")):
                for suffix in ("weight", "bias"):
                    if suffix == "bias" and "layernorm" not in target and not source.get("enable_bias", True):
                        continue
                    add(prefix + target + "." + suffix, origin + src + "." + suffix)
    # Official GPT2 base-model safetensors omit the causal wrapper prefix;
    # newly saved causal checkpoints retain it. These explicit import aliases
    # do not affect module names used by calibration or native execution.
    if kind == "gpt2":
        for specification in tensors.values():
            if specification["source"].startswith("transformer."):
                specification["source_aliases"] = [specification["source"].removeprefix("transformer.")]
    return {"format": "leaf-decoder-plan-v1", "config": config, "tensors": tensors, "source_architecture": kind}


def config_values(config: dict) -> tuple:
    fields = {"hidden", "intermediate", "heads", "kv_heads", "layers", "vocab", "max_positions", "eos",
              "head_dim", "theta", "epsilon", "norm", "activation", "position", "gated", "parallel_residual",
              "rotary_dim", "position_offset", "final_norm", "embedding_scale", "sliding_window", "eos_tokens"}
    if set(config) - fields:
        raise ValueError(f"Unsupported plan configuration fields: {sorted(set(config) - fields)}")
    h, heads, dim, kv = (int(config[k]) for k in ("hidden", "heads", "head_dim", "kv_heads"))
    if not kv or not heads or not dim or h != heads * dim or heads % kv:
        raise ValueError("Invalid decoder head geometry")
    missing = fields - {"embedding_scale", "sliding_window", "eos_tokens"} - set(config)
    if missing:
        raise ValueError(f"Missing plan configuration fields: {sorted(missing)}")
    for name in fields - {"theta", "epsilon", "embedding_scale", "sliding_window", "eos_tokens"}:
        if not isinstance(config[name], int) or not 0 <= config[name] <= 0xffffffff:
            raise ValueError(f"Plan dimension/operator {name} must be an unsigned 32-bit integer")
    if any(config[name] not in (0, 1) for name in ("gated", "parallel_residual", "final_norm")):
        raise ValueError("Invalid plan boolean operator")
    if config["eos"] >= config["vocab"] and config["eos"] != 0xffffffff:
        raise ValueError("EOS token is outside the vocabulary")
    eos_token_ids(config)
    if any(not math.isfinite(float(config.get(name, 1.0))) or float(config.get(name, 1.0)) <= 0
           for name in ("theta", "epsilon", "embedding_scale")):
        raise ValueError("Plan numerical constants must be finite and positive")
    rotary = int(config["rotary_dim"])
    if rotary < 0 or rotary > dim or rotary % 2 or config["position"] == 0 and rotary == 0:
        raise ValueError("Invalid rotary dimension")
    if config["norm"] not in (0, 1) or config["activation"] not in range(4) or config["position"] not in (0, 1, 2):
        raise ValueError("Unsupported plan operator")
    first = tuple(int(config[k]) for k in ("hidden", "intermediate", "heads", "kv_heads", "layers", "vocab", "max_positions", "eos", "head_dim"))
    if any(v <= 0 for v in first[:7]):
        raise ValueError("Decoder dimensions must be positive")
    window = config.get("sliding_window", 0)
    if type(window) is not int or not 0 <= window <= 0xffffffff:
        raise ValueError("Plan sliding_window must be an unsigned 32-bit integer")
    return (*first, window, float(config["theta"]), float(config["epsilon"]),
            *(int(config[k]) for k in ("norm", "activation", "position", "gated", "parallel_residual", "rotary_dim", "position_offset", "final_norm")),
            float(config.get("embedding_scale", 1.0)))


def validate_plan(plan: dict) -> None:
    if (not isinstance(plan, dict) or plan.get("format") != "leaf-decoder-plan-v1" or
            set(plan) - {"format", "config", "tensors", "source_architecture"}):
        raise ValueError("Unsupported decoder plan format or fields")
    if not isinstance(plan.get("config"), dict) or not isinstance(plan.get("tensors"), dict):
        raise ValueError("Decoder plan requires config and tensor mapping objects")
    if not {"hidden", "heads", "head_dim", "kv_heads"}.issubset(plan["config"]):
        raise ValueError("Decoder plan is missing head geometry")
    config_values(plan["config"])
    expected = tensor_shapes(plan["config"])
    for name, specification in plan["tensors"].items():
        partner = name.removesuffix(".bias") + ".weight" if name.endswith(".bias") else name
        if partner not in expected:
            raise ValueError(f"Unsupported plan tensor/operator: {name}")
        if (not isinstance(specification, dict) or set(specification) - {"source", "transforms", "input_module", "source_aliases"} or
                not isinstance(specification.get("source"), str)):
            raise ValueError(f"Invalid tensor mapping for {name}")
        if "source_aliases" in specification and (not isinstance(specification["source_aliases"], list) or
                not specification["source_aliases"] or
                any(not isinstance(alias, str) or not alias for alias in specification["source_aliases"])):
            raise ValueError(f"Invalid source aliases for {name}")
        if "input_module" in specification and not isinstance(specification["input_module"], str):
            raise ValueError(f"Invalid calibration module mapping for {name}")
    if set(expected) - set(plan["tensors"]):
        raise ValueError("Plan is missing required native operator tensors")


def tensor_shapes(config: dict) -> dict:
    h, f, kv, dim, layers, vocab = (config[k] for k in ("hidden", "intermediate", "kv_heads", "head_dim", "layers", "vocab"))
    expected = {"model.embed_tokens.weight": (vocab, h), "lm_head.weight": (vocab, h)}
    if config["final_norm"]:
        expected["model.norm.weight"] = (1, h)
    if config["position"] == 1:
        expected["model.position_embeddings.weight"] = (config["max_positions"] + config["position_offset"], h)
    for i in range(layers):
        prefix = f"model.layers.{i}."
        expected.update({prefix + "input_layernorm.weight": (1, h), prefix + "post_attention_layernorm.weight": (1, h),
                         prefix + "self_attn.q_proj.weight": (h, h), prefix + "self_attn.k_proj.weight": (kv * dim, h),
                         prefix + "self_attn.v_proj.weight": (kv * dim, h), prefix + "self_attn.o_proj.weight": (h, h),
                         prefix + "mlp.up_proj.weight": (f, h), prefix + "mlp.down_proj.weight": (h, f)})
        if config["gated"]:
            expected[prefix + "mlp.gate_proj.weight"] = (f, h)
    return expected
