"""Lightweight, structure-based nonlegacy tokenizer compatibility."""
from __future__ import annotations

import json
import sys

import pytest

from leaf import cli


@pytest.fixture
def snapshot(tmp_path):
    from tokenizers import AddedToken, Tokenizer, decoders, normalizers
    from tokenizers.models import BPE
    from tokenizers.processors import TemplateProcessing

    vocabulary = {"[UNK]": 0, "<s>": 1, "</s>": 2, "\u2581": 3, "a": 4, "b": 5, "\n": 6}
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token="[UNK]", byte_fallback=True))
    tokenizer.add_special_tokens([AddedToken("<s>", normalized=False, special=True),
                                  AddedToken("</s>", normalized=False, special=True)])
    tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend("\u2581"), normalizers.Replace(" ", "\u2581")])
    tokenizer.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(),
                                           decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)])
    tokenizer.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = {"tokenizer_class": "AnUnrelatedTokenizerName", "legacy": False, "eos_token": "</s>",
              "chat_template": "{{ messages[0]['content'] }}{{ eos_token }}\nb"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


def update_config(snapshot, **changes):
    path = snapshot / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def test_nonlegacy_special_token_does_not_prepend_another_space(snapshot):
    before = set(sys.modules)
    tokenizer, ids = cli.encode_prompt(snapshot, "a")
    assert ids == [3, 4, 2, 6, 5]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "a\nb"
    state = json.loads(tokenizer.pre_tokenizer.__getstate__())
    assert state == {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": False}
    assert tokenizer.normalizer is None
    assert not any(name == "torch" or name.startswith("torch.") or name == "transformers" or
                   name.startswith("transformers.") for name in set(sys.modules) - before)


@pytest.mark.parametrize("value", [True, None])
def test_default_or_explicit_prefix_space_preserves_initial_prefix(snapshot, value):
    update_config(snapshot, add_prefix_space=value)
    assert cli.encode_prompt(snapshot, "a")[1] == [3, 4, 2, 6, 5]


def test_no_prefix_policy_updates_encoder_and_canonical_decoder(snapshot):
    update_config(snapshot, add_prefix_space=False)
    tokenizer, ids = cli.encode_prompt(snapshot, "a")
    assert ids == [4, 2, 6, 5]
    assert tokenizer.decode([3, 4]) == " a"  # no unconditional removal of genuine leading whitespace
    assert json.loads(tokenizer.pre_tokenizer.__getstate__())["prepend_scheme"] == "never"


@pytest.mark.parametrize("changes", [{"legacy": True}, {"legacy": None}])
def test_legacy_or_unspecified_policy_keeps_serialized_backend(snapshot, changes):
    update_config(snapshot, **changes)
    tokenizer, ids = cli.encode_prompt(snapshot, "a")
    assert ids == [3, 4, 2, 3, 6, 5]
    assert tokenizer.pre_tokenizer is None


def test_raw_input_still_uses_existing_special_token_postprocessor(snapshot):
    assert cli.encode_prompt(snapshot, "a", raw=True)[1] == [1, 3, 4]


def test_modern_metaspace_backend_is_not_replaced(snapshot):
    from tokenizers import Tokenizer, pre_tokenizers
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="first", split=False)
    tokenizer.save(str(snapshot / "tokenizer.json"))
    loaded, ids = cli.encode_prompt(snapshot, "a")
    assert ids == [3, 4, 2, 6, 5]
    assert loaded.normalizer is None


def test_unrelated_normalizer_is_not_removed(snapshot):
    from tokenizers import Tokenizer, normalizers
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.Prepend("\u2581")])
    original = tokenizer.normalizer.__getstate__()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    loaded, _ = cli.encode_prompt(snapshot, "a")
    assert loaded.normalizer.__getstate__() == original
    assert loaded.pre_tokenizer is None


def test_bytelevel_backend_is_not_changed_by_nonlegacy_flag(snapshot):
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    original = tokenizer.pre_tokenizer.__getstate__()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    loaded, _ = cli.encode_prompt(snapshot, "a")
    assert loaded.pre_tokenizer.__getstate__() == original
    assert loaded.decoder.__getstate__() == tokenizer.decoder.__getstate__()


def test_custom_decoder_is_preserved(snapshot):
    from tokenizers import Tokenizer, decoders
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.decoder = decoders.Fuse()
    original = tokenizer.decoder.__getstate__()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    loaded, ids = cli.encode_prompt(snapshot, "a")
    assert ids == [3, 4, 2, 6, 5]
    assert loaded.decoder.__getstate__() == original


def test_malformed_prefix_flag_fails_closed(snapshot):
    update_config(snapshot, add_prefix_space="false")
    with pytest.raises(ValueError, match="boolean or null"):
        cli.encode_prompt(snapshot, "a")
