# Smol/Qwen FP32 comparison, October 19

Six balanced pairs, 66 measured samples/runtime/model, same 64 input tokens
and two output tokens, CPU 2, one thread, FP32 weights/KV. Every pass and pooled
spread passes; visible generated outputs match. The accurate experimental Leaf
binary has lower prefill/decode medians on both models.

**The original strict selected-log-probability preflight fails**, so overall
speed claims remain false. Original preflights, raw quality responses and explicit
diagnostic comparisons are all retained. The independent PyTorch diagnostic on
the disputed prompts places Leaf within 0.0000125 of reference, while Ollama's
reported probabilities differ more. This does not override the original gate.

[Interpretation, binaries, power audit and next implementation](../../../docs/checkpoint-2026-10-19-linear-and-storage.md).

`tools/prepare_ollama_comparison.py` provides separate `convert`, `audit`,
`import`, and `export-leaf` stages. Each needs `--snapshot` and `--output`.
Conversion is offline via the pinned local llama.cpp converter. Auditing verifies
every source tensor; import uses a separate local Ollama store/server. The Leaf
export stage can require `--reference-artifact` exact equality before writing
source provenance. No benchmark verification requirement was bypassed.
