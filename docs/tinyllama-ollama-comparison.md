# TinyLlama FP32: Leaf and Ollama — October 15, 2026

Leaf has a lower measured cached-decode median than Ollama on this Windows
workload, but a higher prefill median. A selected-token log-probability check
narrowly misses the declared tolerance, so these remain diagnostic timings;
there is no qualified overall speed or quality claim.

## Matched execution

The model is TinyLlama-1.1B-Chat-v1.0. Both runtimes use FP32 weights and KV
cache, one inference thread on logical CPU 2, Above Normal priority, and no GPU.
The comparison uses the October 15 Leaf default before FP32 SiLU integration,
without the exact-erf GELU or
final-attention-row experiments. Ollama is the official Windows v0.40.4 CPU
runtime, launched as a separate temporary localhost server with its own model
directory. It is unloaded before each Leaf pass.

The F32 GGUF was converted using llama.cpp b11540. An
[audit of all 201 parameters](../benchmark/results/tinyllama-ollama-fp32/gguf-audit.json)
finds exact equality with the source tensors after the standard Q/K RoPE
permutation. Ollama's imported blob hash matches the audited GGUF. Its loaded
model reports zero VRAM usage. The
[runtime provenance](../benchmark/results/tinyllama-ollama-fp32/runtime-provenance.json)
records per-file hashes: the official archive's CPU subset was fetched over
HTTPS and ZIP CRC checked; the whole archive SHA256 was not verified.

Two frozen 64-token prompts alternate immediately after their common BOS,
preventing reuse of their content. Both runtimes reuse exactly **one cached BOS
and compute 63 uncached prompt tokens**. Each generates two output tokens:
the first comes from prefill; the generation phase performs **one cached
decoder forward**. Ollama's `eval_count=2` is therefore not a divisor for
reporting one-forward latency. Backend tokenization, cached counts and output
counts are checked. Raw mode, greedy selection, repeat penalty 1 and empty
textual stops avoid template and sampling differences.

Six balanced Leaf/Ollama pairs run serially, with three warmups and eleven
measurements per pass. All samples are retained. Timings exclude model loading,
HTTP overhead and the initial BOS forward. They are not full-command latency.

## Results

| Phase | Leaf pooled median | Ollama pooled median | Leaf / Ollama | All pass and pooled spread checks |
|---|---:|---:|---:|---|
| 63 uncached prefill tokens | 1,872.24065 ms | 1,516.171 ms | 1.23485 | Pass |
| One cached forward / generation phase | 218.8493 ms | 251.9945 ms | 0.86847 | Pass |

The observed decode median is **13.15% lower**, while prefill is **23.48%
higher**. Median paired ratios are 0.87776 and 1.23752 respectively. The
declared prefill criterion fails; decode passes its phase criterion. These
are separate phase observations, not an overall advantage.

All eight preflight greedy continuations match as text, and every timed
continuation matches. One Unicode prompt has selected-token log-probability
maximum absolute error **0.0002179475**, exceeding the unchanged **0.0002**
limit. The other seven checks pass. This is not full-logit or perplexity
qualification. The strict run correctly stopped at preflight; the separate
timing run explicitly retained the failed quality verdict.

Evidence: [strict preflight](../benchmark/results/tinyllama-ollama-fp32/preflight.json),
[complete diagnostic comparison](../benchmark/results/tinyllama-ollama-fp32/diagnostic-comparison.json).
The latter contains every measured and warmup sample, source/config/artifact
hashes, workloads and gates. Windows power events place the later standby
interval after this run finished; it did not overlap these timings.

## Reproduce and follow up

A requested rerun on October 15 was interrupted by a mains-power failure.
Its partial local files remain in `build/ollama-20261015/diagnostic-c/`, with an
[explicit invalidation record](../benchmark/results/tinyllama-ollama-fp32/rerun-power-loss.json).
Those samples do not replace the completed comparison above. The updated
runner and native adapter reject offline/unknown AC state; a fresh rerun waits
for mains power. The next Leaf candidate enables the existing FP32 vector
[SiLU gate](silu-gate-experiment.md#fp32-follow-up--october-15-2026), whose trained
correctness and native incremental latency gates now pass. The user requested
a checkpoint push before another long Ollama run; no fresh post-integration
Ollama speedup is claimed.

`tools/benchmark_ollama_decoder.py --help` describes the runner's explicit local
runtime, snapshot, audited GGUF and Leaf artifact inputs.
`benchmark/decoder_generation_benchmark.cpp` links the actual Leaf decoder;
`tools/ollama_validation.py` owns the temporary Ollama server. No existing user
service or global power setting is changed.

Before claiming a matched win, investigate the probability discrepancy without
relaxing its limit, and measure the newly integrated FP32 SiLU default. The
[diagnostic startup-affinity race](windows-timing-diagnosis.md#october-15-2026-diagnostic-startup-race)
has been corrected so future profiles do not mix P-core and E-core work. This comparison does not
establish performance for other models, operating systems or quantized modes.

Protocol references: [Ollama generation API](https://docs.ollama.com/api/generate),
[GGUF import](https://docs.ollama.com/import).
