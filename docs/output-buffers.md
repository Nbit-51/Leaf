# Reusing decoder output storage

The C++ decoder can overwrite caller-owned logits storage while preserving its
capacity across calls:

```cpp
leaf::runtime::Decoder decoder("model.leaf");
std::vector<float> logits;
decoder.forward_into({1, 2, 3}, logits); // Prefill, one full vocabulary row.
decoder.forward_into({4}, logits);       // Decode, same allocation when capacity fits.
```

`forward_into(tokens, logits, true)` writes complete `[tokens, vocabulary]`
logits. Otherwise it writes the final vocabulary row, exactly like `forward`.
The destination is resized and overwritten. The caller owns its lifetime and
can retain or switch buffers. Contents after an exception are unspecified.
The decoder's KV state still advances, and a session is not reentrant.

The original owned-return `forward()` API remains compatible. It creates an
independent vector and delegates to the same computation. The native command
uses one reusable destination across generation, verification and benchmarking;
it does not skip any output projection or change math, precision or tiling.
Metrics expose `reusable_logits: true`, and `--runtime-info` advertises
`reusable_logits_supported`.

The artifact-backed `leaf_decoder_output_buffer_tests` CMake target compares
both APIs bit for bit, tests full and final-row logits, cache/reset behavior,
capacity/pointer reuse, growth/shrink, independent old results, destination
switching, exceptions and caller-affinity restoration. Pass a local unquantized
artifact as its first argument; it never downloads models.

Use [optional benchmark accounting](benchmark-counters.md) to inspect allocation
and fault behavior. Page-fault reduction is distinct from a qualified latency
improvement; raw Windows fault counts include soft and hard faults.

## October 27 Windows evidence

The current runtime passes the complete retained accurate-reference migration
(16 blocks, 12 continuations, ten trained controls and eight earlier default
regressions), 1,350 unit tests with two optional skips, and 130 API comparisons.
The isolated installed wheel runs accurate FP32 + source-BF16 head storage +
explicit workers without a compiler, decoder override, Torch or Transformers.
Its executable code/constants match the trained-qualified cached runtime.
These are reference-rebinding checks, not a new independent PyTorch evaluation.

A two-pair, seven-sample-per-pass diagnostic compares the old and reused-output
executables with identical accurate math, FP32 artifacts and CPU2 on this host.
Windows remains in the user's `better-battery` effective mode. Both models pass
every individual/pooled p90/p10 spread check in this short screen:

| Model | Prefill pooled median, old -> reuse | Decode pooled median, old -> reuse | Faults per phase sample, old -> reuse |
|---|---:|---:|---:|
| SmolLM2-360M | 483.402 -> 485.033 ms | 78.567 -> 78.484 ms | 0 -> 0 |
| Qwen2.5-0.5B | 557.804 -> 556.228 ms | 107.261 -> 107.239 ms | 148 -> 0 |

This validates removal of the repeated allocation-related faults for Qwen in
these cases. Pooled latency changes are below 0.4%; no material speedup or release
acceptance is claimed. The faults were not evidence of a disk stall or the
primary prefill bottleneck. This also does not resolve historical long-run
stability failures. The default math policy and artifact format are unchanged.

[Summary](../benchmark/validation/benchmark-counters/20261027/summary.json) and
[raw evidence manifest](../benchmark/validation/benchmark-counters/20261027/manifest.json)
bind the executables, tests, source snapshots and unfiltered diagnostic records.
