# Decisions

ADR-style log of design tradeoffs. Newest entries at the top.

---

## ADR-002: im2col scratch buffer reuse via thread_local static

**Date:** 2026-08-19
**Status:** Accepted
**Related:** Precision-Safe Optimization Queue item 3 (partial — activation
buffers in executor.cpp not yet addressed, see Open Follow-ups below)

### Context

`conv2d()` allocated a fresh `std::vector<float> col(patch_size * out_spatial)`
on every call. In `leaf_bench`'s repeated-run loop this meant the same
heap allocation and deallocation happened on every layer, every forward
pass, despite `patch_size * out_spatial` being fixed once the model shape
is known. Measured baseline (M5a, AVX2 4x8 GEMM): 140.904 ms mean latency.

### Options considered

1. **Executor-owned buffer pool.** Add a `mutable` cache keyed by node
   output name (or shape) on the `Executor` class, sized once and reused
   across `run()` calls. Correct long-term direction — this is what a
   proper liveness-based memory plan (M2b) will eventually need — but
   `Executor::run()` currently has no per-layer object to attach a buffer
   to; the graph is walked node-by-node with no persistent per-conv state.
   Wiring this in properly means either changing `Executor`'s API surface
   or threading a workspace object through every op, which is a bigger
   change than the immediate problem warrants.
2. **Explicit workspace object passed into `conv2d()`.** Caller
   (executor) owns and passes a scratch buffer explicitly. Cleaner
   ownership story than (1) but still requires plumbing a new parameter
   through `conv2d()`'s signature and every call site.
3. **`thread_local static std::vector<float>` inside `conv2d()`, grow-only.**
   No API changes anywhere. Buffer persists across calls within a thread,
   only grows (never shrinks) to fit the largest patch matrix seen so far.
   Safe because `im2col()` fully overwrites `[0, needed)` before `gemm_f32`
   reads it on every call — no stale data can leak between layers or runs.

### Decision

Chose option 3. It isolates the fix to a single file (`conv2d.cpp`), keeps
`Executor::run()`'s `const` public API and the existing test suite's
call sites untouched, and captures the majority of the win with the
least risk. Options 1 and 2 remain the correct target for M2b's
liveness-based memory plan, where the same reuse logic should extend to
*activation* buffers in `executor.cpp` (every `Tensor output` per node
is still a fresh allocation per run — see Open Follow-ups).

### Result

Verified via `ctest` (3/3 pass) and `tools/verify_cpp_runtime.py`
(full ResNet-18 vs PyTorch parity, max abs diff `3.93e-06`, unchanged
from M5a baseline). Benchmark:

| Stage | Mean latency | Throughput | Parity |
|---|---:|---:|---|
| M5a baseline | 140.904 ms | 7.097 img/s | 3.93e-06 |
| + im2col buffer reuse | 106.812 ms | 9.362 img/s | 3.93e-06 |

24.2% latency reduction, 31.9% throughput gain, zero numerical impact.
Logged in `benchmark/results/wsl_dev_machine.csv`.

### Open follow-ups

- `executor.cpp`'s op loop still heap-allocates a fresh `Tensor` (and
  therefore a fresh `std::vector<float>`) per node per `run()` call.
  Same class of problem, different layer. Natural next target, and the
  point where a real liveness-based memory plan (M2b) becomes worth
  building properly rather than patching around with `thread_local`.
- `thread_local` reuse means the buffer is per-thread. Fine for the
  current single-threaded executor; needs revisiting once M5b's
  multi-threading/output-channel-tile parallelization lands, since each
  worker thread will get its own copy (memory cost, not a correctness
  risk — but worth noting before scaling thread count).

---

## ADR-001: (retro-fill as needed)

Earlier milestones (M1a/M1b/M2a/M5a) predate this log. Backfill if the
Conv/BatchNorm/ReLU fusion tradeoffs or the AVX2 4x8 micro-kernel design
choice are worth capturing for the report writeup.
