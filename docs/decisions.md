# Decisions

ADR-style log of design tradeoffs. Newest entries at the top.

---

## ADR-003: thread_local activation-buffer pool in Executor::run()

**Date:** 2026-08-20
**Status:** Accepted
**Related:** Direct continuation of ADR-002's "Open follow-ups" (executor.cpp
per-node `Tensor` allocation); Precision-Safe Optimization Queue item 3.

### Context

ADR-002 fixed the im2col scratch buffer inside `conv2d()` but explicitly
flagged that `Executor::run()`'s op loop still heap-allocates a fresh
`Tensor` (and therefore a fresh `std::vector<float>`) for every node's
output, on every `run()` call. `leaf_bench` calls `run()` once per warmup
and measured iteration on the same thread, so this is repeated,
same-shape allocation churn on every forward pass -- the same class of
problem ADR-002 solved for the im2col buffer, one layer up.

### Options considered

Same three options ADR-002 weighed, revisited for this case:

1. **Executor-owned buffer pool as a class member.** Would require
   `Executor::run()` to stop being effectively stateless per-call, or
   threading a workspace object through every op function
   (`execute_conv`, `execute_maxpool`, etc.). Larger API surface change
   than the problem warrants right now.
2. **Explicit workspace object passed into each `execute_*` helper.**
   Cleaner ownership, but touches every op function's signature and every
   call site in `run()`'s dispatch loop.
3. **`thread_local` pool of retired buffers, acquired/released inside
   `run()`'s existing liveness tracking.** No API changes. `run()`
   already computes exactly when an intermediate's last consumer has run
   (the `remaining_uses` map used to decide when to erase from
   `intermediates`) -- that's already liveness analysis, just not yet
   feeding a buffer pool. Extending it to route retired buffers into a
   pool instead of discarding them, and to pull from that pool instead of
   allocating fresh, needed no new bookkeeping.

### Decision

Chose option 3, for the same reasons as ADR-002: single-file change
(`executor.cpp`), `Executor::run()`'s public signature and `const`-ness
untouched, existing call sites (all three `main_*.cpp` binaries) unaffected.

Implementation:
- A `thread_local std::vector<std::vector<float>> g_buffer_pool` holds
  retired activation buffers.
- `acquire_buffer(needed)` does a best-fit scan: reuse the smallest freed
  buffer with `capacity() >= needed` if one exists (avoids discarding a
  right-sized buffer in favor of an oversized one); otherwise grow the
  largest available buffer; otherwise allocate fresh. Grow-only, same
  spirit as ADR-002's `col` buffer.
- When the existing `remaining_uses` liveness check retires an
  intermediate, its buffer is moved into the pool via `release_buffer()`
  instead of being destroyed with the erased map entry.
- Every op that acquires a buffer this way either has its full range
  overwritten by the kernel it calls (`Conv` -- `conv2d()` already
  `memset`s internally, so the prior outer zero-init was redundant
  regardless of this change), has every element assigned exactly once by
  a loop (`MaxPool`, `GlobalAveragePool`, `Add`), copies the full input
  range in before modifying in place (`Relu`, `Flatten`,
  `BatchNormalization`), or is explicitly zero-filled before an
  accumulating write (`Gemm`'s `beta * C` term, which already needed
  this before the pool existed). No op reads a byte of a reused buffer
  before writing it, so stale content from a previous node or a previous
  `run()` call cannot leak through.

### Verification

The existing `test_executor` (Conv/Relu/GlobalAveragePool/Flatten/Gemm)
doesn't exercise `Add`, `MaxPool`, or `BatchNormalization` -- three of
the seven paths this change touches -- so it alone wasn't enough
evidence. Added `test_executor_buffer_pool`, a second hand-built `.leaf`
artifact covering exactly those three ops, with the executor's `run()`
called twice on the same graph and asserting bit-for-bit identical
output both times (catches cross-call pool corruption, not just
cross-node corruption within one call). Also ran the full suite under
`-fsanitize=address,undefined` during development: zero ASan errors
(no overflow, no use-after-free, no leaks from the pool logic).

UBSan separately flagged misaligned `float` loads when reading
initializer data straight out of the `.leaf` file's byte buffer --
reproduced with the new test's initializer layout but *not* with the
original `test_executor`'s smaller one, and unrelated to this change
(it's a property of `graph_parser.cpp`'s raw-pointer reads and whatever
byte alignment `tools/graph_opt/export_binary.py` does or doesn't
guarantee for a given initializer's offset). Flagged as a follow-up
below rather than fixed here, since it's outside this ADR's scope and
touches the export path, not the runtime.

| Check | Result |
|---|---|
| `ctest` (4/4, incl. new test) | Passed |
| `test_executor_buffer_pool`, run 1 and run 2 | Bit-identical, matches hand-computed reference |
| ASan (`-fsanitize=address`) | Zero errors |

Parity and latency measured on the dev machine after applying this
patch: `verify_cpp_runtime.py` confirms `3.93e-06` max abs diff, unchanged
from M5a/ADR-002. For latency, the ADR-002 session's `106.812 ms` figure
is not a fair comparison point -- re-measuring today (different session,
different host load) put the *unpatched* build at `129.620 ms` mean, well
above that recorded baseline, indicating this machine was under more load
today than during the ADR-002 session. To isolate the actual effect of
this patch from that session-to-session drift, unpatched and patched were
rebuilt back-to-back in the same session via `git stash`/`git stash pop`
(so only this patch's four files differed between the two runs, host load
held constant):

| Build | Mean | Min | p50 | p95 |
|---|---:|---:|---:|---:|
| Unpatched (this session) | 129.620 ms | 120.647 ms | 127.739 ms | 139.167 ms |
| Patched (this session) | 126.314 ms | 120.715 ms | 124.268 ms | 137.941 ms |

~2.5% mean latency reduction. Modest, not dramatic -- `min` latency is
essentially unchanged (120.647 vs 120.715 ms, within noise), consistent
with the win coming from reduced allocation churn across the ~50-70 node
graph's mean/p50 rather than the best-case run, which is dominated by
compute rather than allocation overhead either way. Logged in
`benchmark/results/wsl_dev_machine.csv`. **Open question, not yet
investigated:** `acquire_buffer()`'s O(n) best-fit linear scan across the
pool may itself be leaving some of the win on the table for a graph this
size -- worth comparing against a plain O(1) LIFO stack before concluding
this is the ceiling for this approach.

### Open follow-ups

- **Misaligned initializer reads (new finding, see Verification).**
  `graph_parser.cpp` and the kernels that read initializer data
  (`gemm.cpp`, `conv2d.cpp`) dereference `const float*` pointers straight
  into the loaded file buffer without checking alignment. x86 tolerates
  this silently, so it hasn't caused an observed correctness problem
  (M1b's `3.93e-06` parity holds), but it's UB, and `-mavx2 -mfma`
  intrinsics reading misaligned data is exactly the kind of thing that
  can regress silently on a different compiler or CPU. Worth checking
  whether `export_binary.py` already pads/aligns initializer offsets
  within the data block, and if not, deciding whether to align on export
  or to make the runtime's reads alignment-safe (e.g. `memcpy` into a
  local before use). Separate from this ADR; not fixed here.
- **Multi-threading interaction (carried over from ADR-002, still open).**
  Both the `conv2d()` scratch buffer and this new pool are
  `thread_local`. Fine for today's single-threaded executor; each worker
  thread in M5b's planned output-channel-tile parallelization will get
  its own independent pool copy -- a memory-footprint consideration, not
  a correctness one, but worth sizing before scaling thread count.
- **This is a runtime-level cache, not the M2b memory plan itself.** The
  README's M2b goal is a liveness-based memory plan computed once
  up-front (ahead-of-time buffer assignment, ideally exported into the
  `.leaf` artifact or planned at load time), which could do better than
  this pool's per-run best-fit scan -- e.g. knowing the full node
  sequence in advance to assign a fixed small set of buffers with zero
  runtime search cost. This ADR closes the gap for now with a low-risk,
  no-API-change patch, same relationship option 3 had to options 1/2 in
  ADR-002.

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
