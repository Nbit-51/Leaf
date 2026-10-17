# Resume shared accumulation work

Base release: `107cb98` (PR #8 merged; temporary release branch deleted).
Current experiment is isolated in
`benchmark/experiments/linear-accumulation/`. The user requested small checks
and a durable checkpoint, deferring full suites and larger benchmarks because
of remaining usage. No agents, timers or long-running tests remain active.
Preserve the unrelated root image.

The build and thirteen small GEMM/RMS cases pass. The new six-row K=128
partial reduction plus FP64 outer sums and pure-FP32 precise RMSNorm passes
both retained failing 128-token blocks: Smol 154→0 violations, Qwen 451→0.
The unchanged tolerance is `2e-4 + 2e-4*abs(reference)`. Both blocks retain
100% next-token agreement. Prior norm + two-bank GEMM left two Smol failures.

One short pinned micro probe does not establish a speed win: pooled candidate
medians are roughly 4–5% higher and samples contain spikes. This is an accuracy
candidate, not a latency release. Its twelve inner FMA accumulators retain
six-row reuse; inspect the repeated partial-call/buffer-conversion overhead
before another long acceptance run. Avoid a blind tile-parameter sweep.

Local successful build: `build/linear-accumulation-20261017-v2/`:

- `build.json`: exact commands, source hashes, base commit and binary hashes.
- `leaf_decoder.exe`, `probe.exe`: already compiled; no rebuild needed.
- `probe-correctness.json`: thirteen passing shape cases.
- `probe-timing.json` and `.stdout.txt`: short diagnostic samples, affinity `[2]`.
- `saved-blocks.json`: complete, zero violations for both saved blocks.
- `{smol,qwen}-block1-logits.npy`: retain locally, never commit weights/logits.

Source, portable JSON evidence and exact queued commands are saved in the
experiment README. The first failed build and timing-wrapper error are
documented there; do not repeat them. Larger eight-block quality, complete
cache/generation/thread regressions, whole-model timing and installed-package
validation have **not** run for this candidate. Release defaults are unchanged.

Next: inspect overhead, run the queued larger quality checks when budget
permits, and only then consider a measured release through a PR. Do not rerun
PR #8's completed decode acceptance or change strict quality limits.
