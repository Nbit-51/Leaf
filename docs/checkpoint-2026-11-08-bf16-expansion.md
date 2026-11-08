# November 8: BF16 prefill conversion

Stay on `codex/mlp-schedule`; do not merge or delete the branch. Preserve the
unrelated embedding document and October 8 image. AC and effective
`better-battery` mode are verified. No system power setting changes.

## Resumed evidence

Commit 217e244 passed all portable CI jobs (run 40998930502). Read the October 30
checkpoint before repeating anything. Source BF16 artifacts, installed wheel
and trained quality were already qualified; no model download is needed.

Local root: `build/source-bf16-timing-20261108/`.

The final six-row runtime84ef3bff71eeb111 was measured against e08b482db58ebd47
with the older head-only artifacts. Accurate/accurate, one worker on CPU2,
63-token prefill plus one fixed cached decode. Two balanced pairs, seven samples
and three warmups; profile/counters disabled. Every pass and pooled phase is
stable; source identity, AC and effective-mode guards pass.

| Model | Head-only prefill | Matrix prefill | Head-only decode | Matrix decode |
|---|---:|---:|---:|---:|
| Smol |506.05880ms|551.27635ms|76.39815ms|51.08885ms|
| Qwen |568.05965ms|597.89170ms|96.05570ms|65.27265ms|

These are short diagnostics, not release acceptance. Matrix storage remains a
latency tradeoff: decode~32-33% lower, prefill~5-9% higher.

Matched 96-row versus 6-row v6 CPU-work diagnostics use two balanced pairs,
five samples and three warmups, with counters enabled. Large scratch records
38,720-38,880 Smol and74,568 Qwen process page faults per 5-sample pass. Every
six-row measured phase has zero faults. Process faults include allocation/soft
faults; this does not prove disk paging. CPU accounting also supports real work
cost, with coarse timing granularity; do not inferGHz or exact off-CPU time.

## Candidate

The ordinary conversion loop compiles to SSE2 unpack/shift/store instructions.
An explicit AVX2 path uses eight-value integer widening/shift and a 256-bit
store. New 4-argument `expand_source_bf16(...,use_avx2)` guards AVX2/FMA, retains the
original 3-argument portable function and handles zero-count/null input. Decoder passes
its existing `avx` policy; only packed BF16 projections change. Decode GEMV,
FP32 and quantized arithmetic are unchanged. Six-row buffers are retained.

Nativetest passes 4,292,253 value/guard checks and 6,609 calls, including all
65,536 BF16 patterns on default/false/true paths, offsets/tails/guards and source
immutability. Full Python suite: 1,513 passed,two optional legacy skips.

Isolatedbench has 18cases with production/unaligned layouts and keeps every
sample. The prototype sources and original baseline header are hash-preserved.
`expansion-dispatched.json` measures the actual new API. All four production
six-row cases are stable: paired ratios~0.599,0.595,0.265,0.997 at 5,376/5,760/
15,360/29,184values. The neutral longest case rules out a universal expansion
speed claim. Reused-buffer operator timings are not whole-model latency.

`candidate.txt` identifies5647d6c4f8000528. Full trained audit is running under
`trained/`; installed wheel/offline doctor and generation checks pass, including
an additional 17-token prefill to exercise packed conversion. Bind its PE code/
data sections after the trained report completes. Do not time while correctness
or builds run.

`candidate-plan.json` declares six balanced pairs, 31samples and 10warmups on each
model, accurate/accurate, CPU2, counters/profileoff. Compare against the previous
six-row implementation with the same v6 tensor values. Prefill<=0.98,decode<=1.02,
all individual/pooled p90/p10<=1.25; retain all samples. This differs from the
head-only storage tradeoff above. No new competitor comparison is planned.


## Final checkpoint: stop after push

The user requested committing/pushing and stopping as soon as Qwen finishes.
Both full runs have finished. No further tests or experiments should be started
in this session. No benchmark/build/qualification workloads remain active.

Selected runtime: `candidate.txt`, cache `5647d6c4f8000528/leaf_decoder.exe`.
`trained/quality.json` is complete/passing:32 full 128-token blocks, 411,828,224
bit-identical logits,16 continuations and 16 controls. `all-units.xml`: 1,513 passes,
two optional legacy skips. `native-test.log`: 4,292,253 value/guard checks and 6,609 calls.
Installed ordinary and 17-token packed-prefill checks pass; installed .text/
.rdata/.data match the fully trained-qualified candidate.

`smol-acceptance/` and `qwen-acceptance/` each contain 12 passes, 186 samples/runtime/
phase. Every individual/pooled stability check and power audit passes. No
profiling/counters during these timings. `incremental-gate.json` applies the
predeclared gate without modifying raw records:

- Smol 521.22455->513.15220ms prefill: pooled ratio 0.984513,paired 0.985271.
  Decode 49.04615->48.82465ms; non-regression passes.
- Qwen 590.55420->587.68990ms prefill: pooled ratio 0.995150,paired 0.991301.
  Decode 63.91170->63.88160ms; non-regression passes.
- Both fail only the 2% prefill improvement requirement. Do not call this a
  passed model-speed gate. Preserve the implementation with only its verified
  bit-preserving operator scope and observed smaller stable model differences.
  Do not re-run this unchanged hoping for a threshold pass.

Next session: inspect the pushed revision's CI, then select a new workload or
measured bottleneck. A useful product-level check is native multi-token greedy
generation, which measures the prefill/decode tradeoff over real KV growth; it
has not been implemented/measured here. Keep warmup and load/tokenizer/streaming
costs explicit, verify generated IDs and stop-token handling, and avoid claiming
complete-command latency from summed forward timers. Another available follow-up
is matched scaling on assigned physical P-cores with this smaller storage.
Neither follow-up authorizes a new model or cross-runtime speed claim now.

Do not repeat unchanged full correctness, rebuild the same source, reopen the
alignment allocator idea, or merge/delete this branch. Preserve the unrelated
embedding document and image. The user's efficiency mode remains unchanged.


## Post-run power clarification: supersedes Qwen timing eligibility

After the push, the owner confirmed unplugging or mains power loss during
Qwen. The exact interruption time is unknown. Its final run covered
07:59:03-08:04:35 IST. All saved boundary AC checks and the native per-iteration
guard reported success; the effective-mode guard passed. A read-only lookup
found no matching Windows Kernel-Power source/sleep events around the run.
These observations cannot override the reported interruption.

Treat the entire final Qwen latency/non-regression result as provisional.
Preserve raw samples, stability calculations and the original archive unchanged;
do not select supposedly unaffected passes. The public `power-followup.json`
and summary annotation record this correction. Correctness, installed-wheel
and operator-bit tests remain valid. Smol's timing status is unchanged.

No new benchmark or correctness test was started. Stop after this documentation
correction is pushed. At the next user-authorized session, a fresh Qwen timing
confirmation under verified stable AC is justified by this new environmental
information. This supersedes the earlier instruction against repeating an
unchanged run merely to seek a threshold pass.
