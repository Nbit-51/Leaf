# October 20 prefill experiments

These are unqualified kernel experiments and scoped Windows worker-placement diagnostics.
No new default kernel or competitor speed claim is released. See the
[checkpoint](../../../../../docs/checkpoint-2026-10-20-prefill-jobs.md) for decisions
and the [runner guide](../../README.md) for reproduction.

- `full-model-v2/` and `full-model-final/`: complete quality and **failed** acceptance.
- `process-diagnostic/`: four unchanged-candidate passes that reproduce two timing bands.
- `placement-v2/`: owned-worker off/on/on/off controls for Smol and Qwen.
- `placement-summary.json`: derived per-pass latency, parked fraction and busy counters.
- `placement-diagnostic/`: incomplete harness attempt before timing, retained as incomplete.
- `source-snapshot/`: exact candidate source bytes before restoring the default engine.
- `reconstruction/`: private rebuild, source equivalence and full-logit smoke proof.
- Unit, architecture, ordinary regression, layout and installed-wheel records are
  non-timing checks; none overrides a failed performance gate.

`evidence.zip` contains the folders/files listed above. `manifest.json` hashes
both the archive and every unmodified entry. Raw files were byte-verified after
compression. The public `placement-summary.json` is also kept outside the archive.

Extract from the repository root when inspecting the full samples:

```powershell
Expand-Archive benchmark/experiments/mlp-schedule/results/20261020/evidence.zip -DestinationPath build/mlp-evidence-20261020
```

Nested attributes preserve artifact bytes.
Build manifests record local source/binary hashes; source/runner snapshots retain
superseded versions. Baseline trained quality is linked through the source-BF16
checkpoint. Large binaries, artifacts and logits remain local in ignored `build/`.
