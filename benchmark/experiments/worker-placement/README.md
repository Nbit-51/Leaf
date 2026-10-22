# Worker placement and synchronization

The decoder accepts explicit Windows/Linux worker CPUs. Its assigned
multi-thread pool uses bounded polling before sleeping; normal placement and
one-thread execution remain blocking. No model IDs select this policy.

- `validate.py`: full trained exact rebinding and separate, predeclared placement,
  scheduling, synchronization and ordinary-runtime timing gates. No filtering.
- `check_ordinary.py`: prefix/cache/generation regression across Smol/Qwen FP32,
  GPT-2 FP32 and TinyLlama W8A8; not strict PyTorch qualification.
- `build_jobs.py`: private tile-aligned job-size experiment, not enabled.
- `build_spin.py`: reconstruct the bounded-spin experiment from historical
  placement source `31fff42`, without editing the current engine.
- `diagnose.py`: private A/A process/barrier/dispatch diagnostics. Instrumented
  timing cannot promote a speed claim. Its CPU numbers describe the tested host.

The accurate K=256 arithmetic used for trained fidelity and scheduling comparisons
remains separate from ordinary released FP32 math. Required local model artifacts
and prior quality records live under ignored `build/`; the scripts reject stale
identities instead of substituting another model or relaxing tolerances.

The six-pair assigned-worker synchronization comparison passes every spread
check on both models and reduces decode medians by 5.6–5.9% on this Windows host.
The job-size comparison and final ordinary one-thread guard have failed joint
gates; they are retained as failures. The latter's intermittent speed bands
remain under investigation. See the [checkpoint](../../../docs/checkpoint-2026-10-22-worker-placement.md)
and [user-facing policy](../../../docs/worker-placement.md) for boundaries and
local resume paths.
