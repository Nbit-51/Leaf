# Explicit decoder worker placement

`leaf run --worker-cpus` assigns one distinct logical CPU to each native inference
thread on Windows and Linux. The normal command uses the operating system's
scheduler. Placement is an advanced, optional setting; it is not an automatic
model optimization or a guarantee of faster inference.

For example, on a machine where logical CPUs 0 and 2 are allowed:

```sh
leaf run EleutherAI/pythia-70m --bits 32 --threads 2 --worker-cpus 0 2 --prompt "Local inference is" --max-tokens 32
```

Choose CPU numbers for your machine. The first CPU belongs to the calling thread;
the remaining CPUs belong to the pool's owned workers. The list must contain
exactly `--threads` distinct, nonnegative indices within the allowed affinity.
`--cpu` and `--worker-cpus` cannot be combined. Explicit placement currently
supports processor group zero on Windows and the fixed `cpu_set_t` range on Linux.
Unsupported requests fail instead of silently running unbound.

The CLI validates the list and native capability before downloading large weight
files. Existing automatic lower-precision speed profiles do not describe this
placement policy, so `--bits auto` falls back to FP32 with explicit worker CPUs.
Explicit precision choices keep their existing semantics. Metrics record
`worker_cpus` in thread order.

## Native behavior

The pool verifies worker binding at construction. Workers keep their assigned
CPUs until they exit. The calling thread is bound only during a forward/pool
operation and its previous affinity is restored afterward, including exceptional
exits. Nested operations already bound to the requested CPU reuse that mask.
Process affinity and system power settings are untouched.

Worker task exceptions are collected and rethrown after the pool has finished
the current operation. The pool can be reused after a failed task. A failed
startup joins already-created workers before returning an error.

The C++ decoder also accepts an optional CPU list after `activation_bits`:

```cpp
leaf::runtime::Decoder decoder("model.leaf", 2, false, 32, {0, 2});
auto logits = decoder.forward({1, 2, 3});
```

Allowed CPUs must remain available during execution. Separate physical cores
are useful for controlled scaling tests; SMT siblings, efficiency cores and
NUMA placement can change results. Select them from actual machine topology.

## Validation scope

The native pool tests cover exactly-once ranges, all-worker participation,
binding/restoration, nested scopes, invalid lists, task exceptions and reuse.
CLI/native protocol tests cover early rejection, order preservation, capability
checks and placement metadata. A compiler-free installed-wheel run exercises
explicit placement with a freshly prepared offline fixture.

The trained checks rebind the previously qualified accurate K=256 FP32 build on
all eight held-out blocks and six continuations per model for SmolLM2-360M and
Qwen2.5-0.5B. Scalar, chunked-cache, one/two/four-thread and ordinary-default
regression controls are separate. The accurate accumulation policy remains
experimental; these checks do not resolve the ordinary default's strict logit
tolerance misses or create a new cross-runtime quality qualification.

The [October 20 traces](checkpoint-2026-10-20-prefill-jobs.md) caught an intermittent
parked/idle allowed core. Affinity is a testable response to that finding, rather
than a proven explanation for every historical timing spike. Full new timing
criteria and results belong in the [current checkpoint](checkpoint-2026-10-22-worker-placement.md).

The implementation follows the Windows
[thread-affinity contract](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask)
and Linux [pthread affinity API](https://man7.org/linux/man-pages/man3/pthread_setaffinity_np.3.html).
