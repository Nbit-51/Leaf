"""Check compiler-free wheel runtimes, or automatically build source runtimes."""
import json
import os
from pathlib import Path
import subprocess


def check(runtime: Path, name: str) -> dict:
    try:
        result = subprocess.run([str(runtime), "--runtime-info"], capture_output=True,
                                text=True, timeout=15)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Runtime startup timed out: {runtime}") from error
    try:
        info = json.loads(result.stdout)
        expected = ([2, 3, 4, 5], [2, 3, 4, 5, 6]) if name == "leaf_decoder" else ([1],)
        if result.returncode or info.get("runtime") != name or info.get("artifact_versions") not in expected:
            raise ValueError("Unexpected runtime identity or artifact support")
    except (ValueError, AttributeError) as error:
        raise RuntimeError(f"Runtime check failed for {runtime}. Reinstall a current platform wheel "
                           "or remove an outdated LEAF_DECODER_BIN/LEAF_EMBED_BIN override.") from error
    return dict(path=str(runtime.resolve()), **info)


def check_decoder_options(runtime: Path, *, source_bf16=False, source_bf16_head=False,
                          fp32_math='fast', worker_cpus=None):
    if not (source_bf16 or source_bf16_head or worker_cpus is not None or fp32_math == 'accurate'):
        return
    info = check(runtime, 'leaf_decoder')
    modes = info.get('fp32_math_modes')
    unsupported = None
    if worker_cpus is not None and info.get('worker_affinity_supported') is not True:
        unsupported = '--worker-cpus'
    elif source_bf16 and (info.get('source_bf16_matrices_supported') is not True or
                          6 not in info.get('artifact_versions', [])):
        unsupported = '--source-bf16'
    elif fp32_math == 'accurate' and (not isinstance(modes, list) or 'accurate' not in modes):
        unsupported = '--fp32-math accurate'
    if unsupported:
        raise RuntimeError(f'The selected native runtime does not support {unsupported}; '
                           'reinstall a current wheel or remove an outdated LEAF_DECODER_BIN override')


def doctor(task: str):
    from leaf.cli import cache_root, executable
    from leaf.embedding import executable as embedding_executable
    cache = cache_root()
    checks = {}
    for kind, resolver, name, variable in (
            ("generation", executable, "leaf_decoder", "LEAF_DECODER_BIN"),
            ("embedding", embedding_executable, "leaf_embed", "LEAF_EMBED_BIN")):
        if task not in (kind, "all"):
            continue
        runtime = resolver(cache)
        info = check(runtime, name)
        bundled = Path(__file__).parent / "bin" / runtime.name
        info["selection"] = ("override" if os.environ.get(variable) else
                             "bundled" if runtime.resolve() == bundled.resolve() else "source-cache")
        checks[kind] = info
    print(json.dumps(dict(ready=True, cache=str(cache.resolve()), runtimes=checks,
                         scope="Runtime startup only; no model downloaded or evaluated."), indent=2))
