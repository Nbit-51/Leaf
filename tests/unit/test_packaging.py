"""Fast packaging contracts; no compiler or model execution is required."""
from __future__ import annotations

import importlib.util
import builtins
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from leaf import cli


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup_module(monkeypatch):
    pytest.importorskip("wheel")
    import setuptools
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    specification = importlib.util.spec_from_file_location("leaf_package_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target,filename", [("nt", "leaf_decoder.exe"), ("posix", "leaf_decoder")])
def test_wheel_build_bundles_portable_native_runtime(setup_module, monkeypatch, tmp_path, target, filename):
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    monkeypatch.setattr(build_py, "run", lambda self: None)
    setup_module.os = SimpleNamespace(name=target, environ={"CXX": "test-cxx"})
    captured = []
    monkeypatch.setattr(setup_module.subprocess, "run", lambda command, **kwargs: captured.append((command, kwargs)))
    build = setup_module.NativeBuild(Distribution())
    build.build_lib = str(tmp_path / "package")
    build.run()
    command, options = captured[0]
    output = tmp_path / "package" / "leaf" / "bin" / filename
    assert command[0] == "test-cxx"
    assert command[command.index("-o") + 1] == str(output)
    assert output.parent.is_dir()
    assert options == {"check": True}
    assert "-std=c++17" in command
    assert command[command.index("-I") + 1] == str(ROOT / "engine/include")
    assert not any(flag.startswith(("-march", "-mavx", "-mfma")) for flag in command)
    sources = {Path(value).name for value in command if value.endswith(".cpp")}
    assert sources == {"decoder.cpp", "main_decoder.cpp", "kv_cache.cpp", "transformer.cpp"}
    if target == "nt":
        assert {"-static", "-static-libgcc", "-static-libstdc++", "-lpsapi"}.issubset(command)
    else:
        assert "-lpsapi" not in command


def test_token_panel_header_is_included_in_packaged_native_sources():
    # TOML parsing is stdlib on the hosted Python 3.12 validation matrix.
    # Python 3.10 installations can also run this check when tomli is present.
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    headers = config["tool"]["setuptools"]["data-files"]["share/leaf/engine/include/leaf/kernels"]
    source = "engine/include/leaf/kernels/token_panel.h"
    assert source in headers
    assert (ROOT / source).is_file()
    assert "engine/include/leaf/kernels/gelu.h" in headers
    assert "engine/include/leaf/kernels/attention_vector.h" in headers


def test_source_wheel_build_without_compiler_fails_explicitly(setup_module, monkeypatch, tmp_path):
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    monkeypatch.setattr(build_py, "run", lambda self: None)
    setup_module.os = SimpleNamespace(name="posix", environ={})
    monkeypatch.setattr(setup_module.shutil, "which", lambda command: None)
    build = setup_module.NativeBuild(Distribution())
    build.build_lib = str(tmp_path / "package")
    with pytest.raises(RuntimeError, match="C\\+\\+17 GCC/Clang"):
        build.run()


def test_wheel_tag_does_not_depend_on_python_native_abi(setup_module, monkeypatch):
    from setuptools import Distribution
    bdist_wheel = setup_module.PlatformWheel.__bases__[0]
    monkeypatch.setattr(bdist_wheel, "get_tag", lambda self: ("cp314", "cp314", "win_amd64"))
    command = setup_module.PlatformWheel(Distribution())
    assert command.get_tag() == ("py3", "none", "win_amd64")
    assert setup_module.PlatformDistribution().has_ext_modules()


def test_setup_prefers_canonical_setuptools_wheel_command(setup_module):
    pytest.importorskip("setuptools.command.bdist_wheel")
    assert setup_module.PlatformWheel.__bases__[0].__module__ == "setuptools.command.bdist_wheel"


def test_setup_supports_legacy_setuptools_command_location(monkeypatch):
    pytest.importorskip("wheel.bdist_wheel")
    import setuptools
    from wheel.bdist_wheel import bdist_wheel
    original = builtins.__import__

    def without_integrated_command(name, *args, **kwargs):
        if name == "setuptools.command.bdist_wheel":
            raise ImportError("Simulate setuptools 69.x")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setattr(builtins, "__import__", without_integrated_command)
    specification = importlib.util.spec_from_file_location("leaf_legacy_package_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert module.PlatformWheel.__bases__[0] is bdist_wheel


def test_installed_bundled_runtime_does_not_require_a_compiler(monkeypatch, tmp_path):
    package = tmp_path / "installed" / "leaf"
    package.mkdir(parents=True)
    binary = package / "bin" / ("leaf_decoder.exe" if cli.os.name == "nt" else "leaf_decoder")
    binary.parent.mkdir()
    binary.write_bytes(b"bundled runtime")
    monkeypatch.delenv("LEAF_DECODER_BIN", raising=False)
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    monkeypatch.setattr(cli, "source_engine", lambda: pytest.fail("Installed wheel attempted source compilation"))
    monkeypatch.setattr(cli.shutil, "which", lambda command: pytest.fail("Installed wheel searched for a compiler"))
    assert cli.executable(tmp_path / "cache") == binary
    assert not (tmp_path / "cache").exists()


def test_core_imports_do_not_load_validation_or_heavy_model_libraries():
    program = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(ROOT)!r})",
        "import leaf.cli, tools.export_decoder, tools.decoder_plan, tools.verify_decoder_portability",
        "for module in ('torch', 'transformers', 'onnx', 'pyarrow', 'safetensors'):",
        "    assert module not in sys.modules, module",
    ])
    completed = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("source,expected", [
    (r"C:\models\test.leaf", "/mnt/c/models/test.leaf"),
    (r"D:\path with spaces\tokens.bin", "/mnt/d/path with spaces/tokens.bin"),
    (r"c:\Users\local\metrics.json", "/mnt/c/Users/local/metrics.json"),
])
def test_wsl_fixture_paths_are_mapped_without_shell_interpolation(source, expected):
    from tools.verify_decoder_portability import wsl_path
    assert wsl_path(Path(source)) == expected


@pytest.mark.parametrize("source", ["tokens.bin", r"C:tokens.bin", r"\\server\share\tokens.bin", "/tmp/tokens.bin"])
def test_wsl_fixture_paths_reject_non_drive_absolute_paths(source):
    from tools.verify_decoder_portability import wsl_path
    with pytest.raises(ValueError, match="absolute Windows drive path"):
        wsl_path(Path(source))


def test_portable_token_request_is_explicitly_little_endian(tmp_path):
    from tools.verify_decoder_portability import write_request
    request = tmp_path / "tokens.bin"
    write_request(request, [[1, 256], [65536]])
    assert request.read_bytes() == bytes.fromhex("020000000200000001000000000100000100000000000100")
