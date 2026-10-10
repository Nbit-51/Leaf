"""Build platform wheels with a portable, self-contained native decoder."""
from pathlib import Path
import os
import shutil
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
try:
    # Canonical location since setuptools 70.1; keep the declared 69.x floor.
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


class NativeBuild(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise RuntimeError("Building a Leaf wheel requires a C++17 GCC/Clang compiler")
        output = Path(self.build_lib) / "leaf/bin" / ("leaf_decoder.exe" if os.name == "nt" else "leaf_decoder")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-pthread", "-I", str(root / "engine/include")]
        command.extend(str(root / "engine" / filename) for filename in
                       ("src/decoder.cpp", "src/main_decoder.cpp", "src/kv_cache.cpp", "src/kernels/transformer.cpp"))
        command.extend(["-o", str(output)])
        if os.name == "nt":
            command.extend(["-static", "-static-libgcc", "-static-libstdc++", "-lpsapi"])
        subprocess.run(command, check=True)
        embed_output = output.with_name("leaf_embed.exe" if os.name == "nt" else "leaf_embed")
        embed_command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-ffp-contract=off", "-pthread",
                         "-I", str(root / "engine/include")]
        embed_command.extend(str(root / "engine" / name) for name in
                             ("src/embedding.cpp", "src/main_embed.cpp", "src/kernels/transformer.cpp"))
        embed_command.extend(["-o", str(embed_output)])
        if os.name == "nt":
            embed_command.extend(["-static", "-static-libgcc", "-static-libstdc++"])
        subprocess.run(embed_command, check=True)


class PlatformDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(cmdclass={"build_py": NativeBuild, "bdist_wheel": PlatformWheel}, distclass=PlatformDistribution)
