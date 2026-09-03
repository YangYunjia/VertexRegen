"""One-shot dev setup for the optShapeGen VertexRegen fork.

Compiles the patched ``vertexregen_tokenizer`` C++ extension, editable-installs
both packages, and writes a ``.pth`` file so ``import vertexregen`` /
``import vertexregen_tokenizer`` / ``import optshapegen`` work without
``export PYTHONPATH``.

Prerequisites (conda ``armesh`` env):
  gxx_linux-64, pybind11, CGAL headers, gmp, mpfr, eigen

Usage:
    conda activate armesh
    cd experiments/baselines/vertexregen
    python setup.py setup_dev          # full setup (recommended)
    python setup.py build_tokenizer    # rebuild .so only
    python setup.py setup_dev --verify # setup + smoke test

After C++ changes, rerun ``python setup.py build_tokenizer`` (~40s).
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

from setuptools import Command, find_packages, setup

ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT / "tokenizer"
TOKENIZER_PKG = TOKENIZER_DIR / "vertexregen_tokenizer"
OPTSHAPEGEN_ROOT = ROOT.parents[2]  # experiments/baselines/vertexregen -> optShapeGen
PTH_NAME = "optshapegen_vertexregen.pth"
_SETUP_FLAG = "_VR_SETUP_DEV_RUNNING"


def _conda_prefix() -> Path | None:
    prefix = os.environ.get("CONDA_PREFIX")
    return Path(prefix) if prefix else None


def _find_cxx() -> str:
    if cxx := os.environ.get("CXX"):
        return cxx
    prefix = _conda_prefix()
    if prefix:
        candidate = prefix / "bin" / "x86_64-conda-linux-gnu-g++"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("g++") or "g++"


def build_tokenizer_so(verbose: bool = True) -> Path:
    """Compile patched pybind extension; return path to the .so."""
    import sysconfig

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = TOKENIZER_DIR / f"_vertexregen_tokenizer_pybind{ext_suffix}"

    prefix = _conda_prefix()
    include_flags = ["-I/usr/include/eigen3"]
    lib_flags = ["-lgmp", "-lgmpxx", "-lmpfr"]
    if prefix:
        include_flags[:0] = [
            f"-I{prefix}/include",
            f"-I{prefix}/include/eigen3",
        ]
        lib_flags.extend([f"-L{prefix}/lib", f"-Wl,-rpath,{prefix}/lib"])

    pybind_includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        text=True,
    ).split()

    cmd = [
        _find_cxx(),
        "-O3",
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-fvisibility=hidden",
        *pybind_includes,
        *include_flags,
        "src/pybind.cpp",
        "src/simplify.cpp",
        "src/mesh.cpp",
        "src/visitor.cpp",
        *lib_flags,
        "-o",
        str(out),
    ]

    if verbose:
        print(f"[build_tokenizer] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=TOKENIZER_DIR, check=True)

    dest = TOKENIZER_PKG / out.name
    shutil.copy2(out, dest)
    if verbose:
        print(f"[build_tokenizer] wrote {dest}", flush=True)
    return dest


def _pip_install_editable(path: Path, *, no_deps: bool = True) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(path)]
    if no_deps:
        cmd.append("--no-deps")
    env = os.environ.copy()
    env[_SETUP_FLAG] = "1"
    subprocess.run(cmd, check=True, env=env)


def write_path_pth(*, verbose: bool = True) -> Path:
    """Register optShapeGen + vertexregen roots on sys.path (replaces PYTHONPATH)."""
    paths = [str(OPTSHAPEGEN_ROOT), str(ROOT)]
    body = "\n".join(paths) + "\n"

    targets: list[Path] = []
    try:
        targets.extend(Path(p) for p in site.getsitepackages())
    except AttributeError:
        pass
    user = site.getusersitepackages()
    if user:
        targets.append(Path(user))

    written: Path | None = None
    for base in targets:
        if not base.is_dir():
            continue
        pth = base / PTH_NAME
        pth.write_text(body)
        written = pth
        break

    if written is None:
        raise RuntimeError("could not find a site-packages directory for .pth")

    if verbose:
        print(f"[setup_dev] wrote {written}:", flush=True)
        for line in paths:
            print(f"  {line}", flush=True)
    return written


def verify_install() -> None:
    import inspect

    from vertexregen_tokenizer import _vertexregen_tokenizer_pybind as VR

    sig = inspect.signature(VR.edge_collapse_with_record)
    if "pinned_vertices" not in sig.parameters:
        raise RuntimeError("tokenizer .so missing pinned_vertices — rebuild?")

    import vertexregen  # noqa: F401
    import optshapegen  # noqa: F401

    print("[verify] pinned_vertices OK, vertexregen + optshapegen import OK", flush=True)


def run_setup_dev(*, verify: bool = False) -> None:
    if os.environ.get(_SETUP_FLAG):
        return
    os.environ[_SETUP_FLAG] = "1"
    try:
        build_tokenizer_so()
        _pip_install_editable(TOKENIZER_DIR)
        _pip_install_editable(ROOT)
        write_path_pth()
        if verify:
            verify_install()
        print(
            "\n[setup_dev] done. Open a new terminal (or restart Python) and run "
            "without PYTHONPATH:\n"
            "  python experiments/bracket/train/preprocess_vertexregen_seeded.py ...\n"
            "  cd experiments/baselines/vertexregen && python train.py ...",
            flush=True,
        )
    finally:
        os.environ.pop(_SETUP_FLAG, None)


class BuildTokenizerCommand(Command):
    description = "compile patched vertexregen_tokenizer C++ extension"
    user_options: list[tuple[str, str | None, str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        build_tokenizer_so()


class SetupDevCommand(Command):
    description = "build tokenizer, editable-install packages, write .pth paths"
    user_options = [
        ("verify", None, "run import smoke test after setup"),
    ]

    def initialize_options(self) -> None:
        self.verify = False

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        run_setup_dev(verify=bool(self.verify))


setup(
    name="vertexregen-optshapegen",
    version="0.0.1",
    packages=find_packages(include=["vertexregen*"]),
    cmdclass={
        "build_tokenizer": BuildTokenizerCommand,
        "setup_dev": SetupDevCommand,
    },
)
