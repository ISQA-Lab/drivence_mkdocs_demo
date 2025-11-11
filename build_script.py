#!/usr/bin/env python3
"""
Utility script for producing a distributable wheel from precompiled artifacts.

The script expects a directory containing the compiled `.so/.pyi` files laid out
as a Python package (see `selda_core/` for a reference structure). Command-line
arguments let you customize artifact, output, and package names.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

DEFAULT_SOURCE_PACKAGE = "lidar_core"
DEFAULT_STAGE_DIR = "lidar_core"
DEFAULT_DIST_DIR = "lidar_core_dist"
DEFAULT_PACKAGE_NAME = "lidar_core"
DEFAULT_VERSION = "1.0.0"
DEFAULT_DESCRIPTION = "Binary distribution of the Selda core package"
DEFAULT_PYTHON_REQUIRES = ">=3.8"


def copy_artifacts(source_dir: Path, staging_dir: Path, package: str) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Artifact directory '{source_dir}' does not exist. "
            "Ensure the compiled .so/.pyi artifacts are available before building the wheel."
        )

    target_dir = staging_dir / package
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    init_file = target_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")


def write_setup_py(staging_dir: Path, package_name: str, version: str, description: str) -> None:
    setup_content = textwrap.dedent(
        f"""\
from setuptools import find_packages, setup


setup(
    name="{package_name}",
    version="{version}",
    description="{description}",
    packages=find_packages(),
    include_package_data=True,
    package_data={{"": ["*.so", "*.pyd", "*.pyi"]}},
    python_requires="{DEFAULT_PYTHON_REQUIRES}",
    zip_safe=False,
)
"""
    ).strip()
    (staging_dir / "setup.py").write_text(setup_content + "\n", encoding="utf-8")


def build_wheel(staging_dir: Path, dist_dir: Path) -> Path:
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "setup.py",
        "bdist_wheel",
        "--py-limited-api",
        "cp38",
    ]
    result = subprocess.run(cmd, cwd=staging_dir)
    if result.returncode != 0:
        raise RuntimeError("Wheel build failed; inspect the console output for details.")

    wheel_src_dir = staging_dir / "dist"
    wheel_files = list(wheel_src_dir.glob("*.whl"))
    if not wheel_files:
        raise FileNotFoundError("No wheel was generated inside the temporary build directory.")

    for wheel_path in wheel_files:
        destination = dist_dir / wheel_path.name
        shutil.copy2(wheel_path, destination)
        print(f"✓ Copied wheel to: {destination}")
    return dist_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wheel from precompiled Selda artifacts.")
    parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_STAGE_DIR,
        help=f"Directory containing the precompiled package artifacts (default: {DEFAULT_STAGE_DIR})",
    )
    parser.add_argument(
        "--dist-dir",
        default=DEFAULT_DIST_DIR,
        help=f"Destination directory for the generated wheel (default: {DEFAULT_DIST_DIR})",
    )
    parser.add_argument(
        "--package",
        default=DEFAULT_SOURCE_PACKAGE,
        help=f"Package directory name to include in the wheel (default: {DEFAULT_SOURCE_PACKAGE})",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_PACKAGE_NAME,
        help=f"Published wheel name (default: {DEFAULT_PACKAGE_NAME})",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Wheel version (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="Wheel description",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    artifact_dir = (project_root / args.artifact_dir).resolve()
    dist_dir = (project_root / args.dist_dir).resolve()

    with tempfile.TemporaryDirectory(prefix="drivence_wheel_") as tmp_dir:
        staging_dir = Path(tmp_dir)
        (staging_dir / args.package).mkdir(parents=True, exist_ok=True)
        copy_artifacts(artifact_dir, staging_dir, args.package)
        write_setup_py(staging_dir, args.name, args.version, args.description)
        build_wheel(staging_dir, dist_dir)


if __name__ == "__main__":
    main()

