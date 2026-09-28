"""Example: release validation, collected dependencies, and mapped builds.

The fixed validation tasks fan out from one package snapshot. Their results are
collected into a keyed dictionary before a build task is expanded over the
configured release targets.

    LoadPackage ─┬─ RunTests ──┐
                 ├─ RunLint ───┼─ ValidateRelease ── BuildTarget[*] ── Manifest
                 └─ CheckTypes ┘

Run:
    python examples/release_pipeline/pipeline.py
    python examples/release_pipeline/pipeline.py --yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from taskmaestro import (
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    Runner,
    Task,
    TaskMap,
    Workflow,
    collect,
)
from taskmaestro.hooks import LoggingHook, TimingHook

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PackageInput(BaseModel):
    """Source package supplied as job configuration."""

    name: str
    version: str
    files: dict[str, str]


class PackageSnapshot(BaseModel):
    """Immutable package information passed to each validation task."""

    name: str
    version: str
    files: dict[str, str]
    source_digest: str


class CheckResult(BaseModel):
    """Common output type that allows validation results to be collected."""

    passed: bool
    message: str


class ValidateReleaseInput(BaseModel):
    package: PackageSnapshot
    checks: dict[str, CheckResult]


class ValidatedRelease(BaseModel):
    package: PackageSnapshot
    checks: dict[str, CheckResult]


class TargetSettings(BaseModel):
    operating_system: str
    architecture: str
    extension: str


class BuildTargetInput(BaseModel):
    release: ValidatedRelease
    target_name: str
    settings: TargetSettings


class Artifact(BaseModel):
    target: str
    path: str
    sha256: str


class ManifestInput(BaseModel):
    artifacts: dict[str, Artifact]


class ReleaseManifest(BaseModel):
    manifest_path: str
    artifacts: dict[str, Artifact]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class LoadPackage(Task[PackageInput, PackageSnapshot]):
    name = "load_package"

    def run(self, input: PackageInput, ctx: ExecutionContext) -> PackageSnapshot:
        serialized_files = json.dumps(input.files, sort_keys=True).encode()
        digest = hashlib.sha256(serialized_files).hexdigest()
        ctx.logger.info(
            "Loaded %s %s with %d files",
            input.name,
            input.version,
            len(input.files),
        )
        return PackageSnapshot(
            name=input.name,
            version=input.version,
            files=input.files,
            source_digest=digest,
        )


class RunTests(Task[PackageSnapshot, CheckResult]):
    name = "run_tests"

    def run(self, input: PackageSnapshot, ctx: ExecutionContext) -> CheckResult:
        failing_files = [name for name, content in input.files.items() if "FAIL_TEST" in content]
        return CheckResult(
            passed=not failing_files,
            message=(
                "All tests passed"
                if not failing_files
                else f"Test failures in: {', '.join(failing_files)}"
            ),
        )


class RunLint(Task[PackageSnapshot, CheckResult]):
    name = "run_lint"

    def run(self, input: PackageSnapshot, ctx: ExecutionContext) -> CheckResult:
        files_with_tabs = [name for name, content in input.files.items() if "\t" in content]
        return CheckResult(
            passed=not files_with_tabs,
            message=(
                "No lint errors"
                if not files_with_tabs
                else f"Tabs found in: {', '.join(files_with_tabs)}"
            ),
        )


class CheckTypes(Task[PackageSnapshot, CheckResult]):
    name = "check_types"

    def run(self, input: PackageSnapshot, ctx: ExecutionContext) -> CheckResult:
        ignored_files = [
            name for name, content in input.files.items() if "# type: ignore" in content
        ]
        return CheckResult(
            passed=not ignored_files,
            message=(
                "Type checks passed"
                if not ignored_files
                else f"Unchecked types in: {', '.join(ignored_files)}"
            ),
        )


class ValidateRelease(Task[ValidateReleaseInput, ValidatedRelease]):
    name = "validate_release"

    def run(self, input: ValidateReleaseInput, ctx: ExecutionContext) -> ValidatedRelease:
        failed = [name for name, result in input.checks.items() if not result.passed]
        if failed:
            details = "; ".join(f"{name}: {input.checks[name].message}" for name in failed)
            raise ValueError(f"Release validation failed: {details}")

        ctx.logger.info("All %d release checks passed", len(input.checks))
        return ValidatedRelease(package=input.package, checks=input.checks)


class BuildTarget(Task[BuildTargetInput, Artifact]):
    name = "build_target"

    def run(self, input: BuildTargetInput, ctx: ExecutionContext) -> Artifact:
        package = input.release.package
        settings = input.settings
        ctx.scratch_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{package.name}-{package.version}-{settings.operating_system}-"
            f"{settings.architecture}.{settings.extension}"
        )
        artifact_path = ctx.scratch_dir / filename
        payload = {
            "package": package.name,
            "version": package.version,
            "source_digest": package.source_digest,
            "target": input.target_name,
            "operating_system": settings.operating_system,
            "architecture": settings.architecture,
        }
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        ctx.logger.info("Built %s at %s", input.target_name, artifact_path)
        return Artifact(
            target=input.target_name,
            path=str(artifact_path),
            sha256=digest,
        )


class CreateReleaseManifest(Task[ManifestInput, ReleaseManifest]):
    name = "create_release_manifest"

    def run(self, input: ManifestInput, ctx: ExecutionContext) -> ReleaseManifest:
        ctx.scratch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = ctx.scratch_dir / "release-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {name: artifact.model_dump() for name, artifact in input.artifacts.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        return ReleaseManifest(
            manifest_path=str(manifest_path),
            artifacts=input.artifacts,
        )


# ---------------------------------------------------------------------------
# Workflow and execution
# ---------------------------------------------------------------------------


def build_workflow() -> Workflow:
    """Build the release DAG using unambiguous task handles."""
    builder = Workflow.builder("release_pipeline")
    package = builder.task(
        LoadPackage,
        config_fields=["name", "version", "files"],
    )
    tests = builder.task(RunTests, depends_on=package)
    lint = builder.task(RunLint, depends_on=package)
    types = builder.task(CheckTypes, depends_on=package)
    validated = builder.task(
        ValidateRelease,
        depends_on={
            "package": package,
            "checks": collect(tests=tests, lint=lint, types=types),
        },
    )
    builds = builder.task(
        BuildTarget,
        name="build_targets",
        depends_on={"release": validated},
        mapped_over=TaskMap(
            over="targets",
            key_as="target_name",
            value_as="settings",
            error_mode="collect_all",
        ),
    )
    builder.task(
        CreateReleaseManifest,
        depends_on={"artifacts": builds},
    )
    return builder.build()


def sample_job_configuration() -> JobConfiguration:
    return JobConfiguration(
        {
            "load_package": {
                "name": "taskmaestro-demo",
                "version": "1.0.0",
                "files": {
                    "src/demo.py": "def greet(name: str) -> str:\n    return f'Hello {name}'\n",
                    "tests/test_demo.py": "def test_greet():\n    assert True\n",
                },
            },
            "build_targets": {
                "targets": {
                    "linux-x64": {
                        "operating_system": "linux",
                        "architecture": "x86_64",
                        "extension": "tar.gz",
                    },
                    "windows-x64": {
                        "operating_system": "windows",
                        "architecture": "x86_64",
                        "extension": "zip",
                    },
                    "macos-arm64": {
                        "operating_system": "macos",
                        "architecture": "arm64",
                        "extension": "tar.gz",
                    },
                }
            },
        }
    )


def print_result(result: Job[Any], timing: TimingHook, workflow: Workflow) -> None:
    print("=" * 60)
    print("Release pipeline")
    print("=" * 60)
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
        return

    assert result.result is not None
    manifest = cast(ReleaseManifest, result.result)
    print(f"Manifest: {manifest.manifest_path}")
    for name, artifact in manifest.artifacts.items():
        print(f"  {name:16s} {artifact.path}")
    print(f"Duration: {timing.job_duration:.4f}s")
    print("\nMermaid diagram:")
    print("```mermaid")
    print(workflow.to_mermaid(), end="")
    print("```")


def run_python_mode() -> None:
    workflow = build_workflow()
    timing = TimingHook()
    result = Runner(hooks=[LoggingHook(), timing]).run(
        Job(
            workflow,
            EmptyConfig(),
            job_configuration=sample_job_configuration(),
        )
    )
    print_result(result, timing, workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    from taskmaestro.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    result = loaded.run()
    timing = next(hook for hook in loaded.runner.hooks if isinstance(hook, TimingHook))
    print_result(result, timing, loaded.workflow)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Mapped release pipeline example")
    parser.add_argument(
        "--yaml",
        metavar="FILE",
        nargs="?",
        const=str(example_dir / "workflow.yaml"),
        help="Load YAML workflow (default: workflow.yaml)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        default=str(example_dir / "input.yaml"),
        help="Input YAML file (default: input.yaml)",
    )
    args = parser.parse_args()

    if args.yaml:
        run_yaml_mode(args.yaml, args.input)
    else:
        run_python_mode()


if __name__ == "__main__":
    main()
