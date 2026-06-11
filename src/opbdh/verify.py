from __future__ import annotations

import py_compile
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


SKIP_DIRS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "node_modules"}


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    checked: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _iter_code_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part in SKIP_DIRS for part in candidate.relative_to(path).parts):
            continue
        if candidate.suffix in {".py", ".sh", ".bash"}:
            files.append(candidate)
    return files


def default_command_for_code(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.is_file():
        remote = f"/opbdh-run/user/{path.name}"
        if path.suffix == ".py":
            return f"python {remote}"
        if path.suffix in {".sh", ".bash"}:
            return f"bash {remote}"
        raise ValueError("Pass --command for non-Python/non-shell code files.")

    if (path / "run.py").exists():
        return "python /opbdh-run/user/run.py"
    if (path / "main.py").exists():
        return "python /opbdh-run/user/main.py"
    if (path / "run.sh").exists():
        return "bash /opbdh-run/user/run.sh"
    raise ValueError("Pass --command when the code directory has no run.py, main.py, or run.sh.")


def verify_code(path: Path, *, command: str = "") -> VerificationResult:
    path = path.expanduser().resolve()
    result = VerificationResult(ok=True)
    if not path.exists():
        return VerificationResult(ok=False, errors=[f"Code path does not exist: {path}"])
    if path.is_dir() and not command:
        try:
            default_command_for_code(path)
        except ValueError as exc:
            result.ok = False
            result.errors.append(str(exc))

    for file_path in _iter_code_files(path):
        result.checked.append(file_path)
        try:
            if file_path.suffix == ".py":
                py_compile.compile(str(file_path), doraise=True)
            elif file_path.suffix in {".sh", ".bash"} and shutil.which("bash"):
                completed = subprocess.run(["bash", "-n", str(file_path)], capture_output=True, text=True)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip())
        except Exception as exc:
            result.ok = False
            result.errors.append(f"{file_path}: {exc}")

    if path.is_file() and path.suffix not in {".py", ".sh", ".bash"} and not command:
        result.ok = False
        result.errors.append("Pass --command for non-Python/non-shell code files.")
    return result
