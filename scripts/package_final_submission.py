"""Package the final reproducible project submission."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_project_dir() -> Path:
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_exclude(path: Path, project_dir: Path) -> bool:
    parts = set(path.relative_to(project_dir).parts)
    if "__pycache__" in parts or ".ipynb_checkpoints" in parts:
        return True
    if path.suffix == ".pyc":
        return True
    if path.suffix.lower() == ".zip":
        return True
    relative = path.relative_to(project_dir).as_posix()
    if relative.startswith("data/raw/"):
        return True
    if path.name in {"ecommerce_sentiment_rag_final_submission.zip"}:
        return True
    return False


def iter_files(path: Path, project_dir: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [] if should_exclude(path, project_dir) else [path]
    files = []
    for item in path.rglob("*"):
        if item.is_file() and not should_exclude(item, project_dir):
            files.append(item)
    return files


def main() -> None:
    project_dir = resolve_project_dir()
    output_zip = project_dir / "ecommerce_sentiment_rag_final_submission.zip"
    manifest_path = project_dir / "artifact_manifest.json"

    include_paths = [
        "src",
        "scripts",
        "requirements_module1_2.txt",
        "requirements_module4.txt",
        "requirements_module6.txt",
        "docs",
        "README_MODULE1.md",
        "README_MODULE2.md",
        "README_MODULE3.md",
        "README_MODULE4.md",
        "README_MODULE5.md",
        "README_MODULE6.md",
        "data/processed",
        "data/eval",
        "data/reports",
        "data/evaluation_queries.json",
        "models/module2",
        "models/module4",
        "results/module2",
        "results/module3",
        "results/module4",
        "results/module5",
        "results/module6",
        "reports",
        "figures/module1",
        "figures/module2",
        "figures/module3",
        "figures/module4",
        "figures/module5",
        "figures/module6",
    ]

    missing_optional_paths = []
    files: list[Path] = []
    for relative in include_paths:
        path = project_dir / relative
        if not path.exists():
            missing_optional_paths.append(relative)
            continue
        files.extend(iter_files(path, project_dir))

    files = sorted(set(files), key=lambda path: path.relative_to(project_dir).as_posix())

    manifest_entries = [
        {
            "path": path.relative_to(project_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir),
        "missing_optional_paths": missing_optional_paths,
        "files": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    packaged_files = files + [manifest_path]
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in packaged_files:
            arcname = path.relative_to(project_dir).as_posix()
            zip_file.write(path, arcname)

    print(f"[package] Saved: {output_zip}")
    print(f"[package] Saved manifest: {manifest_path}")
    print(f"[package] Packaged files: {len(packaged_files)}")
    if missing_optional_paths:
        print("[package] Missing optional paths:")
        for path in missing_optional_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
