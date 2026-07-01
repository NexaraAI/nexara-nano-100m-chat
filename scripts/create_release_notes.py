"""Create release notes and package metadata for tagged releases."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import zipfile


def main() -> None:
    dist = Path("dist")
    dist.mkdir(exist_ok=True)
    tag = git_output("describe", "--tags", "--always")
    latest_commit = git_output("rev-parse", "--short", "HEAD")
    notes = dist / "RELEASE_NOTES.md"
    notes.write_text(
        "\n".join(
            [
                f"# Nexara {tag}",
                "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Commit: `{latest_commit}`",
                "",
                "## Pre-trained Weights",
                "",
                "Model weights and tokenizer files for **Nexara Nano 100M-Chat** are hosted on Hugging Face:",
                "- 🤗 **[Hugging Face Model Page](https://huggingface.co/Emperordzd/Nexara-Nano-100M-Chat)**",
                "",
                "## Included Artifacts",
                "",
                "- Configuration files from `configs/`",
                "- Project documentation markdown files",
                "- Project roadmap and progress logs",
                "",
            ]
        ),
        encoding="utf-8",
    )

    package = dist / f"nexara-{tag}-configs-docs.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_inputs():
            if path.exists():
                archive.write(path, path.as_posix())
    print(f"wrote {notes}")
    print(f"wrote {package}")


def package_inputs() -> list[Path]:
    paths = [
        Path("README.md"),
        Path("ROADMAP.md"),
        Path("PROGRESS.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
    ]
    paths.extend(Path("configs").glob("*.toml"))
    paths.extend(Path("docs").glob("*.md"))
    return sorted(paths)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


if __name__ == "__main__":
    main()
