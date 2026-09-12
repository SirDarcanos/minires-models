"""Compute an identity for the installed MiniRes Python source."""

from pathlib import Path

from .ingestion import fingerprint


def code_fingerprint() -> str:
    """Return a stable fingerprint covering every Python module in the package."""
    package_root = Path(__file__).parent
    sources = {
        path.relative_to(package_root).as_posix(): path.read_text()
        for path in sorted(package_root.rglob("*.py"))
    }
    return fingerprint(sources)
