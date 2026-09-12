"""Private command and file boundaries."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Never


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.exit(2, "Invalid command arguments; use --help.\n")


def create_private_file(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb")


def is_private_path(path: Path) -> bool:
    return "private" in path.resolve().parts


def write_private_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with create_private_file(path) as stream:
        stream.write(content.encode("utf-8"))


def replace_private_json(path: Path, value: Any) -> None:
    """Durably replace private JSON without exposing a partial destination."""
    if not is_private_path(path):
        raise ValueError("private_output_required")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
