"""Private command and file boundaries."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Never


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.exit(2, "Invalid command arguments; use --help.\n")


def create_private_file(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb")


def write_private_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with create_private_file(path) as stream:
        stream.write(content.encode("utf-8"))
