"""Export identity-only batch inventory without decoding prepared record payloads."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..ingestion import InputError
from ..private_io import PrivateArgumentParser, is_private_path, write_private_json

VERSION = "minires-surface-signature-batch-linkage-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT = (
    PROJECT_ROOT / "private" / "geometry-feature-feasibility" / "batch-linkage-001.json"
)
_SELECTED_KEYS = {"schema_version", "outcome", "accounting", "inventory"}


@dataclass(frozen=True)
class BatchIdentityLinkageResult:
    status: str
    inventory_count: int


def _space(data: bytes, index: int) -> int:
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    return index


def _string_end(data: bytes, index: int) -> int:
    if index >= len(data) or data[index] != ord('"'):
        raise ValueError("invalid json string")
    index += 1
    escaped = False
    while index < len(data):
        value = data[index]
        if escaped:
            escaped = False
        elif value == ord("\\"):
            escaped = True
        elif value == ord('"'):
            return index + 1
        index += 1
    raise ValueError("unterminated json string")


def _value_end(data: bytes, index: int) -> int:
    if index >= len(data):
        raise ValueError("missing json value")
    if data[index] == ord('"'):
        return _string_end(data, index)
    if data[index] not in (ord("{"), ord("[")):
        end = index
        while end < len(data) and data[end] not in b",}\r\n\t ":
            end += 1
        if end == index:
            raise ValueError("invalid json value")
        return end
    pairs = {ord("{"): ord("}"), ord("["): ord("]")}
    stack = [pairs[data[index]]]
    cursor = index + 1
    while cursor < len(data) and stack:
        value = data[cursor]
        if value == ord('"'):
            cursor = _string_end(data, cursor)
            continue
        if value in pairs:
            stack.append(pairs[value])
        elif value in (ord("}"), ord("]")):
            if value != stack.pop():
                raise ValueError("invalid nested json")
        cursor += 1
    if stack:
        raise ValueError("unterminated json value")
    return cursor


def _select_top_level(data: bytes) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    cursor = _space(data, 0)
    if cursor >= len(data) or data[cursor] != ord("{"):
        raise ValueError("object required")
    cursor = _space(data, cursor + 1)
    while cursor < len(data) and data[cursor] != ord("}"):
        key_end = _string_end(data, cursor)
        try:
            key = json.loads(data[cursor:key_end])
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("invalid json key") from None
        if not isinstance(key, str):
            raise ValueError("invalid json key")
        cursor = _space(data, key_end)
        if cursor >= len(data) or data[cursor] != ord(":"):
            raise ValueError("missing json colon")
        value_start = _space(data, cursor + 1)
        value_end = _value_end(data, value_start)
        if key in _SELECTED_KEYS:
            if key in selected:
                raise ValueError("duplicate selected key")
            try:
                selected[key] = json.loads(data[value_start:value_end])
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("invalid selected json value") from None
        cursor = _space(data, value_end)
        if cursor < len(data) and data[cursor] == ord(","):
            cursor = _space(data, cursor + 1)
            continue
        if cursor >= len(data) or data[cursor] != ord("}"):
            raise ValueError("invalid top-level object")
    if cursor >= len(data) or data[cursor] != ord("}") or _space(data, cursor + 1) != len(data):
        raise ValueError("invalid top-level object")
    if set(selected) != _SELECTED_KEYS:
        raise ValueError("batch linkage fields missing")
    return selected


def export_batch_identity_linkage(
    *, batch_result: str | Path, private_output: str | Path,
) -> BatchIdentityLinkageResult:
    """Copy only inventory metadata and aggregate accounting from a batch result."""
    source = Path(batch_result)
    output = Path(private_output)
    if not is_private_path(source):
        raise InputError("private_batch_result_required")
    if not is_private_path(output):
        raise InputError("private_output_required")
    try:
        data = source.read_bytes()
        selected = _select_top_level(data)
    except (OSError, ValueError):
        raise InputError("batch_linkage_source_invalid") from None
    inventory = selected["inventory"]
    accounting = selected["accounting"]
    if (
        selected["schema_version"] != 1
        or selected["outcome"] != "completed"
        or not isinstance(inventory, list)
        or not isinstance(accounting, Mapping)
        or accounting.get("inventory_count") != len(inventory)
        or accounting.get("inventory_count")
        != accounting.get("accepted_count", -1) + accounting.get("rejected_count", -1)
    ):
        raise InputError("batch_linkage_source_invalid")
    payload = {
        "version": VERSION,
        "source_batch_sha256": sha256(data).hexdigest(),
        "schema_version": selected["schema_version"],
        "outcome": selected["outcome"],
        "accounting": accounting,
        "inventory": inventory,
        "record_payloads_decoded": False,
        "labels_decoded_or_inspected": False,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private_json(output, payload)
    except OSError:
        raise InputError("private_output_unavailable") from None
    return BatchIdentityLinkageResult("completed", len(inventory))


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Export checksum-bound identity-only STL batch linkage."
    )
    parser.add_argument("--batch-result", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.private_output.resolve() != PREDECLARED_OUTPUT.resolve():
            raise InputError("batch_linkage_output_mismatch")
        result = export_batch_identity_linkage(
            batch_result=args.batch_result,
            private_output=args.private_output,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    print(json.dumps({
        "status": result.status,
        "inventory_count": result.inventory_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
