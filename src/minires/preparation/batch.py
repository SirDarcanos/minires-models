"""Resumable private preparation for a directory of pre-supported STL files."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import as_file
import json
import math
from pathlib import Path
import stat
import time
from typing import Any, Mapping

from ..private_io import replace_private_json
from .stl import (
    BUNDLED_PROFILE_RESOURCE,
    DEFAULT_TIMEOUT_S,
    ProcessRunner,
    StlPreparationResult,
    SubprocessRunner,
    _contract,
    _preflight,
    _prepare_stl_after_preflight,
)

BATCH_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
MEASUREMENT_ALGORITHM_VERSION = "minires-stl-measurement-v1"
MAX_BATCH_WORKERS = 4


class BatchPreparationError(ValueError):
    """A bounded run-level batch failure."""


@dataclass(frozen=True)
class StlBatchPreparationResult:
    outcome: str
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _inventory_file(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    entry: dict[str, Any] = {
        "entry_id": sha256(relative.encode("utf-8")).hexdigest(),
        "relative_path": relative,
        "byte_count": None,
        "sha256": None,
        "status": "rejected",
        "reason": "inventory_unreadable",
    }
    try:
        before = path.lstat()
        entry["byte_count"] = before.st_size
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            entry["reason"] = "unsupported_inventory_entry"
            return entry
        digest = sha256()
        byte_count = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
        after = path.stat()
        entry["byte_count"] = byte_count
        entry["sha256"] = digest.hexdigest()
        unchanged = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not unchanged:
            entry["reason"] = "inventory_changed"
        elif path.suffix.lower() != ".stl":
            entry["reason"] = "unsupported_inventory_entry"
        elif byte_count == 0:
            entry["reason"] = "zero_length"
        else:
            entry["status"] = "candidate"
            entry["reason"] = None
        return entry
    except OSError:
        return entry


def _inventory(root: Path) -> list[dict[str, Any]]:
    try:
        if not root.is_dir():
            raise BatchPreparationError("input_directory_unavailable")
        paths = []
        for path in root.rglob("*"):
            try:
                mode = path.lstat().st_mode
                if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                    continue
            except OSError:
                pass
            paths.append(path)
        paths.sort(key=lambda path: path.relative_to(root).as_posix())
    except OSError:
        raise BatchPreparationError("input_directory_unavailable") from None
    return [_inventory_file(path, root) for path in paths]


def _preflight_contract(
    runner: ProcessRunner, timeout_s: float
) -> tuple[dict[str, Any], dict[str, str], str, bytes]:
    try:
        with as_file(BUNDLED_PROFILE_RESOURCE) as profile_path:
            profile = profile_path.read_bytes()
            rejection, versions, _, profile_digest = _preflight(
                runner, timeout_s, profile_path
            )
    except OSError:
        raise BatchPreparationError("profile_unavailable") from None
    if rejection is not None:
        raise BatchPreparationError(rejection)
    assert profile_digest is not None
    contract = _contract(profile_digest)
    envelope = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "measurement_algorithm_version": MEASUREMENT_ALGORITHM_VERSION,
        "record_schema_version": 1,
        "contract": contract,
        "tool_versions": versions,
    }
    return contract, versions, sha256(_canonical_bytes(envelope)).hexdigest(), profile


def _checkpoint_path(checkpoint_dir: Path, input_digest: str, fingerprint: str) -> Path:
    key = sha256(f"{input_digest}:{fingerprint}".encode("ascii")).hexdigest()
    return checkpoint_dir / f"{key}.json"


def _load_checkpoint(
    path: Path, input_digest: str, fingerprint: str
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("complete") is not True):
        return None
    if payload.get("input_sha256") != input_digest or payload.get("contract_fingerprint") != fingerprint:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    outcome = result.get("outcome")
    if outcome not in {"prepared", "rejected"}:
        return None
    if outcome == "prepared":
        record = result.get("record")
        if not isinstance(record, dict):
            return None
        inventory = record.get("input_inventory")
        if not isinstance(inventory, dict) or inventory.get("sha256") != input_digest:
            return None
    elif result.get("record") is not None:
        return None
    return payload


def _checkpoint_payload(
    input_digest: str,
    fingerprint: str,
    result: StlPreparationResult,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "complete": True,
        "input_sha256": input_digest,
        "contract_fingerprint": fingerprint,
        "elapsed_seconds": elapsed_seconds,
        "result": result.to_dict(),
    }


def prepare_stl_batch(
    input_directory: str | Path,
    *,
    checkpoint_dir: str | Path,
    runner: ProcessRunner | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    scope_confirmed: bool = False,
    workers: int = 1,
) -> StlBatchPreparationResult:
    """Prepare and reconcile every entry in a private directory."""
    if not scope_confirmed:
        raise BatchPreparationError("scope_confirmation_required")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_BATCH_WORKERS:
        raise BatchPreparationError("invalid_worker_count")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise BatchPreparationError("invalid_timeout")
    root = Path(input_directory)
    checkpoints = Path(checkpoint_dir)
    process_runner = runner or SubprocessRunner()
    started = time.monotonic()
    contract, versions, fingerprint, profile = _preflight_contract(
        process_runner, float(timeout_s)
    )
    inventory = _inventory(root)

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in inventory:
        if entry["status"] == "candidate":
            groups.setdefault(entry["sha256"], []).append(entry)

    outcomes: dict[str, dict[str, Any]] = {}
    reused = 0
    pending: list[tuple[str, list[dict[str, Any]], Path]] = []
    for digest, entries in groups.items():
        checkpoint_path = _checkpoint_path(checkpoints, digest, fingerprint)
        cached = _load_checkpoint(checkpoint_path, digest, fingerprint)
        if cached is not None:
            outcomes[digest] = cached
            reused += 1
        else:
            pending.append((digest, entries, checkpoint_path))

    def process(item: tuple[str, list[dict[str, Any]], Path]) -> tuple[str, dict[str, Any], Path]:
        digest, entries, checkpoint_path = item
        item_started = time.monotonic()
        source = root / entries[0]["relative_path"]
        result = _prepare_stl_after_preflight(
            source,
            expected_sha256=digest,
            profile=profile,
            runner=process_runner,
            timeout_s=float(timeout_s),
            versions=versions,
            profile_digest=contract["profile_sha256"],
        )
        payload = _checkpoint_payload(
            digest, fingerprint, result, time.monotonic() - item_started
        )
        replace_private_json(checkpoint_path, payload)
        return digest, payload, checkpoint_path

    if workers == 1:
        for item in pending:
            digest, payload, _ = process(item)
            outcomes[digest] = payload
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(process, item) for item in pending]
        try:
            for future in as_completed(futures):
                digest, payload, _ = future.result()
                outcomes[digest] = payload
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for digest, entries in groups.items():
        duplicate_group = (
            sha256(f"exact-duplicate:{digest}".encode("ascii")).hexdigest()
            if len(entries) > 1 else None
        )
        if duplicate_group:
            duplicate_groups.append({
                "duplicate_group": duplicate_group,
                "sha256": digest,
                "entry_ids": [entry["entry_id"] for entry in entries],
                "count": len(entries),
            })
        result = outcomes[digest]["result"]
        for entry in entries:
            if result["outcome"] == "prepared":
                accepted.append({
                    "entry_id": entry["entry_id"],
                    "input_sha256": digest,
                    "duplicate_group": duplicate_group,
                    "record": result["record"],
                })
            else:
                rejected.append({
                    "entry_id": entry["entry_id"],
                    "input_sha256": digest,
                    "duplicate_group": duplicate_group,
                    "reason": result["rejection"],
                })
    for entry in inventory:
        if entry["status"] == "rejected":
            rejected.append({
                "entry_id": entry["entry_id"],
                "input_sha256": entry["sha256"],
                "duplicate_group": None,
                "reason": entry["reason"],
            })

    accepted.sort(key=lambda item: item["entry_id"])
    rejected.sort(key=lambda item: item["entry_id"])
    reasons = Counter(item["reason"] for item in rejected)
    inventory_count = len(inventory)
    accepted_count = len(accepted)
    rejected_count = len(rejected)
    accounting = {
        "inventory_count": inventory_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejection_reasons": dict(sorted(reasons.items())),
        "unique_candidate_count": len(groups),
        "processed_unique_outcomes": len(pending),
        "reused_unique_outcomes": reused,
    }
    if inventory_count != accepted_count + rejected_count:
        raise BatchPreparationError("reconciliation_failed")
    report = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "outcome": "completed",
        "inventory": inventory,
        "accepted_records": accepted,
        "rejected_entries": rejected,
        "duplicate_groups": duplicate_groups,
        "accounting": accounting,
        "provenance": {
            "contract": contract,
            "contract_fingerprint": fingerprint,
            "measurement_algorithm_version": MEASUREMENT_ALGORITHM_VERSION,
            "tool_versions": versions,
            "worker_count": workers,
        },
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "completed_checkpoint_elapsed_seconds": sum(
                float(payload.get("elapsed_seconds", 0.0)) for payload in outcomes.values()
            ),
        },
    }
    return StlBatchPreparationResult("completed", report)


def write_batch_result(result: StlBatchPreparationResult, output: str | Path) -> None:
    replace_private_json(Path(output), result.to_dict())
