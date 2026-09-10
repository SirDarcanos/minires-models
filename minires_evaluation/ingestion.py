"""Internal local adapters and the versioned, non-fitting normalization contract."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

TRANSFORMATION_VERSION = "minires-normalization-v2"
VOLUME_FACTORS = {"mm3": 1.0, "cm3": 1000.0, "ml": 1000.0}
# Only these measurements may enter a prediction feature table.
FEATURE_ALIASES = {
    "volume_mm3": "volume",
    "surface_area_mm2": "surface_area",
    "bounding_box_x_mm": "bbox_x",
    "bounding_box_y_mm": "bbox_y",
    "bounding_box_z_mm": "bbox_z",
    "bounding_box_volume_mm3": "bbox_area",
    "euler_number": "euler_number",
}
Dataset = Sequence[Mapping[str, Any]] | str | Path


class InputError(ValueError):
    """A bounded error whose text contains no input values or paths."""


@dataclass(frozen=True)
class CanonicalRow:
    row_index: int
    features: dict[str, float | None]
    sliced_resin_mass_g: float | None
    outcome: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]


def fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def load_records(dataset: Dataset) -> tuple[list[Any], str]:
    records: list[Any]
    if not isinstance(dataset, (str, Path)):
        records = list(dataset)
        return records, fingerprint(records)
    try:
        path = Path(dataset)
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        if path.suffix.lower() == ".csv":
            records = list(csv.DictReader(io.StringIO(text)))
        elif path.suffix.lower() == ".jsonl":
            # Bad lines remain individually accountable, with no raw text retained.
            records = []
            for line in io.StringIO(text):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append(None)
        elif path.suffix.lower() == ".json":
            records = json.loads(text)
            if not isinstance(records, list):
                raise InputError("records_array_required")
        else:
            raise InputError("unsupported_local_format")
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError):
        raise InputError("local_input_unreadable") from None
    return records, sha256(content).hexdigest()


def number(value: Any) -> float | None:
    if isinstance(value, dict) and len(value) == 1:
        key = next(iter(value))
        if key in {"$numberDouble", "$numberInt", "$numberLong"}:
            value = value[key]
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        return float(value)
    except (ValueError, OverflowError):
        return None


def _token(value: Any) -> str | None:
    if isinstance(value, dict) and set(value) == {"$oid"}:
        value = value["$oid"]
    return None if value is None or value == "" else fingerprint(value)


def normalize(records: Sequence[Any], config: Any) -> list[CanonicalRow]:
    rows = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            rows.append(CanonicalRow(index, dict.fromkeys(FEATURE_ALIASES), None,
                                     "excluded", ("invalid_record",), (), {}))
            continue
        reasons: list[str] = []
        warnings: list[str] = []
        unit = raw.get("volume_unit", config.volume_unit)
        factor = VOLUME_FACTORS.get(unit) if isinstance(unit, str) else None
        features: dict[str, float | None] = {}
        for canonical, alias in FEATURE_ALIASES.items():
            value = number(raw.get(canonical, raw.get(alias)))
            if value is not None and not math.isfinite(value):
                reasons.append("non_finite_" + canonical)
                value = None
            elif value is not None and (value <= 0 if canonical != "euler_number" else not value.is_integer()):
                reasons.append("invalid_volume" if canonical == "volume_mm3" else "invalid_" + canonical)
                value = None
            elif value is None and canonical == "volume_mm3":
                reasons.append("invalid_volume")
            elif value is None and raw.get(canonical, raw.get(alias)) not in (None, ""):
                reasons.append("invalid_" + canonical)
            # Unit-bearing canonical fields are already mm-based. Legacy bbox and
            # surface measurements have a fixed mm contract independent of volume_unit.
            if canonical == "volume_mm3" and canonical not in raw:
                value = value * factor if value is not None and factor is not None else None
                if value is not None and not math.isfinite(value):
                    reasons.append("non_finite_volume_mm3")
                    value = None
            features[canonical] = value
        target_raw = raw.get("sliced_resin_mass_g", raw.get("weight"))
        target = number(target_raw)
        if target_raw is None or target_raw == "":
            reasons.append("missing_target_sliced_resin_mass")
        elif target is not None and not math.isfinite(target):
            reasons.append("non_finite_target_sliced_resin_mass")
            target = None
        elif target is None or target < 0:
            reasons.append("invalid_target_sliced_resin_mass")
            target = None
        base = number(raw.get("base_mm"))
        if base is not None and (not math.isfinite(base) or base <= 0):
            base = None
        if base is None and raw.get("base_mm") not in (None, ""):
            warnings.append("invalid_optional_base_mm")
        density = number(raw.get("resin_density_g_per_ml", config.resin_density_g_per_ml))
        scope = raw.get("scope_confirmed", config.scope_confirmed)
        if isinstance(scope, str) and scope.lower() in {"true", "false"}:
            scope = scope.lower() == "true"
        excluded = any(reason != "missing_target_sliced_resin_mass" for reason in reasons)
        if factor is None and "volume_mm3" not in raw:
            reasons.append("unsupported_volume_unit")
        if density is None:
            reasons.append("resin_density_required")
        elif not math.isfinite(density) or density <= 0:
            reasons.append("invalid_resin_density")
            density = None
        if scope is False:
            reasons.append("unsupported_scope")
        elif scope is not True:
            reasons.append("scope_confirmation_required")
        if not isinstance(config.tolerance_g, (int, float)) or isinstance(config.tolerance_g, bool) or not math.isfinite(config.tolerance_g) or config.tolerance_g <= 0:
            reasons.append("invalid_tolerance")
        volume = features["volume_mm3"]
        if not reasons and volume is not None and density is not None:
            prediction = volume / 1000.0 * density
            if not math.isfinite(prediction):
                reasons.append("non_finite_prediction")
        conditions = raw.get("slicing_conditions")
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError:
                conditions = None
                warnings.append("invalid_optional_slicing_conditions")
        safe_conditions = {}
        if isinstance(conditions, Mapping):
            for key in ("layer_height_mm", "exposure_seconds", "bottom_exposure_seconds"):
                value = number(conditions.get(key))
                if value is not None and math.isfinite(value) and value > 0:
                    safe_conditions[key] = value
        metadata = {
            "anonymous_source_group": _token(raw.get("anonymous_source_group", raw.get("artist"))),
            "source_identity_evidence": _token(raw.get("artist")),
            "duplicate_group": _token(raw.get("duplicate_group")),
            "geometry_fingerprint": _token(raw.get("geometry_fingerprint")),
            "miniature_family": _token(raw.get("miniature_family")),
            "record_identity": _token(raw.get("_id")),
            "location_evidence": _token(raw.get("file")),
            "source_name_evidence": _token([raw.get("artist"), raw.get("mini")])
                if raw.get("artist") and raw.get("mini") else None,
            "base_mm": base,
            "geometry_valid": None,
            "support_presence": None,
            "volume_unit": "mm3" if "volume_mm3" in raw else (unit if factor is not None else None),
            "resin_density_g_per_ml": density,
            "density_origin": "record" if "resin_density_g_per_ml" in raw else "configuration",
            "scope_confirmed": scope if isinstance(scope, bool) else None,
            "scope_confirmation_origin": "record" if "scope_confirmed" in raw else "configuration",
            "slicing_conditions": safe_conditions or None,
            "slicing_conditions_fingerprint": _token(conditions),
        }
        for alias in ("kb", "mass", "scale"):
            value = number(raw.get(alias))
            if value is not None and (not math.isfinite(value) or value < 0):
                value = None
            if value is None and raw.get(alias) not in (None, ""):
                warnings.append("invalid_optional_" + alias)
            metadata["legacy_" + alias + "_unit_unknown"] = value
        rows.append(CanonicalRow(index, features, target, "excluded" if excluded else "needs_review" if reasons else "included",
                                 tuple(reasons), tuple(warnings), metadata))
    return rows
