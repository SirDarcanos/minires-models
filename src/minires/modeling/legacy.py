"""Pinned legacy-model compatibility contract and optional runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from ..ingestion import CanonicalRow, LEGACY_INFERENCE_FEATURES, fingerprint


LEGACY_REVISION = "ef3fe89d643739fa79da2930455eb02e26cc9e2e"
LEGACY_REPOSITORY = "nicolamustone/minires"
LEGACY_FEATURES = LEGACY_INFERENCE_FEATURES
LEGACY_PREPROCESSING_VERSION = "minires-legacy-notebook-2025-11-30"


@dataclass(frozen=True)
class PinnedArtifact:
    filename: str
    sha256: str
    size_bytes: int
    revision: str = LEGACY_REVISION


PINNED_ARTIFACTS = (
    PinnedArtifact("minires.keras", "369cbb70097ab12ca08cad89c4ad356e115f85feacebe1ebddc5222d92b4fe85", 9_992_026),
    PinnedArtifact("minires_xgb.json", "928bb7f87c704de47cc54d387a78e5eb74584c9ad117b24855a3abbf018f80ed", 16_117_079),
    PinnedArtifact("minires_meta.json", "12d17f5538081f8f47de7c752b3117e96a0da60edab37cfd073209b9c45d90f2", 159),
)


@dataclass(frozen=True)
class ArtifactResolution:
    status: str
    blockers: tuple[str, ...]
    paths: dict[str, Path]


@dataclass(frozen=True)
class LegacyProvenance:
    """What is known about overlap between legacy training data and evaluation data."""

    status: str
    evidence_fingerprint: str | None = None
    excluded_source_group_fingerprints: tuple[str, ...] = ()

    @classmethod
    def unknown(cls) -> LegacyProvenance:
        return cls("unknown")

    @classmethod
    def overlap(cls) -> LegacyProvenance:
        return cls("overlap")

    @classmethod
    def source_held_out(
        cls,
        excluded_source_groups: Sequence[str] = (),
        evidence: str | None = None,
    ) -> LegacyProvenance:
        digest = sha256(evidence.encode()).hexdigest() if evidence else None
        groups = tuple(sorted(fingerprint(group) for group in excluded_source_groups))
        return cls("source_held_out", digest, groups)

    @property
    def classification(self) -> str:
        return {
            "unknown": "legacy_reference_training_provenance_unknown",
            "overlap": "legacy_reference_training_overlap",
            "source_held_out": "legacy_reference_source_held_out",
        }.get(self.status, "legacy_reference_training_provenance_unknown")

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers = []
        if self.status == "source_held_out" and self.evidence_fingerprint is None:
            blockers.append("source_holdout_evidence_required")
        if self.status == "source_held_out" and not self.excluded_source_group_fingerprints:
            blockers.append("source_holdout_excluded_sources_required")
        if self.status not in {"unknown", "overlap", "source_held_out"}:
            blockers.append("invalid_legacy_provenance")
        return tuple(blockers)

    def validate_sources(self, rows: Sequence[CanonicalRow]) -> tuple[str, ...]:
        if self.status != "source_held_out" or self.blockers:
            return self.blockers
        evaluated = {row.metadata.get("anonymous_source_group") for row in rows}
        if not evaluated or None in evaluated:
            return ("source_holdout_evaluation_sources_required",)
        if not evaluated.issubset(set(self.excluded_source_group_fingerprints)):
            return ("source_holdout_evidence_mismatch",)
        return ()

    @property
    def private_evidence(self) -> dict[str, Any] | None:
        if self.evidence_fingerprint is None:
            return None
        return {
            "evidence_fingerprint": self.evidence_fingerprint,
            "excluded_source_group_count": len(self.excluded_source_group_fingerprints),
        }


BatchPredictor = Callable[[Sequence[tuple[float, ...]]], Sequence[float]]


def _predict_keras_model(model: Any, np: Any, rows: Sequence[tuple[float, ...]]) -> list[float]:
    matrix = np.asarray(rows, dtype=np.float32)
    return np.asarray(model(matrix, training=False)).reshape(-1).tolist()


@dataclass(frozen=True)
class LegacyReference:
    """The released neural network, XGBoost model, and fixed weighted ensemble."""

    neural_network: BatchPredictor | None
    xgboost: BatchPredictor | None
    neural_network_weight: float
    provenance: LegacyProvenance
    blockers: tuple[str, ...] = ()
    name: str = "legacy_ensemble"

    @classmethod
    def from_predictors(
        cls,
        *,
        neural_network: BatchPredictor,
        xgboost: BatchPredictor,
        neural_network_weight: float,
        provenance: LegacyProvenance,
    ) -> LegacyReference:
        blockers = list(provenance.blockers)
        if not math.isfinite(neural_network_weight) or not 0 <= neural_network_weight <= 1:
            blockers.append("invalid_legacy_ensemble_metadata")
        return cls(neural_network, xgboost, neural_network_weight, provenance, tuple(blockers))

    @classmethod
    def blocked(
        cls, blockers: Sequence[str], provenance: LegacyProvenance
    ) -> LegacyReference:
        return cls(None, None, 0.2, provenance, tuple(sorted(set(blockers) | set(provenance.blockers))))

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "classification": "legacy_reference_only",
            "repository": LEGACY_REPOSITORY,
            "revision": LEGACY_REVISION,
            "features": list(LEGACY_FEATURES),
            "feature_units": [
                "unit_unknown",
                "mm3",
                "mm2",
                "mm3",
                "dimensionless",
                "unit_unknown",
                "mm-1",
            ],
            "preprocessing_version": LEGACY_PREPROCESSING_VERSION,
            "released_inference_preprocessing": "feature_selection_and_order_only",
            "training_notebook_preprocessing": (
                "kb_integer_cast; volume_ceiling_0.1_mm3; selected_columns_round_0.1; "
                "surface_volume_ratio_recomputed_then_rounded_0.1"
            ),
            "known_training_inference_mismatch": True,
            "neural_network_normalization": "embedded_keras_normalization_layer",
            "xgboost_preprocessing": "same_legacy_feature_matrix_without_additional_normalization",
            "neural_network_weight": self.neural_network_weight,
            "xgboost_weight": 1.0 - self.neural_network_weight,
            "output_unit": "g",
            "artifacts": [
                {"filename": item.filename, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in PINNED_ARTIFACTS
            ],
        }


def prepare_legacy_features(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Match the released wrapper: select/order supplied values without fitting.

    The training notebook transformed these columns, but the released inference
    wrapper did not. Callers needing parity must therefore supply the already
    prepared legacy columns; this adapter must not silently recreate notebook
    preprocessing from canonical measurements.
    """
    return tuple(float(record[name]) for name in LEGACY_FEATURES)


def prepare_canonical_legacy_features(row: CanonicalRow) -> tuple[float, ...]:
    values = {
        "kb": row.metadata["legacy_kb_unit_unknown"],
        "volume": row.features["volume_mm3"],
        "surface_area": row.features["surface_area_mm2"],
        "bbox_area": row.features["bounding_box_volume_mm3"],
        "euler_number": row.features["euler_number"],
        "scale": row.metadata["legacy_scale_unit_unknown"],
        "surface_volume_ratio": row.metadata["legacy_surface_volume_ratio"],
    }
    return prepare_legacy_features(values)


def resolve_legacy_artifacts(
    directory: str | Path, *, download: bool = False
) -> ArtifactResolution:
    """Resolve only the pinned release, returning bounded blockers on failure."""
    root = Path(directory)
    paths: dict[str, Path] = {}
    blockers: list[str] = []
    for artifact in PINNED_ARTIFACTS:
        path = root / artifact.filename
        if not path.exists() and download:
            blocker = _download_artifact(path, artifact)
            if blocker:
                blockers.append(blocker)
        if not path.is_file():
            blockers.append("legacy_artifact_missing_" + artifact.filename.replace(".", "_"))
            continue
        try:
            content = path.read_bytes()
        except OSError:
            blockers.append("legacy_artifact_unreadable")
            continue
        if len(content) != artifact.size_bytes or sha256(content).hexdigest() != artifact.sha256:
            blockers.append("legacy_artifact_checksum_mismatch")
            continue
        paths[artifact.filename] = path
    if not blockers:
        try:
            metadata = json.loads(paths["minires_meta.json"].read_text())
            if metadata != {"w_nn": 0.2, "features": list(LEGACY_FEATURES)}:
                blockers.append("legacy_metadata_mismatch")
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append("legacy_metadata_unreadable")
    blockers = sorted(set(blockers))
    return ArtifactResolution("blocked" if blockers else "ready", tuple(blockers), paths if not blockers else {})


def load_legacy_reference(
    directory: str | Path,
    *,
    download: bool = False,
    provenance: LegacyProvenance | None = None,
) -> LegacyReference:
    """Load verified artifacts when optional inference dependencies are installed."""
    provenance = provenance or LegacyProvenance.unknown()
    resolution = resolve_legacy_artifacts(directory, download=download)
    if resolution.blockers:
        return LegacyReference.blocked(resolution.blockers, provenance)
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        from xgboost import XGBRegressor
    except ImportError:
        return LegacyReference.blocked(("legacy_inference_dependencies_required",), provenance)
    try:
        neural_network_model = load_model(resolution.paths["minires.keras"], compile=False)
        xgboost_model = XGBRegressor()
        xgboost_model.load_model(resolution.paths["minires_xgb.json"])
    except Exception:
        # Runtime loader messages may contain local cache paths; expose a code only.
        return LegacyReference.blocked(("legacy_artifact_load_failed",), provenance)

    def neural_network(rows: Sequence[tuple[float, ...]]) -> Sequence[float]:
        return _predict_keras_model(neural_network_model, np, rows)

    def xgboost(rows: Sequence[tuple[float, ...]]) -> Sequence[float]:
        return xgboost_model.predict(np.asarray(rows, dtype=np.float32)).reshape(-1).tolist()

    return LegacyReference.from_predictors(
        neural_network=neural_network,
        xgboost=xgboost,
        neural_network_weight=0.2,
        provenance=provenance,
    )


def _download_artifact(path: Path, artifact: PinnedArtifact) -> str | None:
    url = f"https://huggingface.co/{LEGACY_REPOSITORY}/resolve/{artifact.revision}/{artifact.filename}"
    temporary = path.with_name(path.name + ".partial")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as source, temporary.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        os.replace(temporary, path)
    except (OSError, URLError, TimeoutError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return "legacy_artifact_download_failed"
    return None
