"""Evidence-limited record reconciliation; feature equality is not identity."""

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Sequence

from .ingestion import CanonicalRow, fingerprint

EVIDENCE_FIELDS = ("record_identity", "location_evidence", "source_name_evidence")


def reconcile(primary: Sequence[CanonicalRow], comparison: Sequence[CanonicalRow],
              input_fingerprint: str) -> dict[str, Any]:
    indices: list[dict[str, dict[str, list[int]]]] = []
    for rows in (primary, comparison):
        index: dict[str, dict[str, list[int]]] = {}
        for field in EVIDENCE_FIELDS:
            values: dict[str, list[int]] = defaultdict(list)
            for row in rows:
                token = row.metadata.get(field)
                if token is not None:
                    values[token].append(row.row_index)
            index[field] = values
        indices.append(index)
    # Gather all evidence before matching. Conflicting keys and repeated keys
    # stay ambiguous instead of being resolved by row order or weaker evidence.
    candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
    ambiguous: list[set[int]] = [set(), set()]
    for side, index in enumerate(indices):
        for values in index.values():
            for row_indices in values.values():
                if len(row_indices) > 1:
                    ambiguous[side].update(row_indices)
    for field in EVIDENCE_FIELDS:
        left, right = indices[0][field], indices[1][field]
        for token in left.keys() & right.keys():
            if len(left[token]) != 1 or len(right[token]) != 1:
                ambiguous[0].update(left[token])
                ambiguous[1].update(right[token])
            else:
                candidates[left[token][0], right[token][0]].append(field)
    degrees = [Counter(pair[side] for pair in candidates) for side in (0, 1)]
    for a, b in candidates:
        if degrees[0][a] > 1 or degrees[1][b] > 1:
            ambiguous[0].add(a)
            ambiguous[1].add(b)
        # Two available but different stable IDs contradict a weaker match.
        left_id = primary[a].metadata.get("record_identity")
        right_id = comparison[b].metadata.get("record_identity")
        if left_id and right_id and left_id != right_id:
            ambiguous[0].add(a)
            ambiguous[1].add(b)
    # Propagate ambiguity across candidate edges.
    changed = True
    while changed:
        before = len(ambiguous[0]) + len(ambiguous[1])
        for a, b in candidates:
            if a in ambiguous[0] or b in ambiguous[1]:
                ambiguous[0].add(a)
                ambiguous[1].add(b)
        changed = before != len(ambiguous[0]) + len(ambiguous[1])
    matches = []
    matched: list[set[int]] = [set(), set()]
    for (a, b), evidence in sorted(candidates.items()):
        if a in ambiguous[0] or b in ambiguous[1]:
            continue
        left_values = {**primary[a].features, "sliced_resin_mass_g": primary[a].sliced_resin_mass_g,
                       "outcome": primary[a].outcome, **primary[a].metadata}
        right_values = {**comparison[b].features, "sliced_resin_mass_g": comparison[b].sliced_resin_mass_g,
                        "outcome": comparison[b].outcome, **comparison[b].metadata}
        differences = {key: [left_values.get(key), right_values.get(key)]
                       for key in sorted(left_values.keys() | right_values.keys())
                       if left_values.get(key) != right_values.get(key)}
        matches.append({"primary_index": a, "comparison_index": b,
                        "evidence": evidence[0], "differences": differences})
        matched[0].add(a)
        matched[1].add(b)
    signatures = [Counter(fingerprint({**r.features, "target": r.sliced_resin_mass_g}) for r in rows)
                  for rows in (primary, comparison)]
    return {
        "input_fingerprint": input_fingerprint,
        "primary_count": len(primary), "comparison_count": len(comparison),
        "matched_count": len(matches), "matches": matches,
        "unmatched_primary_indices": sorted(set(range(len(primary))) - matched[0] - ambiguous[0]),
        "unmatched_comparison_indices": sorted(set(range(len(comparison))) - matched[1] - ambiguous[1]),
        "ambiguous_primary_indices": sorted(ambiguous[0]),
        "ambiguous_comparison_indices": sorted(ambiguous[1]),
        "equal_measurement_tuple_count": sum((signatures[0] & signatures[1]).values()),
        "dataset_equivalence_proven": False,
        "limitations": ["identity_keys_are_record_evidence_not_geometry_proof",
                        "equal_counts_or_features_do_not_prove_equivalence",
                        "unmatched_does_not_prove_absence"],
        "comparison_rows": [asdict(row) for row in comparison],
    }
