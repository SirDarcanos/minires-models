"""Evidence-only grouping and create-only frozen source-holdout allocation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..ingestion import CanonicalRow, InputError, TRANSFORMATION_VERSION, fingerprint
from ..private_io import write_private_json

ALLOCATION_VERSION = 'source-family-holdout-v1'
LIMITATIONS = [
    'grouping_depends_on_supplied_evidence',
    'unreported_duplicates_and_variants_cannot_be_detected',
    'equal_features_are_not_duplicate_evidence',
    'no_mesh_validation',
    'no_fitting_or_model_selection',
]


def freeze_splits(rows: Sequence[CanonicalRow], input_fingerprint: str,
                  configuration: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Validate evidence, allocate whole components, and persist before scoring.

    A single unresolved included row or conflicting component blocks the run:
    sources must never disappear from an ostensibly complete rotation.
    """
    destination = Path(path)
    if 'private' not in destination.resolve().parts:
        raise InputError('private_split_manifest_required')
    included = [row for row in rows if row.outcome == 'included']
    parent = {row.row_index: row.row_index for row in rows}

    def root(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    # Use all rows, including unscorable rows, as evidence bridges.
    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        for key in ('miniature_family', 'duplicate_group', 'geometry_fingerprint', 'record_identity', 'location_evidence'):
            token = row.metadata.get(key)
            if token:
                evidence = (key, token)
                if evidence in seen:
                    left, right = root(row.row_index), root(seen[evidence])
                    parent[max(left, right)] = min(left, right)
                seen[evidence] = row.row_index
    components: dict[int, list[CanonicalRow]] = {}
    for row in rows:
        components.setdefault(root(row.row_index), []).append(row)
    blockers = set()
    source_aliases: dict[str, set[str]] = {}
    alias_origins: dict[str, set[str]] = {}
    for row in rows:
        origin = row.metadata.get('source_identity_evidence')
        alias = row.metadata.get('anonymous_source_group')
        if origin and alias:
            source_aliases.setdefault(origin, set()).add(alias)
            alias_origins.setdefault(alias, set()).add(origin)
    if any(len(values) > 1 for values in [*source_aliases.values(), *alias_origins.values()]):
        blockers.add('conflicting_source_evidence')
    groups: list[dict[str, Any]] = []
    for component in components.values():
        active = [row for row in component if row.outcome == 'included']
        component_sources = {row.metadata.get('anonymous_source_group') for row in component}
        if len(component_sources - {None}) > 1:
            blockers.add('conflicting_source_evidence')
        if not active:
            continue
        if None in component_sources:
            blockers.add('unresolved_source_group')
        if any(not row.metadata.get('miniature_family') for row in active):
            blockers.add('unresolved_miniature_family')
        groups.append({'component': fingerprint([row.row_index for row in component]),
                       'source': next(iter(component_sources)) if len(component_sources) == 1 else None,
                       'rows': [row.row_index for row in active]})
    sources = sorted({group['source'] for group in groups if group['source'] is not None})
    folds: list[dict[str, Any]] = []
    if not blockers:
        if len(sources) < 2:
            blockers.add('insufficient_source_groups')
        else:
            for source in sources:
                held_out = [group for group in groups if group['source'] == source]
                remaining = sorted([group for group in groups if group['source'] != source],
                                   key=lambda group: fingerprint([configuration['seed'], group['component']]))
                if len(remaining) < 2:
                    blockers.add('insufficient_inner_families')
                    continue
                folds.append({'source': source,
                              'test': sorted(i for group in held_out for i in group['rows']),
                              'validation': sorted(remaining[0]['rows']),
                              'train': sorted(i for group in remaining[1:] for i in group['rows'])})
    if blockers:
        folds = []
    for fold in folds:
        partitions = [set(fold[key]) for key in ('train', 'validation', 'test')]
        assert not any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        assert set.union(*partitions) == {row.row_index for row in included}
        for group in groups:
            assert sum(bool(set(group['rows']) & partition) for partition in partitions) == 1
        assert all(row.metadata['anonymous_source_group'] != fold['source']
                   for row in included if row.row_index in partitions[0] | partitions[1])
    manifest = {
        'allocation_version': ALLOCATION_VERSION,
        'transformation_version': TRANSFORMATION_VERSION,
        'input_fingerprint': input_fingerprint,
        'configuration': configuration,
        'allocation': {'outer': 'leave_one_source_out', 'inner': 'one_component_per_fold',
                       'minimum_sources': 2, 'minimum_non_holdout_components': 2},
        'status': 'blocked' if blockers else 'frozen_source_holdout',
        'blockers': sorted(blockers), 'limitations': LIMITATIONS,
        'eligible_source_count': len({row.metadata['anonymous_source_group'] for row in included
                                      if row.metadata.get('anonymous_source_group') is not None}),
        'included_rows': [row.row_index for row in included],
        'unscored_rows': [{'row_index': row.row_index, 'outcome': row.outcome,
                          'reasons': list(row.reasons)} for row in rows if row.outcome != 'included'],
        'groups': groups, 'folds': folds,
    }
    try:
        if destination.exists():
            if json.loads(destination.read_text()) != manifest:
                raise InputError('split_manifest_mismatch')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_private_json(destination, manifest)
    except (OSError, ValueError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError('split_manifest_unavailable') from None
    return manifest
