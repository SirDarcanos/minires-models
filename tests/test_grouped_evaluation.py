import json
import contextlib
import io

from minires_evaluation.__main__ import main
from pathlib import Path
import tempfile
import unittest

from minires_evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records
from minires_evaluation.ingestion import InputError


class GroupedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / 'private' / 'splits.json'
        self.config = EvaluationConfig(1.0, 'mm3', True, seed=17)
        self.rows = [dict(volume=1000, weight=weight, anonymous_source_group=source,
                          miniature_family=family)
                     for source, family, weight in [('a', 'a1', 2), ('a', 'a2', 4),
                                                    ('b', 'b1', 1), ('c', 'c1', 3)]]

    def evaluate(self, rows=None):
        return evaluate_records(self.rows if rows is None else rows, self.config,
                                PhysicalBaseline(), split_manifest=self.manifest)

    def test_freezes_and_reuses_isolated_source_folds(self):
        result = self.evaluate()
        self.assertEqual(result.split_status, 'frozen_source_holdout')
        before = self.manifest.read_bytes()
        manifest = json.loads(before)
        self.assertEqual(len(manifest['folds']), 3)
        held_out = []
        for fold in manifest['folds']:
            partitions = [fold[key] for key in ('train', 'validation', 'test')]
            self.assertTrue(all(partitions))
            self.assertEqual(sorted(sum(partitions, [])), [0, 1, 2, 3])
            source_sets = [{self.rows[i]['anonymous_source_group'] for i in part} for part in partitions]
            self.assertFalse(source_sets[2] & (source_sets[0] | source_sets[1]))
            family_sets = [{self.rows[i]['miniature_family'] for i in part} for part in partitions]
            self.assertFalse(family_sets[0] & family_sets[1])
            held_out.extend(fold['test'])
        self.assertEqual(sorted(held_out), [0, 1, 2, 3])
        self.assertEqual(self.evaluate().predictions, result.predictions)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(result.metrics.sample_count, 4)
        self.assertNotIn('anonymous_source_group', result.canonical_rows[0].features)

    def test_reports_known_pooled_and_source_balanced_metrics_privately(self):
        result = self.evaluate()
        report = result.to_dict(public=True)['grouped_evaluation']
        self.assertAlmostEqual(result.metrics.mae_g, 1.5)
        balanced = report['source_balanced']
        self.assertAlmostEqual(balanced['mae_g'], 4 / 3)
        self.assertAlmostEqual(balanced['rmse_g'], 3 ** 0.5)
        self.assertAlmostEqual(balanced['underestimation_fraction'], 2 / 3)
        self.assertAlmostEqual(balanced['mean_underestimation_g'], 2)
        self.assertEqual(report['source_count'], 3)
        self.assertEqual(report['sample_count'], 4)
        self.assertNotIn('manifest', report)
        self.assertNotIn('source_reports', report)
        self.assertEqual(len(result.to_dict()['grouped_evaluation']['source_reports']), 3)

    def test_blocks_missing_conflicting_and_insufficient_evidence(self):
        cases = [
            ([dict(row, miniature_family=None) for row in self.rows], 'unresolved_miniature_family'),
            ([dict(row, anonymous_source_group=None) for row in self.rows], 'unresolved_source_group'),
            ([dict(row, duplicate_group='shared') for row in self.rows], 'conflicting_source_evidence'),
            (self.rows[:2], 'insufficient_source_groups'),
            (self.rows[2:], 'insufficient_inner_families'),
        ]
        for index, (rows, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                self.manifest = self.manifest.with_name(f'blocked-{index}.json')
                result = self.evaluate(rows)
                self.assertEqual(result.status, 'blocked')
                self.assertIn(reason, result.blockers)
                self.assertEqual(result.predictions, ())
                self.assertEqual(result.metrics.sample_count, 0)
                self.assertEqual(json.loads(self.manifest.read_text())['folds'], [])

    def test_rejects_changed_input_configuration_or_tampered_manifest(self):
        self.evaluate()
        original = self.manifest.read_bytes()
        with self.assertRaisesRegex(InputError, 'split_manifest_mismatch'):
            self.evaluate([dict(row, weight=9) for row in self.rows])
        self.assertEqual(self.manifest.read_bytes(), original)
        self.config = EvaluationConfig(1.0, 'mm3', True, seed=18)
        with self.assertRaisesRegex(InputError, 'split_manifest_mismatch'):
            self.evaluate()
        self.config = EvaluationConfig(1.0, 'mm3', True, seed=17)
        data = json.loads(original)
        data['folds'][0]['test'] = []
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(InputError, 'split_manifest_mismatch'):
            self.evaluate()

    def test_duplicate_variants_merge_and_equal_features_do_not(self):
        rows = self.rows + [dict(self.rows[0], miniature_family='variant', duplicate_group='d')]
        rows[0] = dict(rows[0], duplicate_group='d')
        result = self.evaluate(rows)
        manifest = result.to_dict()['grouped_evaluation']['manifest']
        self.assertEqual(len(manifest['groups']), 4)
        self.assertIn([0, 4], [group['rows'] for group in manifest['groups']])
        for fold in manifest['folds']:
            for key in ('train', 'validation', 'test'):
                self.assertEqual(0 in fold[key], 4 in fold[key])

    def test_accepts_private_source_mapping_without_equating_names_to_aliases(self):
        result = self.evaluate([dict(row, artist='origin-' + row['anonymous_source_group']) for row in self.rows])
        self.assertEqual(result.status, 'completed')

    def test_fresh_manifest_assignments_ignore_labels_and_reproduce_exactly(self):
        first = self.evaluate().to_dict()['grouped_evaluation']['manifest']
        self.manifest = self.manifest.with_name('fresh.json')
        second = self.evaluate().to_dict()['grouped_evaluation']['manifest']
        self.assertEqual(first, second)
        self.manifest = self.manifest.with_name('different-labels.json')
        changed = self.evaluate([dict(row, weight=100) for row in self.rows])
        self.assertEqual(first['folds'], changed.to_dict()['grouped_evaluation']['manifest']['folds'])

    def test_unscorable_duplicate_bridge_preserves_conflicts(self):
        rows = [dict(row) for row in self.rows]
        rows[0]['duplicate_group'] = 'bridge'
        rows.append(dict(rows[2], weight=None, duplicate_group='bridge'))
        result = self.evaluate(rows)
        self.assertEqual(result.status, 'blocked')
        self.assertIn('conflicting_source_evidence', result.blockers)
        self.assertEqual(result.data_quality.needs_review_count, 1)

    def test_conflicts_in_unscorable_components_still_block(self):
        rows = self.rows + [dict(volume=1000, weight=None, anonymous_source_group=source,
                                 miniature_family='unscorable', duplicate_group='conflict')
                            for source in ('d', 'e')]
        result = self.evaluate(rows)
        self.assertEqual(result.status, 'blocked')
        self.assertIn('conflicting_source_evidence', result.blockers)
        self.assertEqual(result.predictions, ())

    def test_blocked_components_preserve_eligible_source_count(self):
        result = self.evaluate([dict(row, duplicate_group='conflict') for row in self.rows])
        summary = result.to_dict(public=True)['grouped_evaluation']
        self.assertEqual(summary['eligible_source_count'], 3)
        self.assertEqual(summary['source_count'], 0)
        self.assertEqual(summary['unscored_input_count'], 4)

    def test_conflicting_source_alias_mapping_blocks_rotation(self):
        rows = [dict(row, artist='same-origin') for row in self.rows]
        self.assertIn('conflicting_source_evidence', self.evaluate(rows).blockers)

    def test_grouped_artifacts_retain_evidence_separately_from_features(self):
        import pyarrow.parquet as pq
        output = Path(self.temp.name) / 'private' / 'run'
        rows = [dict(row, geometry_fingerprint=f'mesh-{i}') for i, row in enumerate(self.rows)]
        result = evaluate_records(rows, self.config, PhysicalBaseline(),
                                  split_manifest=self.manifest, output_dir=output)
        features = pq.read_table(output / 'features.parquet')
        metadata = pq.read_table(output / 'evaluation_metadata.parquet')
        self.assertNotIn('geometry_fingerprint', features.column_names)
        self.assertTrue(all(metadata.column('geometry_fingerprint').to_pylist()))
        private_report = json.loads((output / 'report.json').read_text())
        self.assertEqual(private_report['grouped_evaluation']['source_count'], 3)
        self.assertNotIn('no_held_out_evidence', json.loads((output / 'manifest.json').read_text())['limitations'])
        for row in result.canonical_rows:
            for key in ('anonymous_source_group', 'miniature_family', 'geometry_fingerprint'):
                self.assertNotIn(row.metadata[key], json.dumps(result.to_dict(public=True)))

    def test_cli_persists_private_manifest_and_publishes_aggregates_only(self):
        records = Path(self.temp.name) / 'records.json'
        records.write_text(json.dumps(self.rows))
        original = records.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(['--records', str(records), '--density-g-per-ml', '1',
                         '--volume-unit', 'mm3', '--scope-confirmed', '--public',
                         '--split-manifest', str(self.manifest)])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['split_status'], 'frozen_source_holdout')
        self.assertNotIn(str(self.manifest), output.getvalue())
        self.assertNotIn('miniature_family', output.getvalue())
        self.assertNotIn('anonymous_source_group', output.getvalue())
        self.assertEqual(self.manifest.stat().st_mode & 0o777, 0o600)
        self.assertEqual(records.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
