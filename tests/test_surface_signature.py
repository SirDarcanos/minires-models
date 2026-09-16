from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from minires.preparation import surface_signature


class SurfaceSignatureTests(unittest.TestCase):
    def test_signature_is_winding_translation_and_uniform_scale_invariant(self):
        mesh = trimesh.creation.box(extents=(2.0, 3.0, 5.0))
        expected = surface_signature.from_mesh(mesh)

        reversed_mesh = mesh.copy()
        reversed_mesh.invert()
        translated = mesh.copy()
        translated.apply_translation((17.0, -3.0, 41.0))
        scaled = mesh.copy()
        scaled.apply_scale(7.0)

        for transformed in (reversed_mesh, translated, scaled):
            with self.subTest(transformed=transformed):
                observed = surface_signature.from_mesh(transformed)
                np.testing.assert_allclose(
                    observed.surface_area_by_normalized_z,
                    expected.surface_area_by_normalized_z,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    observed.absolute_xy_projected_area_by_normalized_z,
                    expected.absolute_xy_projected_area_by_normalized_z,
                    rtol=1e-12,
                    atol=1e-12,
                )

        self.assertEqual(len(expected.surface_area_by_normalized_z), 32)
        self.assertAlmostEqual(sum(expected.surface_area_by_normalized_z), 1.0)
        self.assertAlmostEqual(
            sum(expected.absolute_xy_projected_area_by_normalized_z), 1.0
        )

    def test_signature_is_repeat_deterministic_and_chunk_invariant(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        first = surface_signature.from_mesh(mesh)
        second = surface_signature.from_mesh(mesh)
        with patch.object(surface_signature, "FACE_CHUNK_SIZE", 1):
            one_face_chunks = surface_signature.from_mesh(mesh)

        self.assertEqual(first, second)
        np.testing.assert_allclose(
            one_face_chunks.surface_area_by_normalized_z,
            first.surface_area_by_normalized_z,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            one_face_chunks.absolute_xy_projected_area_by_normalized_z,
            first.absolute_xy_projected_area_by_normalized_z,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_triangle_centroids_on_bin_boundaries_use_the_upper_bin(self):
        vertices = np.asarray([
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        ])
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8))),
            process=False,
        )

        signature = surface_signature.from_mesh(mesh)

        expected = np.zeros(32)
        expected[[0, 16, 31]] = 1.0 / 3.0
        np.testing.assert_allclose(signature.surface_area_by_normalized_z, expected)
        np.testing.assert_allclose(
            signature.absolute_xy_projected_area_by_normalized_z, expected
        )

    def test_signature_accepts_open_and_overlapping_shells_as_additive_surface_data(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        shifted = box.copy()
        shifted.apply_translation((0.0, 0.0, 0.5))
        overlapping = trimesh.util.concatenate((box, shifted))
        open_mesh = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=box.faces[:-1].copy(), process=False
        )

        overlap_signature = surface_signature.from_mesh(overlapping)
        open_signature = surface_signature.from_mesh(open_mesh)

        self.assertFalse(open_mesh.is_watertight)
        self.assertTrue(all(value >= 0.0 for value in open_signature.surface_area_by_normalized_z))
        self.assertTrue(all(
            value >= 0.0
            for value in overlap_signature.absolute_xy_projected_area_by_normalized_z
        ))
        self.assertEqual(overlap_signature.semantics, "additive_triangle_surface_not_occupied_volume")

    def test_stl_file_seam_loads_without_processing_or_repair(self):
        mesh = trimesh.creation.box(extents=(2.0, 3.0, 5.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.stl"
            mesh.export(path)
            observed = surface_signature.extract(path)

        expected = surface_signature.from_mesh(mesh)
        np.testing.assert_allclose(
            observed.surface_area_by_normalized_z,
            expected.surface_area_by_normalized_z,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(observed.version, surface_signature.VERSION)

    def test_signature_rejects_degenerate_or_nonfinite_geometry(self):
        invalid_cases = (
            trimesh.Trimesh(vertices=[], faces=[], process=False),
            trimesh.Trimesh(
                vertices=np.asarray(((0.0, 0.0, 0.0),) * 3),
                faces=np.asarray(((0, 1, 2),)),
                process=False,
            ),
            trimesh.Trimesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                     (0.0, float("nan"), 1.0))),
                faces=np.asarray(((0, 1, 2),)),
                process=False,
            ),
            trimesh.Trimesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                     (0.0, 0.0, 1.0))),
                faces=np.asarray(((0, 1, 2),)),
                process=False,
            ),
        )
        for mesh in invalid_cases:
            with self.subTest(mesh=mesh), self.assertRaisesRegex(
                ValueError, "invalid_surface_signature_geometry"
            ):
                surface_signature.from_mesh(mesh)


if __name__ == "__main__":
    unittest.main()
