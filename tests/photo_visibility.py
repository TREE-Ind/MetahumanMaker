"""Photo visibility must distinguish steep surfaces from hidden cavities."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'create_metahuman'))
from photo_color_core import surface_visibility, feather_projection


class PhotoVisibility(unittest.TestCase):
    def test_fade_preserves_detail_and_does_not_cross_uv_gutters(self):
        rng = np.random.default_rng(7)
        output = rng.random((24, 32, 4), dtype=np.float32)
        fallback = np.full_like(output, .2)
        coverage = np.zeros((24, 32), dtype=bool)
        coverage[2:22, 2:15] = True
        coverage[2:22, 16:30] = True
        confidence = coverage.astype(np.float32)
        confidence[8:16, 12:15] = 0
        output[confidence == 0] = fallback[confidence == 0]
        result, softened, report = feather_projection(output, fallback, confidence, coverage, 4)
        self.assertGreater(report['softened_texels'], 0)
        self.assertTrue(np.all(softened <= confidence))
        # The unsupported edge is one gutter pixel away, but cannot affect
        # the other island. Deep trusted texels retain every color bit.
        np.testing.assert_array_equal(result[:, 16:], output[:, 16:])
        np.testing.assert_array_equal(result[:, :7], output[:, :7])
        np.testing.assert_array_equal(result[..., 3], output[..., 3])
        np.testing.assert_array_equal(result[~coverage], output[~coverage])
        self.assertTrue(np.any((softened > 0) & (softened < 1)))
        # Anatomical cleanup must leave other regions byte-exact, even when
        # they have nearby unsupported pixels of their own.
        region = np.zeros_like(confidence)
        region[10:14, 9:13] = .6
        localized, weights, _ = feather_projection(
            output, fallback, confidence, coverage, 12, region=region)
        np.testing.assert_array_equal(localized[region == 0], output[region == 0])
        np.testing.assert_array_equal(weights[region == 0], confidence[region == 0])
        np.testing.assert_array_equal(localized[..., 3], output[..., 3])
        self.assertTrue(np.any(localized != output))

    def test_grazing_surface_is_visible_but_nearby_occluder_is_not(self):
        width, height, scale = 96, 64, .2
        normal = np.array([.8, .5, .07])
        projected = np.array([[.315, .437], [.451, .533], [.697, .612]])
        y, x = np.mgrid[:height, :width]
        plane = -((x + .5 - width / 2) * normal[0] +
                  (y + .5 - height / 2) * normal[1]) * scale / width / normal[2]
        depth = -((projected - .5) * [width, height]) @ normal[:2] * scale / width / normal[2]
        np.testing.assert_allclose(surface_visibility(projected, depth, normal, plane, scale), 1, atol=1e-6)
        # Less than one photo pixel of depth separation must still occlude,
        # even though the receiving surface is nearly edge-on.
        occluded = plane + scale / width * .7
        np.testing.assert_array_equal(surface_visibility(projected, depth, normal, occluded, scale), 0)

    def test_subpixel_occlusion_boundary_has_fractional_coverage(self):
        zbuffer = np.zeros((32, 32))
        zbuffer[:, 16:] = .05
        positions = np.array([[15.5 / 32, .5], [16 / 32, .5], [16.5 / 32, .5]])
        coverage = surface_visibility(positions, np.zeros(3), np.array([0., 0., 1.]), zbuffer, 1.)
        np.testing.assert_allclose(coverage, [1, .5, 0], atol=1e-6)
        zbuffer[:] = -np.inf
        np.testing.assert_array_equal(surface_visibility(positions, np.zeros(3), np.array([0., 0., 1.]), zbuffer, 1.), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
