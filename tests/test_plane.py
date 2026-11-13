import unittest
import numpy as np
from tomato_rgbd.plane import estimate_plane


class PlaneTests(unittest.TestCase):
    def test_known_plane_with_outliers(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(-0.3, 0.3, (1000, 2))
        z = 0.1 * xy[:, 0] - 0.05 * xy[:, 1] + 1.2 + rng.normal(0, 0.001, 1000)
        z[:150] += 0.1
        coeff, fraction, residual = estimate_plane(np.column_stack((xy, z)))
        np.testing.assert_allclose(coeff, [0.1, -0.05, 1.2], atol=0.005)
        self.assertGreater(fraction, 0.75)
        self.assertLess(residual, 0.003)

    def test_insufficient_points(self):
        self.assertIsNone(estimate_plane(np.zeros((99, 3))))
