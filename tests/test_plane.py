import unittest
import numpy as np
from tomato_rgbd.plane import estimate_plane
class PlaneTests(unittest.TestCase):
 def test_known_plane_with_outliers(self):
  rng=np.random.default_rng(42);xy=rng.uniform(-.3,.3,(1000,2));z=.1*xy[:,0]-.05*xy[:,1]+1.2+rng.normal(0,.001,1000);z[:150]+=.1
  coeff,fraction,residual=estimate_plane(np.column_stack((xy,z)))
  np.testing.assert_allclose(coeff,[.1,-.05,1.2],atol=.005)
  self.assertGreater(fraction,.75);self.assertLess(residual,.003)
 def test_insufficient_points(self):
  self.assertIsNone(estimate_plane(np.zeros((99,3))))
