import unittest,numpy as np
from tomato_rgbd.geometry import backproject,grid_features
class GeometryTests(unittest.TestCase):
 def test_camera_center_and_invalid(self):
  cal={'color_int':{'fx':100,'fy':100,'ppx':1,'ppy':1}}
  d=np.array([[0,1000,1000],[1000,1000,1000],[1000,1000,1000]],np.uint16)
  xyz,valid=backproject(d,cal);np.testing.assert_allclose(xyz[1,1],[0,0,1]);np.testing.assert_allclose(xyz[1,2],[.01,0,1]);self.assertFalse(valid[0,0])
 def test_flat_plane_area_and_volume(self):
  x,y=np.meshgrid(np.arange(10)*.01+.001,np.arange(10)*.01+.001)
  pts=np.column_stack((x.ravel(),y.ravel(),np.full(100,.9)));area,h95,h99,volume=grid_features(pts,[0,0,1],cell=.01)
  self.assertAlmostEqual(area,100);self.assertAlmostEqual(h95,10);self.assertAlmostEqual(volume,1000)
 def test_too_few_points(self):
  self.assertTrue(np.isnan(grid_features(np.zeros((3,3)),[0,0,1])[0]))
