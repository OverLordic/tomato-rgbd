import unittest,tempfile
from pathlib import Path
import cv2,numpy as np
from tomato_rgbd.depth_baseline import extract_features
class DepthMaskTests(unittest.TestCase):
 def test_depth_foreground_and_invalid_pixels(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);rgb=np.zeros((540,960,3),np.uint8);rgb[95:475,200:760]=[0,140,255];rgb[200:300,400:500]=[30,30,30]
   depth=np.full((540,960),1000,np.uint16);depth[200:300,400:500]=800;depth[210:220,410:420]=0
   cv2.imwrite(str(p/'rgb.png'),rgb);cv2.imwrite(str(p/'depth.png'),depth)
   features,mask,_=extract_features(p/'rgb.png',p/'depth.png')
   self.assertEqual(mask[250,450],1);self.assertEqual(mask[215,415],0);self.assertAlmostEqual(features['depth_q50_m'],.8,places=5)
