import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from tomato_rgbd.baseline import extract_features

class FeatureTests(unittest.TestCase):
    def test_green_and_depth_units(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rgb=np.zeros((540,960,3),np.uint8);rgb[200:300,400:500]=[0,180,0]
            depth=np.full((540,960),1000,np.uint16)
            cv2.imwrite(str(p/'rgb.png'),rgb);cv2.imwrite(str(p/'depth.png'),depth)
            features,green,red=extract_features(p/'rgb.png',p/'depth.png')
            self.assertGreater(features['green_fraction'],0)
            self.assertAlmostEqual(features['depth_q50_m'],1.0,places=5)
            self.assertEqual(features['red_fraction_near_green'],0)
    def test_mismatched_depth_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cv2.imwrite(str(p/'rgb.png'),np.zeros((20,20,3),np.uint8));cv2.imwrite(str(p/'depth.png'),np.zeros((10,10),np.uint16))
            with self.assertRaises(ValueError):extract_features(p/'rgb.png',p/'depth.png')
