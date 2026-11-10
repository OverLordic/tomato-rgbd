import importlib.util,unittest,tempfile
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.models import resnet18
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rgbd_train',ROOT/'scripts/train_resnet18_rgbd.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RGBDTests(unittest.TestCase):
 def test_initial_output_equal_and_only_depth_weights_update(self):
  model=resnet18(weights=None);old=model.conv1.weight.detach().clone();m.add_depth_channel(model)
  x=torch.randn(2,4,64,64);x[:,3]=1
  reference=torch.nn.functional.conv2d(x[:,:3],old,stride=2,padding=3)
  torch.testing.assert_close(model.conv1(x),reference)
  model.conv1(x).sum().backward()
  self.assertEqual(float(model.conv1.weight.grad[:,:3].abs().sum()),0)
  self.assertGreater(float(model.conv1.weight.grad[:,3].abs().sum()),0)
 def test_depth_bit_depth_and_nearest_resize(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'depth.png';a=np.full((100,200),1500,np.uint16);a[:,100:]=0;cv2.imwrite(str(p),a)
   out=m.make_depth(p,320);self.assertEqual(out.dtype,np.uint16);self.assertEqual(set(np.unique(out)),{0,1500})
