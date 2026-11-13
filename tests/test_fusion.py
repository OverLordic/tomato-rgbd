import importlib.util, unittest
from pathlib import Path
import torch
from unittest.mock import patch
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fusion", ROOT / "scripts/train_fusion.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FusionTests(unittest.TestCase):
    def test_both_branches_receive_gradients(self):
        with patch.object(
            m.models, "resnet18", side_effect=lambda weights: resnet18(weights=None)
        ):
            model = m.FusionNet()
            x = torch.randn(2, 4, 64, 64)
            model(x).sum().backward()
        self.assertEqual(model(x).shape, (2, 4))
        self.assertGreater(float(model.depth[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.rgb.layer4[0].conv1.weight.grad.abs().sum()), 0)
