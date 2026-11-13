import importlib.util
from pathlib import Path
import tempfile, unittest
import cv2, numpy as np, torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rgb_train", ROOT / "scripts/train_resnet18_rgb.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RGBTests(unittest.TestCase):
    def test_rgb_resize_and_padding(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rgb.png"
            img = np.zeros((100, 200, 3), np.uint8)
            img[:] = [10, 20, 200]
            cv2.imwrite(str(p), img)
            x = m.make_image(p, 320)
            self.assertEqual(x.shape, (320, 320, 3))
            np.testing.assert_array_equal(x[160, 160], [200, 20, 10])
            self.assertFalse(x[0].any())

    def test_masked_loss_ignores_missing_label(self):
        pred = torch.zeros((2, 4))
        y = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 2.0, 3.0, 4.0]])
        mask = torch.ones((2, 4), dtype=torch.bool)
        mask[1, 0] = False
        result = m.loss_fn(pred, y, mask)
        y[1, 0] = 10000
        self.assertAlmostEqual(result.item(), m.loss_fn(pred, y, mask).item())
