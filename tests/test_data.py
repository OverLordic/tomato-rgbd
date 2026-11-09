"""Synthetic RGB-D files exercise joins, bit depth, missingness and duplicate detection."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tomato_rgbd.data import (MANIFEST_COLUMNS, SUBSETS, audit_images, build_manifest,
                              load_development_manifest, read_csv, read_json, write_csv)
from tomato_rgbd.splits import image_fingerprint


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data/raw/single_plant_cam"
        self.calibration = {"color_int": {"width": 8, "height": 6}}
        for subset_index, subset in enumerate(SUBSETS):
            folder = self.dataset / subset
            (folder / "rgb").mkdir(parents=True)
            (folder / "depth").mkdir()
            truth, names = {}, {}
            for offset in range(2):
                n = subset_index * 2 + offset + 1
                plant_id = f"{'AB'[offset]}_{n:08x}"
                truth[plant_id] = {"height": 20., "fw_plant": 100., "leaf_area": 500., "number_of_red_fruits": 0.}
                names[plant_id] = f"camir_27_2023_11_23_11_23_{n:02d}_nc6l{n}_rgb"
                Image.fromarray(np.full((6, 8, 3), n, dtype=np.uint8)).save(folder / "rgb" / f"{plant_id}.png")
                depth = np.full((6, 8), 650 + n, dtype=np.uint16)
                depth[0, 0] = 0
                Image.fromarray(depth).save(folder / "depth" / f"{plant_id}_depth.png")
                for side in ("L", "R"):
                    Image.fromarray(np.full((3, 4), n, dtype=np.uint8)).save(folder / "depth" / f"{plant_id}_ir{side}.png")
            (folder / f"ground_truth_{subset}.json").write_text(json.dumps(truth))
            (folder / f"anon_keys_{subset}.json").write_text(json.dumps(names))

    def manifest(self):
        return build_manifest(self.dataset, self.root, expected_plants=10)

    def test_one_row_per_plant_and_relative_paths(self):
        rows = self.manifest()
        self.assertEqual(len(rows), 10)
        self.assertEqual(list(rows[0]), list(MANIFEST_COLUMNS))
        self.assertTrue(all(not Path(r["rgb_path"]).is_absolute() for r in rows))
        self.assertEqual(rows[0]["imaging_date"], "2023-11-23")

    def test_missing_file_is_not_silently_skipped(self):
        p = self.dataset / "train/depth/A_00000001_depth.png"
        p.unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            self.manifest()

    def test_orphan_image_is_detected(self):
        Image.new("RGB", (8, 6)).save(self.dataset / "train/rgb/orphan.png")
        with self.assertRaisesRegex(ValueError, "Orphan RGB"):
            self.manifest()

    def test_nan_label_remains_missing_and_csv_uses_blank(self):
        p = self.dataset / "train/ground_truth_train.json"
        obj = read_json(p)
        obj["A_00000001"]["height"] = float("nan")
        p.write_text(json.dumps(obj))
        rows = self.manifest()
        self.assertTrue(math.isnan(rows[0]["height"]))
        out = self.root / "manifest.csv"
        write_csv(out, rows, MANIFEST_COLUMNS)
        self.assertEqual(read_csv(out)[0]["height"], "")
        self.assertEqual(read_csv(out)[0]["number_of_red_fruits"], "0.0")

    def test_fractional_fruit_count_is_rejected(self):
        p = self.dataset / "train/ground_truth_train.json"
        obj = read_json(p)
        obj["A_00000001"]["number_of_red_fruits"] = 1.5
        p.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "Non-integer"):
            self.manifest()

    def test_duplicate_json_key_is_rejected(self):
        p = self.root / "duplicate.json"
        p.write_text('{"plant": 1, "plant": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            read_json(p)

    def test_depth_keeps_16_bits_zero_is_excluded_from_quantiles(self):
        rows = self.manifest()
        audit = audit_images(rows, self.root, self.calibration, progress=None)
        self.assertEqual(audit[0]["depth_dtype"], "uint16")
        self.assertEqual(audit[0]["depth_png_bits"], 16)
        self.assertAlmostEqual(audit[0]["zero_depth_fraction"], 1 / 48)
        self.assertAlmostEqual(audit[0]["sample_depth_median_m"], .651)

    def test_exact_duplicate_rgb_is_detected_before_splitting(self):
        rows = self.manifest()
        (self.root / rows[1]["rgb_path"]).write_bytes((self.root / rows[0]["rgb_path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate RGB"):
            audit_images(rows, self.root, self.calibration, progress=None)

    def test_wrong_depth_shape_is_rejected(self):
        rows = self.manifest()
        Image.fromarray(np.zeros((2, 3), dtype=np.uint16)).save(self.root / rows[0]["depth_path"])
        with self.assertRaisesRegex(ValueError, "depth format"):
            audit_images(rows, self.root, self.calibration, progress=None)

    def test_ir_and_calibration_changes_affect_fingerprint(self):
        rows = self.manifest()
        audit = audit_images(rows, self.root, self.calibration, progress=None)
        first = image_fingerprint(audit, "calibration-1")
        self.assertNotEqual(first, image_fingerprint(audit, "calibration-2"))
        audit[0]["ir_left_pixels_sha256"] = "modified"
        self.assertNotEqual(first, image_fingerprint(audit, "calibration-1"))

    def test_training_loader_excludes_holdout_and_restores_numbers(self):
        rows = self.manifest()
        write_csv(self.root / "data/processed/manifest.csv", rows, MANIFEST_COLUMNS)
        write_csv(self.root / "splits/cv_folds.csv",
                  [{"plant_id": rows[0]["plant_id"], "variety": rows[0]["variety"], "fold": 0}],
                  ("plant_id", "variety", "fold"))
        loaded = load_development_manifest(self.root)
        self.assertEqual([r["plant_id"] for r in loaded], [rows[0]["plant_id"]])
        self.assertIsInstance(loaded[0]["height"], float)


if __name__ == "__main__":
    unittest.main()
