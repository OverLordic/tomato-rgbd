"""Plant isolation, deterministic stratification and immutable saved splits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tomato_rgbd.splits import freeze_splits, validate_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"plant_id": f"{v}_{i:08x}", "variety": v, "height": object()}
                     for v, n in {"A": 48, "B": 144, "C": 143, "D": 144}.items()
                     for i in range(n)]
        self.protocol = {"seed": 42, "holdout": {"fraction": .2, "stratify_by": "variety"},
                         "cross_validation": {"n_splits": 5, "shuffle": True, "stratify_by": "variety"}}

    def freeze(self, directory=None, **overrides):
        args = {"rows": self.rows, "directory": directory or self.root,
                "manifest_sha": "manifest-sha", "images_sha": "images-sha", "protocol": self.protocol}
        args.update(overrides)
        return freeze_splits(**args)

    def test_479_plants_are_383_development_and_96_holdout(self):
        holdout, folds, metadata = self.freeze()
        self.assertEqual((len(holdout), len(folds)), (96, 383))
        self.assertEqual(metadata["holdout_evaluations"], 0)
        validate_splits(self.rows, holdout, folds, 5)
        for f in range(5):
            train = {r["plant_id"] for r in folds if r["fold"] != f}
            val = {r["plant_id"] for r in folds if r["fold"] == f}
            test = {r["plant_id"] for r in holdout}
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(len(train | val | test), 479)
        for v in "ABCD":
            actual = sum(r["variety"] == v for r in holdout)
            expected = .2 * sum(r["variety"] == v for r in self.rows)
            self.assertLessEqual(abs(actual - expected), 1)

    def test_input_order_does_not_change_saved_ids(self):
        self.freeze(self.root / "one")
        self.freeze(self.root / "two", rows=list(reversed(self.rows)))
        for name in ("holdout.csv", "cv_folds.csv"):
            self.assertEqual((self.root / "one" / name).read_bytes(),
                             (self.root / "two" / name).read_bytes())

    def test_saved_ids_are_reused_without_calling_random_split(self):
        self.freeze()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with patch("tomato_rgbd.splits.train_test_split", side_effect=AssertionError("must reuse")):
            self.freeze()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_new_seed_or_data_fingerprint_cannot_overwrite_splits(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.freeze(images_sha="modified-image")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.freeze(manifest_sha="modified-manifest")
        self.protocol["seed"] = 43
        with self.assertRaisesRegex(ValueError, "changed"):
            self.freeze()

    def test_manually_edited_saved_csv_is_detected(self):
        self.freeze()
        p = self.root / "holdout.csv"
        p.write_text(p.read_text() + "A_wrong,A\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.freeze()

    def test_partial_saved_split_is_not_replaced(self):
        (self.root / "holdout.csv").write_text("plant_id,variety\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.freeze()

    def test_duplicate_plant_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.freeze(rows=self.rows + [self.rows[0]])

    def test_wrong_fold_or_overlap_is_rejected(self):
        holdout, folds, _ = self.freeze()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_splits(self.rows, holdout + [folds[0]], folds, 5)
        folds[0]["fold"] = 10
        with self.assertRaisesRegex(ValueError, "fold numbers"):
            validate_splits(self.rows, holdout, folds, 5)


if __name__ == "__main__":
    unittest.main()
