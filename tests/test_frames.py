"""Verify that saved fold IDs produce separate pandas tables."""

from pathlib import Path
import tempfile
import unittest
import pandas as pd
from tomato_rgbd.data import MANIFEST_COLUMNS, write_csv, load_fold_dataframes


class FoldFrameTests(unittest.TestCase):
    def test_train_validation_and_holdout_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(6):
                row = dict.fromkeys(MANIFEST_COLUMNS, "")
                row.update(
                    plant_id=f"A_{i:08x}",
                    variety="A",
                    height=10.0,
                    fw_plant=20.0,
                    leaf_area=30.0,
                    number_of_red_fruits=0.0,
                )
                rows.append(row)
            write_csv(root / "data/processed/manifest.csv", rows, MANIFEST_COLUMNS)
            write_csv(
                root / "splits/cv_folds.csv",
                [
                    dict(plant_id=r["plant_id"], variety="A", fold=i)
                    for i, r in enumerate(rows[:5])
                ],
                ("plant_id", "variety", "fold"),
            )
            write_csv(
                root / "splits/holdout.csv",
                [dict(plant_id=rows[5]["plant_id"], variety="A")],
                ("plant_id", "variety"),
            )
            train, val = load_fold_dataframes(root, 0)
            self.assertIsInstance(train, pd.DataFrame)
            self.assertEqual((len(train), len(val)), (4, 1))
            self.assertNotIn(
                rows[5]["plant_id"], set(train.plant_id) | set(val.plant_id)
            )
            self.assertFalse(set(train.plant_id) & set(val.plant_id))
            with self.assertRaises(ValueError):
                load_fold_dataframes(root, 5)
