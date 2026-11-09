"""Check the competition formula, ID alignment and missing-label policy."""

import unittest

import numpy as np

from tomato_rgbd.metrics import TARGET_NAMES, rmsre, score_predictions


class RMSRETests(unittest.TestCase):
    def test_perfect_prediction_including_zero(self):
        truth = np.array([[0, 0, 0, 0], [10, 100, 1000, 3]], dtype=float)
        self.assertEqual(rmsre(truth, truth).mean, 0.0)

    def test_hand_calculated_fruit_example(self):
        truth = np.zeros((2, 4))
        truth[:, 3] = [0, 3]
        pred = truth.copy()
        pred[:, 3] = [1, 5]
        result = rmsre(truth, pred)
        self.assertAlmostEqual(result.per_target["number_of_red_fruits"], np.sqrt(0.625))
        self.assertAlmostEqual(result.mean, np.sqrt(0.625) / 4)

    def test_macro_mean_is_not_pooled_root(self):
        truth = np.zeros((2, 4))
        pred = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        result = rmsre(truth, pred)
        self.assertEqual(list(result.per_target.values()), [1, 2, 3, 4])
        self.assertEqual(result.mean, 2.5)
        self.assertNotAlmostEqual(result.mean, np.sqrt(np.mean(pred**2)))

    def test_missing_labels_are_masked_per_target(self):
        truth = [[0, np.nan, 3, 0], [np.nan, 1, 3, 0]]
        pred = [[1, 99, 5, 1], [123, 2, 7, 0]]
        result = rmsre(truth, pred)
        np.testing.assert_allclose(
            list(result.per_target.values()), [1, 0.5, np.sqrt(0.625), np.sqrt(0.5)]
        )
        self.assertEqual(list(result.n_valid.values()), [1, 1, 2, 2])

    def test_matches_official_example_implementation(self):
        rng = np.random.default_rng(42)
        truth = rng.uniform(0, 100, size=(19, 4))
        pred = truth + rng.normal(0, 10, size=truth.shape)
        truth[::3, 2] = np.nan
        # Same calculation order as the official PDF's dictionary example.
        expected = 0
        for column in range(4):
            diff = [
                ((pred[i, column] - truth[i, column]) / (truth[i, column] + 1)) ** 2
                for i in range(len(truth))
            ]
            expected += np.sqrt(np.nanmean(diff))
        self.assertAlmostEqual(rmsre(truth, pred).mean, expected / 4, places=14)

    def test_nonfinite_prediction_is_rejected_even_for_missing_label(self):
        truth = np.zeros((2, 4))
        truth[0, 0] = np.nan
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                pred = np.zeros((2, 4))
                pred[0, 0] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    rmsre(truth, pred)

    def test_no_observed_labels_for_target_is_rejected(self):
        truth = np.zeros((2, 4))
        truth[:, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "leaf_area"):
            rmsre(truth, np.zeros((2, 4)))

    def test_negative_or_infinite_truth_is_rejected(self):
        for invalid in (-1, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                truth = np.zeros((1, 4))
                truth[0, 0] = invalid
                with self.assertRaisesRegex(ValueError, "Ground truth"):
                    rmsre(truth, np.zeros((1, 4)))

    def test_shape_and_empty_input_are_rejected(self):
        for truth, pred in [
            (np.zeros(4), np.zeros(4)),
            (np.zeros((2, 3)), np.zeros((2, 3))),
            (np.zeros((2, 4)), np.zeros((1, 4))),
            (np.empty((0, 4)), np.empty((0, 4))),
        ]:
            with self.subTest(shape=np.shape(truth)):
                with self.assertRaises(ValueError):
                    rmsre(truth, pred)

    def test_predictions_are_not_clipped(self):
        self.assertEqual(rmsre([[0, 0, 0, 0]], [[-1, 0, 0, 0]]).mean, 0.25)

    def test_units_matter_because_of_plus_one(self):
        truth = np.array([[20, 0, 0, 0]], dtype=float)
        pred = np.array([[22, 0, 0, 0]], dtype=float)
        score_in_cm = rmsre(truth, pred).mean
        truth[:, 0] /= 100
        pred[:, 0] /= 100
        self.assertNotAlmostEqual(score_in_cm, rmsre(truth, pred).mean)

    def test_inputs_are_not_modified(self):
        truth = np.array([[np.nan, 2, 3, 0], [1, 2, 3, 0]], dtype=float)
        pred = np.zeros((2, 4))
        truth_before, pred_before = truth.copy(), pred.copy()
        rmsre(truth, pred)
        np.testing.assert_array_equal(truth, truth_before)
        np.testing.assert_array_equal(pred, pred_before)

    def test_dictionary_aligns_plant_and_target_names(self):
        truth = {
            "A_one": dict(zip(TARGET_NAMES, [10, 100, 500, 0])),
            "B_two": dict(zip(TARGET_NAMES, [20, 200, 900, 3])),
        }
        predictions = {
            plant: dict(reversed(list(row.items())))
            for plant, row in reversed(list(truth.items()))
        }
        self.assertEqual(score_predictions(truth, predictions).mean, 0)

    def test_missing_or_extra_plant_is_rejected(self):
        row = dict.fromkeys(TARGET_NAMES, 0.0)
        for predictions in ({}, {"B_two": row}, {"A_one": row, "B_two": row}):
            with self.subTest(ids=list(predictions)):
                with self.assertRaisesRegex(ValueError, "IDs"):
                    score_predictions({"A_one": row}, predictions)

    def test_missing_or_extra_target_is_rejected(self):
        row = dict.fromkeys(TARGET_NAMES, 0.0)
        for bad_row in ({"height": 0}, {**row, "variety": 0}):
            with self.subTest(targets=list(bad_row)):
                with self.assertRaisesRegex(ValueError, "target names"):
                    score_predictions({"A_one": row}, {"A_one": bad_row})

    def test_numerical_overflow_is_reported(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            rmsre([[0, 0, 0, 0]], [[1e308, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
