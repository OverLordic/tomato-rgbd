"""Official WUR score: mean of four per-target RMSRE values.

Source: docs/sources/Autonomous_greenhouse_challenge_technical_document.pdf,
section 4, page 7. See docs/learning/02-rmsre.md for a worked explanation.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


# Column order is shared by training, prediction and evaluation.
TARGET_NAMES = ("height", "fw_plant", "leaf_area", "number_of_red_fruits")


@dataclass(frozen=True)
class RMSREResult:
    """The overall score, each target's score, and its observed label count."""

    mean: float
    per_target: dict[str, float]
    n_valid: dict[str, int]


def rmsre(y_true: ArrayLike, y_pred: ArrayLike) -> RMSREResult:
    """Score arrays shaped (plants, 4), in TARGET_NAMES order.

    Labels may contain NaN; each target uses its own observed rows. Predictions
    must be finite everywhere, including where a label is missing. Truth must
    be nonnegative in the official units (cm, g, cm², count).

    This function never rounds, clips or rescales predictions. A target with
    no labels is an error: the official four-target mean is then undefined.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    if truth.ndim != 2 or truth.shape[1] != len(TARGET_NAMES):
        raise ValueError("y_true must have shape (n_plants, 4) in TARGET_NAMES order")
    if truth.shape[0] == 0:
        raise ValueError("At least one plant is required")
    if pred.shape != truth.shape:
        raise ValueError("y_pred must have the same shape as y_true")
    if not np.isfinite(pred).all():
        raise ValueError("Every prediction must be finite, even for missing labels")
    if np.isinf(truth).any() or (truth < 0).any():
        raise ValueError("Ground truth must be nonnegative or NaN, never infinite")

    per_target = {}
    n_valid = {}
    for column, name in enumerate(TARGET_NAMES):
        observed = ~np.isnan(truth[:, column])
        n_valid[name] = int(observed.sum())
        if n_valid[name] == 0:
            raise ValueError(f"No observed ground truth for target {name}")

        true_values = truth[observed, column]
        pred_values = pred[observed, column]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                relative_error = (pred_values - true_values) / (true_values + 1.0)
                per_target[name] = float(np.sqrt(np.mean(relative_error**2)))
        except FloatingPointError as exc:
            raise ValueError(f"Numerical overflow while scoring target {name}") from exc

    return RMSREResult(
        mean=float(np.mean(list(per_target.values()))),
        per_target=per_target,
        n_valid=n_valid,
    )


def score_predictions(
    truth: Mapping[str, Mapping[str, float]],
    predictions: Mapping[str, Mapping[str, float]],
) -> RMSREResult:
    """Score official-style {plant_id: {target: value}} dictionaries.

    Dictionary insertion order does not matter: align by plant ID and target
    name. Exact ID/target sets are required so missing predictions cannot
    disappear silently from the score.
    """
    if not truth:
        raise ValueError("At least one plant is required")
    if set(truth) != set(predictions):
        raise ValueError("Prediction plant IDs must match ground truth exactly")

    true_rows = []
    pred_rows = []
    for plant_id, true_row in truth.items():
        pred_row = predictions[plant_id]
        if set(true_row) != set(TARGET_NAMES) or set(pred_row) != set(TARGET_NAMES):
            raise ValueError(f"Expected exactly four target names for plant {plant_id}")
        true_rows.append([true_row[name] for name in TARGET_NAMES])
        pred_rows.append([pred_row[name] for name in TARGET_NAMES])

    return rmsre(true_rows, pred_rows)
