"""Create stratified splits once, then validate and reuse the saved IDs."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import read_csv, sha256_file, write_csv


def validate_splits(
    rows: list[dict], holdout: list[dict], folds: list[dict], n_splits: int
) -> None:
    varieties = {r["plant_id"]: r["variety"] for r in rows}
    holdout_ids, dev_ids = [r["plant_id"] for r in holdout], [
        r["plant_id"] for r in folds
    ]
    if (
        len(holdout_ids) != len(set(holdout_ids))
        or len(dev_ids) != len(set(dev_ids))
        or set(holdout_ids) & set(dev_ids)
        or set(holdout_ids) | set(dev_ids) != set(varieties)
    ):
        raise ValueError(
            "Splits must be disjoint, unique and cover every plant exactly once"
        )
    for row in holdout + folds:
        if varieties[row["plant_id"]] != row["variety"]:
            raise ValueError("Saved variety disagrees with manifest")
    if {int(r["fold"]) for r in folds} != set(range(n_splits)):
        raise ValueError("Invalid CV fold numbers")
    for variety in sorted(set(varieties.values())):
        counts = [
            sum(r["variety"] == variety and int(r["fold"]) == f for r in folds)
            for f in range(n_splits)
        ]
        if max(counts) - min(counts) > 1 or min(counts) == 0:
            raise ValueError(f"Unbalanced/empty variety in CV: {variety}")


def freeze_splits(
    rows: list[dict],
    directory: Path,
    manifest_sha: str,
    images_sha: str,
    protocol: dict,
) -> tuple[list[dict], list[dict], dict]:
    """Never overwrite a frozen split or regenerate it after a data/config change."""
    rows = sorted(rows, key=lambda row: row["plant_id"])
    if len({r["plant_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated plant IDs")
    settings = {
        "seed": protocol["seed"],
        "holdout_fraction": protocol["holdout"]["fraction"],
        "n_splits": protocol["cross_validation"]["n_splits"],
        "manifest_sha256": manifest_sha,
        "images_fingerprint_sha256": images_sha,
    }
    if (
        protocol["holdout"]["stratify_by"] != "variety"
        or protocol["cross_validation"]["stratify_by"] != "variety"
        or protocol["cross_validation"]["shuffle"] is not True
    ):
        raise ValueError(
            "This implementation requires stratification by variety and shuffled CV"
        )
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: directory / name
        for name in ("holdout.csv", "cv_folds.csv", "metadata.json")
    }
    if any(p.exists() for p in paths.values()):
        if not all(p.exists() for p in paths.values()):
            raise ValueError("Incomplete frozen split; will not overwrite it")
        metadata = json.loads(paths["metadata.json"].read_text())
        if metadata["settings"] != settings:
            raise ValueError(
                "Data or protocol changed; refusing to regenerate frozen splits"
            )
        for name in ("holdout.csv", "cv_folds.csv"):
            if sha256_file(paths[name]) != metadata["split_files_sha256"][name]:
                raise ValueError(f"Frozen split file was modified: {name}")
        holdout, folds = read_csv(paths["holdout.csv"]), read_csv(paths["cv_folds.csv"])
        validate_splits(rows, holdout, folds, settings["n_splits"])
        return holdout, folds, metadata

    ids = [r["plant_id"] for r in rows]
    varieties = {r["plant_id"]: r["variety"] for r in rows}
    dev_ids, test_ids = train_test_split(
        ids,
        test_size=settings["holdout_fraction"],
        random_state=settings["seed"],
        stratify=[varieties[i] for i in ids],
    )
    dev_ids = sorted(dev_ids)
    if min(Counter(varieties[i] for i in dev_ids).values()) < settings["n_splits"]:
        raise ValueError("Too few development plants in a variety for the requested CV")
    cv = StratifiedKFold(
        settings["n_splits"], shuffle=True, random_state=settings["seed"]
    )
    folds = []
    for fold, (_, validation_indices) in enumerate(
        cv.split(dev_ids, [varieties[i] for i in dev_ids])
    ):
        folds.extend(
            {"plant_id": dev_ids[i], "variety": varieties[dev_ids[i]], "fold": fold}
            for i in validation_indices
        )
    folds.sort(key=lambda row: row["plant_id"])
    holdout = [{"plant_id": i, "variety": varieties[i]} for i in sorted(test_ids)]
    validate_splits(rows, holdout, folds, settings["n_splits"])
    write_csv(paths["holdout.csv"], holdout, ("plant_id", "variety"))
    write_csv(paths["cv_folds.csv"], folds, ("plant_id", "variety", "fold"))
    metadata = {
        "settings": settings,
        "algorithm": "sorted IDs -> train_test_split(stratify=variety) -> sorted dev IDs -> StratifiedKFold",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "counts": {
            "all": len(rows),
            "development": len(folds),
            "holdout": len(holdout),
        },
        "variety_counts": {
            "all": dict(Counter(r["variety"] for r in rows)),
            "development": dict(Counter(r["variety"] for r in folds)),
            "holdout": dict(Counter(r["variety"] for r in holdout)),
        },
        "validation_folds": {
            str(f): {
                "n": sum(int(r["fold"]) == f for r in folds),
                "varieties": dict(
                    Counter(r["variety"] for r in folds if int(r["fold"]) == f)
                ),
            }
            for f in range(settings["n_splits"])
        },
        "split_files_sha256": {
            n: sha256_file(paths[n]) for n in ("holdout.csv", "cv_folds.csv")
        },
        "holdout_evaluations": 0,
        "note": "Only integrity checks apply to all files; target distributions and visual analysis use development only.",
    }
    paths["metadata.json"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return holdout, folds, metadata


def image_fingerprint(audit: list[dict], calibration_sha: str) -> str:
    values = [
        {
            k: r[k]
            for k in (
                "plant_id",
                "rgb_pixels_sha256",
                "depth_pixels_sha256",
                "ir_left_pixels_sha256",
                "ir_right_pixels_sha256",
            )
        }
        for r in sorted(audit, key=lambda r: r["plant_id"])
    ]
    payload = {"images": values, "calibration_sha256": calibration_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
