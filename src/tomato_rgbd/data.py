"""One row per plant; validate files and units without fitting any model."""

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import cv2
import pandas as pd

from .metrics import TARGET_NAMES

SUBSETS = ("train", "validation1", "validation2", "validation3", "test")
IMAGE_COLUMNS = ("rgb_path", "depth_path", "ir_left_path", "ir_right_path")
MANIFEST_COLUMNS = ("plant_id", "variety", "source_subset", "original_name", "imaging_date",
                    *IMAGE_COLUMNS, *TARGET_NAMES)
UNITS = dict(zip(TARGET_NAMES, ("cm", "g", "cm2", "count")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_keys)


def write_csv(path: Path, rows: list[dict], columns) -> None:
    """Use portable project-relative paths and blank cells for missing labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if isinstance(row[name], float) and math.isnan(row[name])
                             else row[name] for name in columns})


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_manifest(dataset_root: Path, project_root: Path, expected_plants=479) -> list[dict]:
    """Join labels, original names and all four images by their plant ID.

    This performs technical label validation, never target-based selection or tuning.
    Original acquisition names help detect duplicate observations. Their date and
    treatment information must not become model inputs.
    """
    rows, seen_ids, seen_originals = [], set(), set()
    for subset in SUBSETS:
        folder = dataset_root / subset
        truth = read_json(folder / f"ground_truth_{subset}.json")
        original_names = read_json(folder / f"anon_keys_{subset}.json")
        if set(truth) != set(original_names):
            raise ValueError(f"Label/original-name ID mismatch in {subset}")
        expected_rgb, expected_depth = set(), set()
        for plant_id, labels in truth.items():
            if not re.fullmatch(r"[ABCD]_[a-f0-9]{8}", plant_id) or plant_id in seen_ids:
                raise ValueError(f"Invalid or repeated plant ID: {plant_id}")
            seen_ids.add(plant_id)
            original = original_names[plant_id]
            if not isinstance(original, str) or original in seen_originals:
                raise ValueError(f"Repeated or invalid original acquisition: {plant_id}")
            seen_originals.add(original)
            date_match = re.search(r"_(\d{4}_\d{2}_\d{2})_", original)
            if date_match is None:
                raise ValueError(f"No acquisition date for {plant_id}")
            date = datetime.strptime(date_match[1], "%Y_%m_%d").date().isoformat()
            if set(labels) != set(TARGET_NAMES):
                raise ValueError(f"Wrong target names for {plant_id}")
            values = {}
            for target in TARGET_NAMES:
                value = labels[target]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Non-numeric label for {plant_id}/{target}")
                value = float(value)
                if math.isinf(value) or value < 0:
                    raise ValueError(f"Invalid label for {plant_id}/{target}")
                if target == "number_of_red_fruits" and math.isfinite(value) and not value.is_integer():
                    raise ValueError(f"Non-integer fruit count for {plant_id}")
                values[target] = value

            files = {
                "rgb_path": folder / "rgb" / f"{plant_id}.png",
                "depth_path": folder / "depth" / f"{plant_id}_depth.png",
                "ir_left_path": folder / "depth" / f"{plant_id}_irL.png",
                "ir_right_path": folder / "depth" / f"{plant_id}_irR.png",
            }
            for image in files.values():
                if not image.is_file():
                    raise ValueError(f"Missing image: {image.relative_to(project_root)}")
            expected_rgb.add(files["rgb_path"])
            expected_depth.update(files[k] for k in IMAGE_COLUMNS[1:])
            rows.append({"plant_id": plant_id, "variety": plant_id[0], "source_subset": subset,
                         "original_name": original, "imaging_date": date,
                         **{k: p.relative_to(project_root).as_posix() for k, p in files.items()},
                         **values})
        if set((folder / "rgb").glob("*.png")) != expected_rgb:
            raise ValueError(f"Orphan RGB files in {subset}")
        if set((folder / "depth").glob("*.png")) != expected_depth:
            raise ValueError(f"Orphan depth/IR files in {subset}")
    if len(rows) != expected_plants:
        raise ValueError(f"Expected {expected_plants} plants, found {len(rows)}")
    return sorted(rows, key=lambda row: row["plant_id"])


def _read_png(path: Path):
    """Decode the full image and retain the original PNG bit depth."""
    with path.open("rb") as stream:
        header = stream.read(29)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 29:
        raise ValueError(f"Invalid PNG header: {path.name}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode PNG: {path.name}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        mode = "RGBA" if image.shape[2] == 4 else "RGB"
    else:
        mode = "I;16" if image.dtype == np.uint16 else "L"
    return image, mode, int(header[24])



def audit_images(rows: list[dict], project_root: Path, calibration: dict,
                 progress=print) -> list[dict]:
    """Decode every modality and fingerprint pixels to detect exact duplicate RGBs.

    Only technical integrity is examined for all plants. No segmentation, visual
    selection, target distribution analysis or model evaluation is performed here.
    """
    width, height = calibration["color_int"]["width"], calibration["color_int"]["height"]
    output, rgb_groups = [], defaultdict(list)
    for number, row in enumerate(rows, 1):
        rgb, rgb_mode, rgb_bits = _read_png(project_root / row["rgb_path"])
        depth, depth_mode, depth_bits = _read_png(project_root / row["depth_path"])
        if rgb.shape[:2] != (height, width) or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"Unexpected RGB shape for {row['plant_id']}: {rgb.shape}")
        if rgb_bits != 8 or rgb.dtype != np.uint8:
            raise ValueError(f"RGB must be 8-bit for {row['plant_id']}")
        if rgb.shape[2] == 4 and not np.all(rgb[:, :, 3] == 255):
            raise ValueError(f"Non-opaque RGB alpha for {row['plant_id']}")
        if depth.shape != (height, width) or depth_bits != 16:
            raise ValueError(f"Unexpected depth format for {row['plant_id']}")
        if not np.issubdtype(depth.dtype, np.integer) or depth.min() < 0 or depth.max() > 65535:
            raise ValueError(f"Invalid depth values for {row['plant_id']}")
        depth = depth.astype(np.uint16, copy=False)
        ir_shapes, ir_hashes = [], {}
        for key in IMAGE_COLUMNS[2:]:
            ir, _, bits = _read_png(project_root / row[key])
            if ir.ndim != 2 or bits != 8 or ir.dtype != np.uint8:
                raise ValueError(f"Unexpected IR format for {row['plant_id']}")
            ir_shapes.append(ir.shape)
            ir_hashes[key.replace("_path", "_pixels_sha256")] = hashlib.sha256(ir.tobytes()).hexdigest()
        if ir_shapes[0] != ir_shapes[1]:
            raise ValueError(f"Left/right IR size mismatch for {row['plant_id']}")
        rgb_sha = hashlib.sha256(rgb[:, :, :3].tobytes()).hexdigest()
        depth_sha = hashlib.sha256(depth.astype("<u2", copy=False).tobytes()).hexdigest()
        rgb_groups[rgb_sha].append(row["plant_id"])
        # Diagnostic quantiles use a deterministic stride-4 grid, not image-specific scaling.
        sampled = depth[::4, ::4]
        valid = sampled[sampled > 0]
        quantiles = np.quantile(valid, [0.05, 0.5, 0.95]) * 0.001 if valid.size else [None] * 3
        output.append({"plant_id": row["plant_id"], "rgb_width": width, "rgb_height": height,
                       "rgb_mode": rgb_mode, "rgb_dtype": str(rgb.dtype),
                       "depth_width": depth.shape[1], "depth_height": depth.shape[0],
                       "depth_mode": depth_mode, "depth_dtype": str(depth.dtype),
                       "depth_png_bits": depth_bits, "ir_width": ir_shapes[0][1],
                       "ir_height": ir_shapes[0][0], "zero_depth_fraction": float(np.mean(depth == 0)),
                       "sample_depth_q05_m": quantiles[0], "sample_depth_median_m": quantiles[1],
                       "sample_depth_q95_m": quantiles[2], "rgb_pixels_sha256": rgb_sha,
                       "depth_pixels_sha256": depth_sha, **ir_hashes})
        if progress and (number % 25 == 0 or number == len(rows)):
            progress(f"Decoded and checked: {number}/{len(rows)} plants", flush=True)
    duplicates = [ids for ids in rgb_groups.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(f"Exact duplicate RGB pixels; resolve grouping before splitting: {duplicates}")
    return output


def development_summary(rows: list[dict], image_audit: list[dict], development_ids: set) -> dict:
    """Summarize targets/depth only on development; do not inspect holdout distributions."""
    dev = [row for row in rows if row["plant_id"] in development_ids]
    dev_images = [row for row in image_audit if row["plant_id"] in development_ids]
    labels = {}
    for target in TARGET_NAMES:
        values = np.array([row[target] for row in dev], dtype=float)
        observed = values[np.isfinite(values)]
        labels[target] = {"unit": UNITS[target], "observed": len(observed),
                          "missing": int(np.isnan(values).sum()),
                          "min": float(observed.min()) if observed.size else None,
                          "median": float(np.median(observed)) if observed.size else None,
                          "max": float(observed.max()) if observed.size else None}
    depth_medians = [r["sample_depth_median_m"] for r in dev_images
                     if r["sample_depth_median_m"] is not None]
    return {"n_plants": len(dev), "variety_counts": dict(Counter(r["variety"] for r in dev)),
            "targets": labels,
            "median_zero_depth_fraction": float(np.median([r["zero_depth_fraction"] for r in dev_images])),
            "median_scene_depth_m": float(np.median(depth_medians)) if depth_medians else None,
            "note": "Depth statistics cover the full scene, not a plant mask; quantiles use stride 4."}


def load_development_manifest(project_root: Path) -> list[dict]:
    """Recommended entry point for later training: return development plants only."""
    ids = {r["plant_id"] for r in read_csv(project_root / "splits/cv_folds.csv")}
    rows = read_csv(project_root / "data/processed/manifest.csv")
    result = [r for r in rows if r["plant_id"] in ids]
    if {r["plant_id"] for r in result} != ids:
        raise ValueError("Manifest does not contain exactly the saved development IDs")
    for row in result:
        for target in TARGET_NAMES:
            row[target] = float(row[target]) if row[target] else float("nan")
    return result


def load_fold_dataframes(project_root: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return development-only train/validation tables for a saved fold.

    Targets may contain NaN; images are loaded later from the path columns.
    Holdout IDs never enter either table.
    """
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    development = pd.DataFrame(load_development_manifest(project_root))
    folds = pd.read_csv(project_root / "splits/cv_folds.csv", dtype={"plant_id": str, "variety": str})
    if folds["plant_id"].duplicated().any():
        raise ValueError("Duplicate development ID in folds")
    joined = development.merge(folds, on=["plant_id", "variety"], validate="one_to_one", how="inner")
    if len(joined) != len(development) or len(joined) != len(folds):
        raise ValueError("Development and fold IDs/varieties do not match")
    holdout = set(pd.read_csv(project_root / "splits/holdout.csv")["plant_id"])
    if set(joined["plant_id"]) & holdout:
        raise ValueError("Holdout plant found in development")
    train = joined.loc[joined["fold"] != fold].copy().reset_index(drop=True)
    validation = joined.loc[joined["fold"] == fold].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Empty train or validation fold")
    return train, validation
