"""Verify/download the dataset, audit every image, and freeze plant-level splits."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2
import sklearn

from tomato_rgbd.data import (
    IMAGE_COLUMNS,
    MANIFEST_COLUMNS,
    UNITS,
    audit_images,
    build_manifest,
    development_summary,
    read_json,
    sha256_file,
    write_csv,
)
from tomato_rgbd.download import download_archive, extract_archive, verify_archive
from tomato_rgbd.splits import freeze_splits, image_fingerprint


def write_explanation(report):
    """Turn verified facts into a small readable report without exposing holdout labels."""
    split = report["splits"]
    lines = [
        "# Подготовка данных: фактические результаты",
        "",
        "Архив полностью загружен, его MD5 совпала с опубликованной 4TU.",
        f"Проверены {report['n_plants']} растений и {report['n_images_decoded']} PNG.",
        "Проверены уникальность ID/исходных записей и отсутствие точных повторов RGB.",
        "",
        "## Фиксированное разбиение",
        "",
        "| Сорт | Всего | Development | Holdout |",
        "|---|---:|---:|---:|",
    ]
    counts = split["variety_counts"]
    for variety in sorted(counts["all"]):
        lines.append(
            f"| {variety} | {counts['all'][variety]} | {counts['development'][variety]} | {counts['holdout'][variety]} |"
        )
    lines.extend(
        [
            f"| Всего | {split['counts']['all']} | {split['counts']['development']} | {split['counts']['holdout']} |",
            "",
            "Seed=42. Разбиение стратифицировано по сорту. ID сохранены в splits/.",
            "",
            "| Фолд validation | Растений | A | B | C | D |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, values in split["validation_folds"].items():
        numbers = " | ".join(str(values["varieties"].get(v, 0)) for v in "ABCD")
        lines.append(f"| {fold} | {values['n']} | {numbers} |")
    lines.extend(
        [
            "",
            "Для фолда k обучение использует остальные development-строки, проверка - строки с fold=k.",
            "Holdout во всех пяти запусках исключён. Повторный запуск читает те же ID.",
            "",
            "## Формат и единицы",
            "",
            f"- RGB и выровненная depth: {report['color_size'][0]}×{report['color_size'][1]}.",
            f"- RGB modes: {report['rgb_modes']}; depth dtype: {report['depth_dtype']}.",
            f"- Размеры IR: {report['ir_sizes']}.",
            "- Depth хранится в 16-битном PNG. Ноль - невалидное измерение.",
            "- Метрический масштаб по демо WUR: raw × 0.001 м; независимая калибровка линейкой ещё не выполнена.",
            "- Значение depth - расстояние от камеры, не высота растения над субстратом.",
            "",
            f"Медиана глубины полной сцены по development: {report['development_only']['median_scene_depth_m']} м.",
            "Она включает фон. Приблизительные 90 см из описания не считаются проверенной опорой;",
            "перед расчётом высоты нужна отдельная проверка геометрии и плоскости субстрата.",
            "",
            "## Целевые измерения: только development",
            "",
            "| Цель | Единица | Известно | Пропущено | Минимум | Медиана | Максимум |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in report["development_only"]["targets"].items():
        lines.append(
            f"| {name} | {values['unit']} | {values['observed']} | {values['missing']} | {values['min']} | {values['median']} | {values['max']} |"
        )
    lines.extend(
        [
            "",
            "Пропуски целей в CSV - пустые ячейки, а не нули. Строки целиком не удалялись.",
            "Полная локальная таблица: data/processed/manifest.csv; рабочая часть: development.csv.",
            "Для holdout отдельно сохранены служебные поля без целей в holdout_metadata.csv.",
            "",
            "## Ограничения и следующий шаг",
            "",
            "Совпадение размеров не доказывает пространственное совмещение RGB/depth.",
            "Проверка точных RGB-копий не заменяет поиск почти одинаковых изображений.",
            "Контрольная сумма помогает обнаружить изменение файлов, но не исправляет ошибки ручных измерений.",
            "Модели не обучались; оценки на holdout не вычислялись.",
            "Следующий шаг - визуальный разбор development-примеров и baseline на масках/признаках.",
            "",
            "Подробный машинный отчёт: [data_audit.json](data_audit.json).",
            "",
        ]
    )
    (ROOT / "reports/data_preparation.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/resume official archive with curl",
    )
    args = parser.parse_args()
    protocol = read_json(ROOT / "configs/protocol.json")
    if {t["name"]: t["unit"] for t in protocol["targets"]} != UNITS:
        raise ValueError("Target names/units must match official WUR units")
    if (
        protocol["data"]["root"] != "data/raw/single_plant_cam"
        or protocol["data"]["manifest"] != "data/processed/manifest.csv"
        or protocol["holdout"]["plant_ids_file"] != "splits/holdout.csv"
        or protocol["cross_validation"]["folds_file"] != "splits/cv_folds.csv"
    ):
        raise ValueError(
            "Unexpected paths in protocol; update the loader explicitly before changing them"
        )
    archive = ROOT / "data/raw/single_plant_camera.zip"
    if args.download:
        download_archive(archive)
    if not archive.exists():
        parser.error("Archive missing. Run with --download.")
    print("Verifying entire archive: size, MD5 and SHA256...", flush=True)
    receipt = verify_archive(archive)
    print(
        "Archive matches publisher MD5. Extracting with ZIP CRC verification...",
        flush=True,
    )
    dataset = extract_archive(archive, archive.parent, receipt)
    rows = build_manifest(dataset, ROOT)
    calibration_path = dataset / "oak-d-s2-poe_conf.json"
    calibration = read_json(calibration_path)
    print(
        f"Manifest: {len(rows)} unique plants; checking all four images per plant...",
        flush=True,
    )
    images = audit_images(rows, ROOT, calibration)

    processed = ROOT / "data/processed"
    manifest_path = processed / "manifest.csv"
    # Avoid changing an already-frozen manifest before checking its identity
    candidate = processed / "manifest.candidate.csv"
    write_csv(candidate, rows, MANIFEST_COLUMNS)
    manifest_sha = sha256_file(candidate)
    if manifest_path.exists() and sha256_file(manifest_path) != manifest_sha:
        raise ValueError("Manifest changed; inspect it before touching frozen data")
    holdout, folds, metadata = freeze_splits(
        rows,
        ROOT / "splits",
        manifest_sha,
        image_fingerprint(images, sha256_file(calibration_path)),
        protocol,
    )
    candidate.replace(manifest_path)
    write_csv(processed / "image_audit.csv", images, list(images[0]))
    dev_ids = {row["plant_id"] for row in folds}
    dev_rows = [r for r in rows if r["plant_id"] in dev_ids]
    write_csv(processed / "development.csv", dev_rows, MANIFEST_COLUMNS)
    write_csv(
        processed / "holdout_metadata.csv",
        [
            {k: r[k] for k in ("plant_id", "variety", "source_subset", *IMAGE_COLUMNS)}
            for r in rows
            if r["plant_id"] not in dev_ids
        ],
        (
            "plant_id",
            "variety",
            "source_subset",
            "rgb_path",
            "depth_path",
            "ir_left_path",
            "ir_right_path",
        ),
    )
    report = {
        "status": "passed",
        "archive": receipt,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "OpenCV": cv2.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_plants": len(rows),
        "n_images_decoded": len(rows) * 4,
        "source_subset_counts": dict(Counter(r["source_subset"] for r in rows)),
        "unique_plant_ids": True,
        "unique_original_acquisitions": True,
        "exact_duplicate_rgb_groups": [],
        "all_image_files_match_labels": True,
        "calibration_sha256": sha256_file(calibration_path),
        "color_size": [
            calibration["color_int"]["width"],
            calibration["color_int"]["height"],
        ],
        "rgb_modes": dict(Counter(r["rgb_mode"] for r in images)),
        "depth_modes": dict(Counter(r["depth_mode"] for r in images)),
        "depth_dtype": dict(Counter(r["depth_dtype"] for r in images)),
        "ir_sizes": dict(Counter(f"{r['ir_width']}x{r['ir_height']}" for r in images)),
        "depth_units": {
            "meters_per_raw_unit": 0.001,
            "basis": "WUR point-cloud-demo fallback; no DepthScale key in calibration",
            "zero": "invalid measurement, excluded from depth quantiles",
            "physical_scale_check": "Plausibility only; no independent ruler calibration yet",
        },
        "splits": metadata,
        "development_only": development_summary(rows, images, dev_ids),
        "limitations": [
            "Exact pixel duplicate check is not a near-duplicate detector",
            "Matching shapes alone does not prove RGB-depth spatial alignment",
            "Full-scene depth is not a validated camera-to-substrate reference; do not assume a constant 0.90 m",
            "Units of manually measured targets come from WUR documentation, not PNG pixels",
            "No holdout target distributions, visual examples, training or model scores examined",
        ],
    }
    (ROOT / "reports/data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    write_explanation(report)
    protocol["status"] = "frozen_splits_created"
    (ROOT / "configs/protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"Done: {len(folds)} development, {len(holdout)} holdout; CV folds saved.",
        flush=True,
    )


if __name__ == "__main__":
    main()
