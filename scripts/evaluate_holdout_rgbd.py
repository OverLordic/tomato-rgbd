"""Финальная оценка ансамбля пяти сохранённых RGB-D моделей"""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomato_rgbd.metrics import TARGET_NAMES, rmsre


def main():
    output = ROOT / "reports/holdout_rgbd_scores.json"
    metadata_path = ROOT / "splits/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if output.exists() or metadata["holdout_evaluations"] != 0:
        raise RuntimeError("Holdout уже оценён - повторный запуск запрещён")

    spec = importlib.util.spec_from_file_location("rgbd", ROOT / "scripts/train_resnet18_rgbd.py")
    rgbd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rgbd)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    holdout_ids = pd.read_csv(ROOT / "splits/holdout.csv").plant_id
    development_ids = set(pd.read_csv(ROOT / "splits/cv_folds.csv").plant_id)
    if set(holdout_ids) & development_ids:
        raise ValueError("Обнаружено пересечение выборок")
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv")
    table = manifest.set_index("plant_id").loc[holdout_ids].reset_index()
    if len(table) != 96 or not table.plant_id.is_unique:
        raise ValueError("Неверный состав holdout")

    # Фиксируем веса и обработку до получения прогнозов и расчёта ошибок
    checkpoints = []
    states = []
    for fold in range(5):
        path = ROOT / f"checkpoints/rgbd_resnet18/fold_{fold}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["fold"] != fold or state["seed"] != 42 + fold or state["image_size"] != 320:
            raise ValueError("Веса не соответствуют протоколу")
        checkpoints.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "epoch": state["epoch"]})
        states.append(state)
    protocol = {"model": "ResNet18 RGB-D", "prediction": "mean of five fold predictions after expm1 and clamp(0,12)", "image_size": 320, "crop": False, "depth": "uint16 millimetres / 2000; nearest resize; zero invalid", "checkpoints": checkpoints}
    (ROOT / "reports/holdout_rgbd_protocol.json").write_text(json.dumps(protocol, indent=2))

    images = {}
    for row in table.itertuples():
        images[row.plant_id] = (rgbd.make_image(ROOT / row.rgb_path), rgbd.make_depth(ROOT / row.depth_path))
    # Передаём в Dataset пустые ответы: метки holdout нужны лишь после прогнозирования
    inputs = table.copy()
    for name in TARGET_NAMES:
        inputs[name] = np.nan
    loader = DataLoader(rgbd.PlantDataset(inputs, images, False), batch_size=16, shuffle=False)
    fold_predictions = []
    for state in states:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = rgbd.add_depth_channel(model)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        predictions = []
        with torch.inference_mode():
            for batch, _, _, _ in loader:
                predictions.extend(torch.expm1(model(batch.to(device)).clamp(0, 12)).cpu().numpy())
        fold_predictions.append(np.asarray(predictions))
    predictions = np.mean(fold_predictions, axis=0)
    prediction_table = pd.DataFrame(predictions, columns=TARGET_NAMES)
    prediction_table.insert(0, "plant_id", table.plant_id.to_numpy())
    prediction_table.to_csv(ROOT / "reports/holdout_rgbd_predictions.csv", index=False)

    truth = table[list(TARGET_NAMES)].to_numpy()
    score = rmsre(truth, predictions)
    mae = {}
    for column, name in enumerate(TARGET_NAMES):
        mae[name] = float(np.nanmean(np.abs(truth[:, column] - predictions[:, column])))
    positive = truth[:, 3] > 0
    report = {"scope": "final holdout; 96 plants; one evaluation; no fitting", "n_plants": len(table), "mean": score.mean, "per_target": score.per_target, "n_valid": score.n_valid, "mae": mae, "positive_fruit_plants": int(positive.sum()), "positive_fruit_mae": float(np.mean(np.abs(truth[positive, 3] - predictions[positive, 3]))), "device": str(device), "torch": torch.__version__, "protocol": "holdout_rgbd_protocol.json"}
    output.write_text(json.dumps(report, indent=2) + "\n")
    metadata["holdout_evaluations"] = 1
    metadata["final_evaluation"] = "reports/holdout_rgbd_scores.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
