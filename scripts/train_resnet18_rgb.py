"""Train pretrained RGB ResNet18 with fixed plant-level CV; no holdout access."""

from pathlib import Path
import argparse, random, sys, os
import cv2, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parents[1] / "data/torch_cache")
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomato_rgbd.data import load_development_manifest, load_fold_dataframes
from tomato_rgbd.metrics import TARGET_NAMES, rmsre


class PlantDataset(Dataset):
    def __init__(self, table, images, train):
        self.table = table.reset_index(drop=True)
        self.images = images
        self.train = train

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        img = self.images[row.plant_id].copy()
        if self.train:
            img = np.rot90(img, random.randrange(4)).copy()
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        target_values = []
        for target_name in TARGET_NAMES:
            target_values.append(row[target_name])
        y = np.asarray(target_values, dtype=np.float32)
        valid = np.isfinite(y)
        y = np.nan_to_num(y, nan=0.0)
        return x, torch.from_numpy(y), torch.from_numpy(valid), row.plant_id


def make_image(path, size=320):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(
        rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
    )
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    output = np.zeros((size, size, 3), np.uint8)
    output[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return output


def loss_fn(pred, y, mask):
    # Nonnegative outputs are log1p targets; mask missing labels separately per trait
    target = torch.log1p(y)
    diff = (pred - target).square() * mask
    return (diff.sum(0) / mask.sum(0).clamp(min=1)).mean()


# Один проход по обучающим изображениям
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    # Статистики BatchNorm фиксируем, как и в исходном эксперименте
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()

    for images, targets, mask, _ in loader:
        images, targets, mask = images.to(device), targets.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets, mask)
        loss.backward()
        optimizer.step()


# Прогнозы для проверочной части: веса модели не обновляются
def validate(model, loader, device):
    model.eval()
    predictions, targets, ids = [], [], []
    with torch.no_grad():
        for images, values, mask, plant_ids in loader:
            raw = model(images.to(device))
            prediction = torch.expm1(raw.clamp(min=0, max=12)).cpu().numpy()
            values = values.numpy()
            values[~mask.numpy()] = np.nan
            predictions.extend(prediction)
            targets.extend(values)
            ids.extend(plant_ids)
    return ids, np.asarray(predictions), np.asarray(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    rows = load_development_manifest(ROOT)
    images = {}
    for row in rows:
        images[row["plant_id"]] = make_image(ROOT / row["rgb_path"])
    allpred = []
    folder = ROOT / "checkpoints/rgb_resnet18"
    folder.mkdir(parents=True, exist_ok=True)

    for fold in range(1 if args.smoke else 5):
        seed = 42 + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train, val = load_fold_dataframes(ROOT, fold)
        if args.smoke:
            train = train.iloc[:32]
            val = val.iloc[:16]
        train_loader = DataLoader(
            PlantDataset(train, images, True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        val_loader = DataLoader(
            PlantDataset(val, images, False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        # BatchNorm statistics stay fixed on small datasets/batches
        optim = torch.optim.AdamW(
            [
                {"params": model.fc.parameters(), "lr": 1e-3},
                {"params": list(model.layer4.parameters()), "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        for p in model.parameters():
            p.requires_grad = False
        for p in list(model.fc.parameters()) + list(model.layer4.parameters()):
            p.requires_grad = True
        best = float("inf")
        stale = 0
        best_preds = None
        for epoch in range(1, (2 if args.smoke else args.epochs) + 1):
            train_one_epoch(model, train_loader, optim, device)
            ids, preds, truth = validate(model, val_loader, device)
            score = rmsre(truth, preds)
            if score.mean < best:
                best = score.mean
                stale = 0
                best_preds = (ids, preds.copy())
                if not args.smoke:
                    torch.save(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "seed": seed,
                            "image_size": 320,
                        },
                        folder / f"fold_{fold}.pt",
                    )
            else:
                stale += 1
                if stale >= args.patience:
                    break
        for i, plant_id in enumerate(best_preds[0]):
            prediction = {"plant_id": plant_id, "fold": fold}
            for j, target_name in enumerate(TARGET_NAMES):
                prediction[target_name] = float(best_preds[1][i, j])
            allpred.append(prediction)
    if args.smoke:
        return
    oof = pd.DataFrame(allpred)
    truth = pd.DataFrame(rows)[["plant_id", *TARGET_NAMES]]
    joined = truth.merge(
        oof, on="plant_id", validate="one_to_one", suffixes=("_true", "_pred")
    )
    if len(joined) != 383 or oof.plant_id.duplicated().any():
        raise ValueError("OOF coverage mismatch")
    true_columns = []
    predicted_columns = []
    for target_name in TARGET_NAMES:
        true_columns.append(target_name + "_true")
        predicted_columns.append(target_name + "_pred")
    score = rmsre(joined[true_columns].to_numpy(), joined[predicted_columns].to_numpy())
    print("OOF RMSRE:", score.mean)
    for target, value in score.per_target.items():
        print(f"{target}: {value:.4f}")


if __name__ == "__main__":
    main()
