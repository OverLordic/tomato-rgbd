"""Train pretrained four-channel RGB-D ResNet18 with fixed plant-level CV; no holdout access."""

from pathlib import Path
import argparse, json, random, sys, time, os

os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parents[1] / "data/torch_cache")
)
import cv2, numpy as np, pandas as pd, torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

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
        img, depth = self.images[row.plant_id]
        img = img.copy()
        depth = depth.copy()
        if self.train:
            turns = random.randrange(4)
            img = np.rot90(img, turns).copy()
            depth = np.rot90(depth, turns).copy()
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                depth = np.fliplr(depth).copy()
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        depth_tensor = torch.from_numpy(depth.copy()).float().unsqueeze(0).div_(2000.0)
        x = torch.cat((x, depth_tensor), dim=0)
        y = np.asarray([row[t] for t in TARGET_NAMES], dtype=np.float32)
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


def make_depth(path, size=320):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Invalid depth {path}")
    h, w = depth.shape
    scale = size / max(h, w)
    resized = cv2.resize(
        depth, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_NEAREST
    )
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    output = np.zeros((size, size), np.uint16)
    output[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return output


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.rgb.fc = nn.Identity()
        self.depth = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512 + 64, 4)

    def forward(self, x):
        rgb = self.rgb(x[:, :3])
        depth = self.depth(x[:, 3:4])
        return self.fc(torch.cat((rgb, depth), dim=1))


def loss_fn(pred, y, mask):
    # Nonnegative outputs are log1p targets; mask missing labels separately per trait
    target = torch.log1p(y)
    diff = (pred - target).square() * mask
    return (diff.sum(0) / mask.sum(0).clamp(min=1)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=[320, 448], default=320)
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
    print("Device:", device, flush=True)
    rows = load_development_manifest(ROOT)
    images = {}
    for n, row in enumerate(rows, 1):
        images[row["plant_id"]] = (
            make_image(ROOT / row["rgb_path"], args.size),
            make_depth(ROOT / row["depth_path"], args.size),
        )
        if n % 100 == 0:
            print("Prepared RGB-D:", n, "/383", flush=True)
    allpred = []
    hist = []
    folder = ROOT / f"checkpoints/fusion_{args.size}"
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
        model = FusionNet().to(device)
        # BatchNorm statistics stay fixed on small datasets/batches
        optim = torch.optim.AdamW(
            [
                {
                    "params": list(model.fc.parameters())
                    + list(model.depth.parameters()),
                    "lr": 1e-3,
                },
                {"params": list(model.rgb.layer4.parameters()), "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        for p in model.parameters():
            p.requires_grad = False
        for p in (
            list(model.fc.parameters())
            + list(model.depth.parameters())
            + list(model.rgb.layer4.parameters())
        ):
            p.requires_grad = True
        best = float("inf")
        best_epoch = 0
        stale = 0
        best_preds = None
        for epoch in range(1, (2 if args.smoke else args.epochs) + 1):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            training = []
            for x, y, mask, _ in train_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optim.zero_grad(set_to_none=True)
                out = model(x)
                loss = loss_fn(out, y, mask)
                loss.backward()
                optim.step()
                training.append(loss.item())
            model.eval()
            preds = []
            truth = []
            ids = []
            with torch.no_grad():
                for x, y, mask, plant_ids in val_loader:
                    raw = model(x.to(device))
                    prediction = torch.expm1(raw.clamp(min=0, max=12)).cpu().numpy()
                    y = y.numpy()
                    y[~mask.numpy()] = np.nan
                    preds.extend(prediction)
                    truth.extend(y)
                    ids.extend(plant_ids)
            preds = np.asarray(preds)
            truth = np.asarray(truth)
            score = rmsre(truth, preds)
            epoch_result = {
                "fold": fold,
                "epoch": epoch,
                "train_log_mse": float(np.mean(training)),
                "val_rmsre": score.mean,
            }
            for target_name, value in score.per_target.items():
                epoch_result[f"val_{target_name}"] = value
            hist.append(epoch_result)
            print(
                f"Fold {fold} epoch {epoch}: train={np.mean(training):.3f} val={score.mean:.3f}",
                flush=True,
            )
            if score.mean < best:
                best = score.mean
                best_epoch = epoch
                stale = 0
                best_preds = (ids, preds.copy())
                if not args.smoke:
                    torch.save(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "seed": seed,
                            "image_size": args.size,
                        },
                        folder / f"fold_{fold}.pt",
                    )
            else:
                stale += 1
                if stale >= args.patience:
                    break
        for i, id in enumerate(best_preds[0]):
            prediction = {
                "plant_id": id,
                "fold": fold,
            }
            for j, t in enumerate(TARGET_NAMES):
                prediction[t] = float(best_preds[1][i, j])
            allpred.append(prediction)
        print(f"Fold {fold}: best epoch={best_epoch}, score={best:.3f}", flush=True)
    if args.smoke:
        print("Smoke test passed; no OOF evaluation saved.", flush=True)
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
    oof.to_csv(ROOT / f"reports/fusion_{args.size}_oof.csv", index=False)
    pd.DataFrame(hist).to_csv(
        ROOT / f"reports/fusion_{args.size}_training.csv", index=False
    )
    summary = {
        "scope": "development only; 5-fold OOF; holdout untouched",
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "image_size": args.size,
        "depth_scaling": "raw millimetres / 2000; zero invalid/padding; nearest resize",
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "loss": "masked per-target log1p MSE",
        "weights": "ResNet18 IMAGENET1K_V1",
        "mean": score.mean,
        "per_target": score.per_target,
        "n_valid": score.n_valid,
        "note": "checkpoint selected on validation RMSRE; optimistic CV selection possible",
    }
    (ROOT / f"reports/fusion_{args.size}_scores.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
