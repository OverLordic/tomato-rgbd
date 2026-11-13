from pathlib import Path
import sys, json, cv2, numpy as np, pandas as pd

r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / "src"))
from tomato_rgbd.geometry import scene, backproject
from tomato_rgbd.data import read_json

row = (
    pd.read_csv(r / "data/processed/development.csv")
    .query("plant_id == 'A_04ade032'")
    .iloc[0]
)
cal = read_json(r / "data/raw/single_plant_cam/oak-d-s2-poe_conf.json")
f, info = scene(r / row.rgb_path, r / row.depth_path, cal)
coef = info["fit"][0]
a, b, c = coef
im = cv2.imread(str(r / row.rgb_path))
depth = cv2.imread(str(r / row.depth_path), cv2.IMREAD_UNCHANGED)
# Crop-local rectangles based on visual inspection; approximate and potentially contaminated
regions = {
    "soil_1": (690, 475, 720, 505),
    "soil_2": (620, 675, 650, 705),
    "soil_3": (805, 690, 835, 720),
    "rim_1": (660, 447, 700, 462),
    "rim_2": (520, 665, 540, 705),
}
result = {}
crop = im[450:1650, 1200:2600].copy()
for name, (x1, y1, x2, y2) in regions.items():
    x1 += 1200
    x2 += 1200
    y1 += 450
    y2 += 450
    d = depth[y1:y2, x1:x2]
    yy, xx = np.indices(d.shape)
    z = d * 0.001
    x = (xx + x1 - cal["color_int"]["ppx"]) * z / cal["color_int"]["fx"]
    y = (yy + y1 - cal["color_int"]["ppy"]) * z / cal["color_int"]["fy"]
    h = (a * x + b * y + c - z) / np.sqrt(a * a + b * b + 1) * 100
    vals = h[d > 0]
    result[name] = {
        "valid_fraction": float((d > 0).mean()),
        "n_valid": int(vals.size),
        "above_tray_median_cm": float(np.median(vals)) if vals.size else None,
        "q10_cm": float(np.quantile(vals, 0.1)) if vals.size else None,
        "q90_cm": float(np.quantile(vals, 0.9)) if vals.size else None,
    }
    cv2.rectangle(
        crop,
        (x1 - 1200, y1 - 450),
        (x2 - 1200, y2 - 450),
        (0, 255, 255) if name.startswith("soil") else (255, 0, 255),
        2,
    )
    cv2.putText(
        crop,
        name,
        (x1 - 1200, y1 - 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
cv2.imwrite(str(r / "reports/substrate_example_marked.png"), crop)
print("plane", coef, "manual target", row.height, "q99", f["height_q99_cm"])
print(json.dumps(result, indent=2))

geometry = pd.read_csv(r / "data/processed/geometry_development.csv")
dev = pd.read_csv(r / "data/processed/development.csv")
j = dev.merge(geometry, on="plant_id", validate="one_to_one")
j = j[(j.plane_ok == 1) & j.height_q99_cm.notna() & j.height.notna()]
comparisons = {}
for label, offset in [
    ("tray", np.zeros(len(j))),
    ("rim_10_B7", np.where(j.variety == "B", 7, 10)),
    ("rim_minus2_B7", np.where(j.variety == "B", 7, 8)),
]:
    pred = np.maximum(j.height_q99_cm.to_numpy() - offset, 0)
    truth = j.height.to_numpy()
    comparisons[label] = {
        "n": len(j),
        "mae_cm": float(np.mean(abs(pred - truth))),
        "bias_cm": float(np.mean(pred - truth)),
        "rmsre": float(np.sqrt(np.mean(((pred - truth) / (truth + 1)) ** 2))),
    }
report = {
    "plant_id": "A_04ade032",
    "regions": result,
    "manual_height_cm": float(row.height),
    "q99_above_tray_cm": float(f["height_q99_cm"]),
    "hypotheses": comparisons,
    "warning": "Exploratory development subset; no independent substrate calibration; holdout not evaluated",
}
(r / "reports/substrate_hypotheses.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(comparisons, indent=2))
