from pathlib import Path
import json, sys
import cv2, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomato_rgbd.data import load_development_manifest, load_fold_dataframes, read_json
from tomato_rgbd.baseline import FEATURES
from tomato_rgbd.metrics import TARGET_NAMES, rmsre
from tomato_rgbd.geometry import scene, FEATURES as GEO

rows = load_development_manifest(ROOT)
cal = read_json(ROOT / "data/raw/single_plant_cam/oak-d-s2-poe_conf.json")
out = []
for i, row in enumerate(rows, 1):
    f, _ = scene(ROOT / row["rgb_path"], ROOT / row["depth_path"], cal)
    out.append({"plant_id": row["plant_id"], **f})
    if i % 50 == 0:
        print("Geometry", i, "/383", flush=True)
g = pd.DataFrame(out)
g.to_csv(ROOT / "data/processed/geometry_development.csv", index=False)
base = pd.read_csv(ROOT / "data/processed/baseline_features_development.csv")
features = base.merge(g, on="plant_id", validate="one_to_one")
pred = []
for fold in range(5):
    tr, va = load_fold_dataframes(ROOT, fold)
    tr = tr.merge(features, on="plant_id", validate="one_to_one")
    va = va.merge(features, on="plant_id", validate="one_to_one")
    for variant, cols in [
        ("baseline", FEATURES),
        ("baseline_geometry", FEATURES + GEO),
    ]:
        p = np.zeros((len(va), 4))
        for k, t in enumerate(TARGET_NAMES):
            valid = tr[t].notna()
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                HistGradientBoostingRegressor(
                    max_iter=150,
                    max_leaf_nodes=7,
                    l2_regularization=1.0,
                    learning_rate=0.05,
                    random_state=42,
                ),
            )
            model.fit(tr.loc[valid, cols], tr.loc[valid, t])
            p[:, k] = np.maximum(model.predict(va[cols]), 0)
        for i, id in enumerate(va.plant_id):
            prediction = {
                "plant_id": id,
                "fold": fold,
                "model": variant,
            }
            for k, t in enumerate(TARGET_NAMES):
                prediction[t] = p[i, k]
            pred.append(prediction)
p = pd.DataFrame(pred)
p.to_csv(ROOT / "reports/geometry_oof.csv", index=False)
truth = pd.DataFrame(rows)[["plant_id", *TARGET_NAMES]]
scores = {}
for name in ["baseline", "baseline_geometry"]:
    j = truth.merge(
        p[p.model == name],
        on="plant_id",
        validate="one_to_one",
        suffixes=("_true", "_pred"),
    )
    assert len(j) == 383
    true_columns = []
    predicted_columns = []
    for target_name in TARGET_NAMES:
        true_columns.append(target_name + "_true")
        predicted_columns.append(target_name + "_pred")
    score = rmsre(j[true_columns].to_numpy(), j[predicted_columns].to_numpy())
    scores[name] = {"mean": score.mean, "per_target": score.per_target}
report = {
    "scope": "development only; same saved folds; no holdout",
    "scores": scores,
    "plane_valid": int(g.plane_ok.sum()),
    "total": len(g),
    "median_plane_coverage_blocks": float(g.plane_coverage.median()),
    "limitations": [
        "Color mask and plane are imperfect",
        "Projection includes visible segmented pixels only",
        "Height is above tray, not substrate",
        "Envelope is not tissue volume",
    ],
}
(ROOT / "reports/geometry_scores.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
