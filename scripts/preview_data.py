"""Preview four development plants with OpenCV; holdout is excluded."""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tomato_rgbd.data import load_development_manifest


def main():
    rows = load_development_manifest(ROOT)
    selected = [next(r for r in rows if r['variety'] == variety) for variety in 'ABCD']
    canvas = np.full((1120, 1200, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Development | RGB / depth: blue=0.30 m, red=2.00 m / validity',
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    for index, row in enumerate(selected):
        rgb = cv2.imread(str(ROOT / row['rgb_path']), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(ROOT / row['depth_path']), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise ValueError(f"Cannot read {row['plant_id']}")
        valid = depth > 0
        t = np.clip((depth.astype(np.float32) - 300) / 1700, 0, 1)
        colors = np.stack([1 - t, 1 - np.abs(2*t - 1), t], axis=-1)
        colors[~valid] = 0
        colored = (colors * 255).astype(np.uint8)
        mask = cv2.cvtColor((valid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        y = 55 + 265 * index
        cv2.putText(canvas, row['plant_id'] + ' | white = valid', (12, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        for col, image in enumerate((rgb, colored, mask)):
            interpolation = cv2.INTER_AREA if col == 0 else cv2.INTER_NEAREST
            canvas[y:y+225, col*400:(col+1)*400] = cv2.resize(image, (400,225), interpolation=interpolation)
    output = ROOT / 'reports/development_rgb_depth.png'
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f'Cannot write {output}')
    print(output)


if __name__ == '__main__':
    main()
