"""Show the first saved development plant of each variety, never holdout examples."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomato_rgbd.data import load_development_manifest


def main():
    rows = load_development_manifest(ROOT)
    selected = [next(r for r in rows if r["variety"] == variety) for variety in "ABCD"]
    tile = (400, 225)
    canvas = Image.new("RGB", (1200, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((12, 8), "Development only | RGB / Depth: blue=0.30 m, red=2.00 m / Valid depth", fill="black", font=font)
    for index, row in enumerate(selected):
        with Image.open(ROOT / row["rgb_path"]) as image:
            rgb = image.convert("RGB").resize(tile)
        with Image.open(ROOT / row["depth_path"]) as image:
            depth = np.asarray(image, dtype=np.float32)
        valid = depth > 0
        t = np.clip((depth - 300) / 1700, 0, 1)
        colors = np.stack([t, 1 - np.abs(2 * t - 1), 1 - t], axis=-1)
        colors[~valid] = 0
        colored = Image.fromarray((colors * 255).astype(np.uint8)).resize(tile, Image.Resampling.NEAREST)
        mask = Image.fromarray((valid * 255).astype(np.uint8)).convert("RGB").resize(tile, Image.Resampling.NEAREST)
        y = 55 + 265 * index
        draw.text((12, y - 23), row["plant_id"] + " | white = valid; black = invalid", fill="black", font=font)
        for col, image in enumerate((rgb, colored, mask)):
            canvas.paste(image, (400 * col, y))
    output = ROOT / "reports/development_rgb_depth.png"
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
