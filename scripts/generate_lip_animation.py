#!/usr/bin/env python3
"""Generate a looping glossy lip animation GIF from the final glossy render."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build" / "glossy.png"
OUTPUT = ROOT / "build" / "final_glossy_kiss.gif"


def lip_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    pixels = image.convert("RGB")
    width, height = pixels.size
    xs: list[int] = []
    ys: list[int] = []

    for y in range(height):
        for x in range(width):
            r, g, b = pixels.getpixel((x, y))
            if r > 40 or g > 25 or b > 25:
                xs.append(x)
                ys.append(y)

    if not xs:
        raise RuntimeError(f"Could not find lip pixels in {SOURCE}")

    pad = 10
    return (
        max(min(xs) - pad, 0),
        max(min(ys) - pad, 0),
        min(max(xs) + pad, width),
        min(max(ys) + pad, height),
    )


def make_frame(base: Image.Image, crop: Image.Image, bbox: tuple[int, int, int, int],
               frame_index: int, frame_count: int) -> Image.Image:
    width, height = base.size
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    lip_w = x1 - x0
    lip_h = y1 - y0

    angle = 2.0 * math.pi * frame_index / frame_count
    pucker = 0.5 - 0.5 * math.cos(angle)
    release = 1.0 - pucker

    scale_x = 1.0 - 0.22 * pucker
    scale_y = 1.0 + 0.12 * pucker
    new_w = max(1, int(lip_w * scale_x))
    new_h = max(1, int(lip_h * scale_y))

    resized = crop.resize((new_w, new_h), Image.Resampling.BICUBIC)
    frame = Image.new("RGB", (width, height), (0, 0, 0))
    frame.paste(resized, (cx - new_w // 2, cy - new_h // 2))

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    crease_w = int((lip_w * (0.76 - 0.18 * pucker)))
    crease_h = max(3, int(5 + 11 * pucker))
    crease_y = int(cy - lip_h * 0.03)
    draw.ellipse(
        (
            cx - crease_w // 2,
            crease_y - crease_h // 2,
            cx + crease_w // 2,
            crease_y + crease_h // 2,
        ),
        fill=(35, 6, 14, int(95 + 70 * pucker)),
    )

    shine_alpha = int(120 + 55 * release)
    draw.ellipse(
        (
            int(cx - lip_w * (0.23 - 0.04 * pucker)),
            int(cy - lip_h * (0.35 + 0.02 * pucker)),
            int(cx - lip_w * (0.07 - 0.04 * pucker)),
            int(cy - lip_h * (0.28 + 0.02 * pucker)),
        ),
        fill=(255, 255, 255, shine_alpha),
    )
    draw.ellipse(
        (
            int(cx + lip_w * (0.08 + 0.03 * pucker)),
            int(cy - lip_h * (0.33 + 0.01 * pucker)),
            int(cx + lip_w * (0.20 + 0.03 * pucker)),
            int(cy - lip_h * (0.26 + 0.01 * pucker)),
        ),
        fill=(255, 255, 255, int(0.85 * shine_alpha)),
    )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.0))
    frame = Image.alpha_composite(frame.convert("RGBA"), overlay)
    return frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)


def main() -> None:
    base = Image.open(SOURCE).convert("RGB")
    bbox = lip_bbox(base)
    crop = base.crop(bbox)
    frame_count = 28
    frames = [make_frame(base, crop, bbox, i, frame_count) for i in range(frame_count)]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=65,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
