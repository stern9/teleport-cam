"""Preview-window text: antialiased system font on translucent panels.

OpenCV's built-in Hershey fonts are jagged, and drawing them at camera resolution
before the window scales the image down makes them worse. Text here is drawn with
Pillow after the frame has been resized to the window's on-screen size.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
)


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def fit_to_window(img: np.ndarray, window: str) -> np.ndarray:
    """Resize to the window's current on-screen size so overlay text isn't rescaled."""
    try:
        _, _, w, h = cv2.getWindowImageRect(window)
    except cv2.error:
        return img
    if w <= 0 or h <= 0 or (w, h) == (img.shape[1], img.shape[0]):
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def draw_panels(
    img: np.ndarray,
    lines: list[str],
    message: str | None = None,
    progress: float = 0.0,
) -> np.ndarray:
    """Status lines bottom-left, an optional big centered message, and a gesture-hold bar."""
    if not lines and not message:
        return img
    h, w = img.shape[:2]
    scale = max(0.6, min(1.6, w / 960))
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if lines:
        f = font(round(15 * scale))
        pad, gap = round(10 * scale), round(5 * scale)
        boxes = [draw.textbbox((0, 0), line, font=f) for line in lines]
        line_h = max(b[3] - b[1] for b in boxes)
        bar_h = round(4 * scale) + gap if progress > 0 else 0
        panel_w = max(b[2] - b[0] for b in boxes) + 2 * pad
        panel_h = len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad + bar_h
        x0, y0 = round(12 * scale), h - panel_h - round(12 * scale)
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=round(8 * scale), fill=(0, 0, 0, 150))
        y = y0 + pad
        if progress > 0:
            bar_w = panel_w - 2 * pad
            draw.rounded_rectangle(
                (x0 + pad, y, x0 + pad + bar_w, y + round(4 * scale)), radius=2, fill=(255, 255, 255, 60)
            )
            draw.rounded_rectangle(
                (x0 + pad, y, x0 + pad + round(bar_w * progress), y + round(4 * scale)),
                radius=2,
                fill=(120, 230, 255, 255),
            )
            y += bar_h
        for i, (line, box) in enumerate(zip(lines, boxes, strict=True)):
            color = (255, 255, 255, 255) if i == 0 else (255, 255, 255, 170)
            draw.text((x0 + pad, y - box[1]), line, font=f, fill=color)
            y += line_h + gap

    if message:
        f = font(round(28 * scale))
        box = draw.textbbox((0, 0), message, font=f)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = round(16 * scale)
        x0, y0 = (w - tw) // 2 - pad, (h - th) // 2 - pad
        draw.rounded_rectangle(
            (x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad), radius=round(12 * scale), fill=(0, 0, 0, 160)
        )
        draw.text((x0 + pad, y0 + pad - box[1]), message, font=f, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(base, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
