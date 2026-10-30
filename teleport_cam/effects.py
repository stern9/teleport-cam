"""Transition effects.

Every effect maps a progress value p to an output frame: p=0 means the person is
fully visible, p=1 means they are fully replaced by the background plate.
Reappearing runs the same effect with p going from 1 back to 0.

Frames are BGR uint8; masks are float32 HxW in [0, 1] (1 = person).
"""

from __future__ import annotations

import cv2
import numpy as np

CYAN = np.array([255, 230, 120], np.float32)  # BGR
WHITE_CYAN = np.array([255, 250, 220], np.float32)


def composite(frame: np.ndarray, bg: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """Show `bg` where removed=1 and the live frame elsewhere."""
    r = removed[..., None]
    return frame.astype(np.float32) * (1.0 - r) + bg.astype(np.float32) * r


def fractal_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    for cells, weight in ((6, 0.5), (14, 0.3), (36, 0.2)):
        gh, gw = cells, max(2, int(cells * w / max(h, 1)))
        grid = rng.random((gh, gw)).astype(np.float32)
        out += weight * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


def mask_bbox(mask: np.ndarray, thresh: float = 0.5) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the person, or None if nobody is in frame."""
    small = mask[::4, ::4] > thresh
    ys, xs = np.nonzero(small)
    if len(xs) < 20:
        return None
    return int(xs.min()) * 4, int(ys.min()) * 4, int(xs.max()) * 4 + 4, int(ys.max()) * 4 + 4


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


class Effect:
    name = "base"
    label = "Base"
    duration = 1.5

    def start(self, frame: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
        """Called once at the start of each transition with the current frame."""

    def render(self, frame: np.ndarray, bg: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
        raise NotImplementedError


class Dissolve(Effect):
    """The person burns away in organic patches with a glowing edge."""

    name = "dissolve"
    label = "Dissolve"
    duration = 1.6

    def start(self, frame, mask, rng):
        h, w = mask.shape
        self.noise = fractal_noise(h, w, rng)

    def render(self, frame, bg, mask, p):
        t = p * 1.3 - 0.15
        d = t - self.noise
        removed = np.clip(d / 0.06, 0.0, 1.0) * mask
        edge = np.exp(-((d / 0.035) ** 2)) * mask * (p > 0.0)
        out = composite(frame, bg, removed)
        out += edge[..., None] * CYAN * 0.9
        return _to_u8(out)


class Beam(Effect):
    """Sci-fi teleport: scanline shimmer, a column of light, dematerialize top-down."""

    name = "beam"
    label = "Teleport beam"
    duration = 1.9

    def start(self, frame, mask, rng):
        h, w = mask.shape
        self.rng = rng
        self.row_noise = rng.random(h).astype(np.float32)
        self.row_noise = cv2.GaussianBlur(self.row_noise[:, None], (1, 0), 1.5).ravel()
        # Beam and sparkles follow the person's silhouette at the start; the render
        # mask may also cover their shadow, which shouldn't sparkle.
        self.bbox = mask_bbox(mask) or (w // 3, h // 6, 2 * w // 3, h)
        self.sparkle_ys, self.sparkle_xs = np.nonzero(mask[::4, ::4] > 0.5)
        self.ys = np.arange(h, dtype=np.float32)[:, None]
        self.xs = np.arange(w, dtype=np.float32)[None, :]

    def render(self, frame, bg, mask, p):
        h, w = mask.shape
        x0, y0, x1, y1 = self.bbox
        bh, bw = max(y1 - y0, 1), max(x1 - x0, 1)
        energy = float(np.sin(np.pi * p))

        # Removal front sweeps top to bottom, broken up into scanlines.
        yfrac = np.clip((self.ys - y0) / bh, 0.0, 1.0)
        thresh = 0.75 * yfrac + 0.25 * self.row_noise[:, None]
        t = p * 1.4 - 0.2
        d = t - thresh  # (h, 1): one value per row
        core = np.clip(d / 0.05, 0.0, 1.0)
        flicker = (self.rng.random((h, 1)) > 0.6).astype(np.float32) * 0.4 * energy
        removed = np.clip(core + flicker, 0.0, 1.0) * mask

        # Tint what's left of the person toward cyan as the energy builds.
        tinted = frame.astype(np.float32)
        tint = (mask * 0.45 * energy)[..., None]
        tinted = tinted * (1 - tint) + CYAN * tint
        out = composite(tinted, bg, removed)

        # Glowing front and vertical light column.
        edge = np.exp(-((d / 0.03) ** 2)) * mask * (p > 0.0)
        cx = (x0 + x1) / 2
        col = np.exp(-(((self.xs - cx) / (bw * 0.4)) ** 2))
        core_col = np.exp(-(((self.xs - cx) / (bw * 0.08)) ** 2))
        vert = np.clip((self.ys - (y0 - 0.15 * bh)) / (0.1 * bh), 0, 1) * np.clip(
            (y1 + 0.05 * bh - self.ys) / (0.1 * bh), 0, 1
        )
        glow = (col * 0.55 + core_col * 0.6) * vert * energy**1.5
        out += (edge * 0.9)[..., None] * CYAN + glow[..., None] * WHITE_CYAN

        # Sparkles scattered over the body.
        n = int(500 * energy)
        if n:
            ys, xs = self.sparkle_ys, self.sparkle_xs
            if len(xs):
                pick = self.rng.integers(0, len(xs), n)
                px = np.clip(xs[pick] * 4 + self.rng.integers(0, 4, n), 0, w - 1)
                py = np.clip(ys[pick] * 4 + self.rng.integers(0, 4, n), 0, h - 1)
                out[py, px] = WHITE_CYAN * 1.0
                out[np.clip(py + 1, 0, h - 1), px] = WHITE_CYAN * 0.8
        return _to_u8(out)


class Particles(Effect):
    """The person breaks into glowing fragments carried away on the wind."""

    name = "particles"
    label = "Particles"
    duration = 2.2
    STRIDE = 4
    LIFE = 0.45  # fraction of the effect each particle lives for

    def start(self, frame, mask, rng):
        h, w = mask.shape
        x0, _, x1, _ = mask_bbox(mask) or (0, 0, w, h)
        xfrac = np.clip((np.arange(w, dtype=np.float32) - x0) / max(x1 - x0, 1), 0, 1)
        noise = fractal_noise(h, w, rng)
        # Release sweeps left to right with organic variation.
        self.delay = (0.55 * (0.6 * xfrac[None, :] + 0.4 * noise)).astype(np.float32)

        s = self.STRIDE
        ys, xs = np.nonzero(mask[::s, ::s] > 0.5)
        ys, xs = ys * s, xs * s
        n = len(xs)
        self.x = xs.astype(np.float32)
        self.y = ys.astype(np.float32)
        self.color = np.clip(frame[ys, xs].astype(np.float32) * 1.15 + 25, 0, 255)
        self.pdelay = self.delay[ys, xs]
        self.vx = rng.uniform(120, 360, n).astype(np.float32)
        self.vy = rng.uniform(-200, -30, n).astype(np.float32)
        self.size = rng.integers(1, 3, n)

    def render(self, frame, bg, mask, p):
        h, w = mask.shape
        removed = np.clip((p - self.delay) / 0.04, 0.0, 1.0) * mask
        out = composite(frame, bg, removed)

        age = np.clip((p - self.pdelay) / self.LIFE, 0.0, 1.0)
        alive = (age > 0) & (age < 1)
        if not alive.any():
            return _to_u8(out)
        a = age[alive]
        px = (self.x[alive] + self.vx[alive] * a + 40 * np.sin(a * 6 + self.y[alive] * 0.05)).astype(np.int32)
        py = (self.y[alive] + self.vy[alive] * a - 60 * a * a).astype(np.int32)
        alpha = ((1.0 - a) ** 1.2)[:, None]
        col = self.color[alive]
        size = self.size[alive]
        for dy in range(2):
            for dx in range(2):
                sel = (size > max(dx, dy)) | ((dx == 0) & (dy == 0))
                qx, qy = px[sel] + dx, py[sel] + dy
                ok = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
                qx, qy, al, c = qx[ok], qy[ok], alpha[sel][ok], col[sel][ok]
                out[qy, qx] = out[qy, qx] * (1 - al) + c * al
        return _to_u8(out)


EFFECTS: dict[str, type[Effect]] = {e.name: e for e in (Beam, Dissolve, Particles)}
