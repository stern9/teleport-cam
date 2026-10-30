"""Hand-pose classification from MediaPipe's 21 hand landmarks.

All checks use distances between landmarks rather than absolute directions,
so they work regardless of mirroring, hand rotation or which hand is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# MediaPipe hand landmark indices
WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extended(pts: Sequence[Point], pip: int, tip: int) -> bool:
    # A straight finger's tip is well beyond its middle joint, measured from the wrist.
    return _dist(pts[WRIST], pts[tip]) > _dist(pts[WRIST], pts[pip]) * 1.15


def finger_states(pts: Sequence[Point]) -> dict[str, bool]:
    return {
        "index": _extended(pts, INDEX_PIP, INDEX_TIP),
        "middle": _extended(pts, MIDDLE_PIP, MIDDLE_TIP),
        "ring": _extended(pts, RING_PIP, RING_TIP),
        "pinky": _extended(pts, PINKY_PIP, PINKY_TIP),
    }


def classify(pts: Sequence[Point]) -> str | None:
    """Return "v_sign", "open_palm" or None for one hand's landmarks."""
    if len(pts) < 21:
        return None
    f = finger_states(pts)
    palm = _dist(pts[WRIST], pts[MIDDLE_MCP]) or 1e-6
    spread = _dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) / palm

    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"] and spread > 0.25:
        return "v_sign"
    if all(f.values()):
        return "open_palm"
    return None


@dataclass
class HoldDetector:
    """Fires once when a gesture has been held continuously for `hold` seconds."""

    gesture: str
    hold: float
    _since: float | None = None
    _fired: bool = False

    def update(self, gestures: set[str], now: float) -> bool:
        if self.gesture not in gestures:
            self._since = None
            self._fired = False
            return False
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.hold:
            self._fired = True
            return True
        return False

    def progress(self, now: float) -> float:
        if self._since is None or self._fired:
            return 0.0
        return min(1.0, (now - self._since) / self.hold)
