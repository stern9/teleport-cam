"""MediaPipe wrappers: person segmentation, hand landmarks, and the background plate."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .gestures import classify

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

ANALYSIS_WIDTH = 640  # models run on a downscaled copy of each frame


def model_path(name: str) -> str:
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        tmp.rename(path)
    return str(path)


class Analyzer:
    """Runs segmentation and hand tracking on each frame."""

    def __init__(self) -> None:
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path("selfie_segmenter.tflite")),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._prev_mask: np.ndarray | None = None
        self._last_ts = -1

    def analyze(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray, set[str], list]:
        """Return (person mask at frame resolution, detected gestures, raw hand landmarks)."""
        ts_ms = max(ts_ms, self._last_ts + 1)  # MediaPipe needs strictly increasing timestamps
        self._last_ts = ts_ms

        h, w = frame.shape[:2]
        sw = min(ANALYSIS_WIDTH, w)
        small = cv2.resize(frame, (sw, int(h * sw / w)), interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        conf = self.segmenter.segment_for_video(image, ts_ms).confidence_masks[0].numpy_view()
        mask = self._refine(np.squeeze(conf).astype(np.float32), (w, h))

        hands = self.hands.detect_for_video(image, ts_ms).hand_landmarks
        gestures = {g for lm in hands if (g := classify([(p.x, p.y) for p in lm]))}
        return mask, gestures, hands

    def _refine(self, conf: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        # Treat even low-confidence pixels as person: a half-hidden body reads as a ghost.
        m = np.clip((conf - 0.12) / 0.25, 0.0, 1.0)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if self._prev_mask is not None and self._prev_mask.shape == m.shape:
            # Decay rather than average, so a moving person never leaves a faded trail.
            m = np.maximum(m, self._prev_mask * 0.6)
        self._prev_mask = m
        m = cv2.GaussianBlur(m, (0, 0), 2.0)
        return cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        self.segmenter.close()
        self.hands.close()


class BackgroundPlate:
    """An image of the scene without the person.

    Captured once while the user steps out of frame and then left untouched:
    updating only the parts away from the person leaves a stale person-shaped
    region behind. Lighting drift is handled by `matched()` instead.
    """

    DIFF_SCALE = 4  # change detection runs at 1/4 resolution
    GAIN_SCALE = 8  # lighting correction runs at 1/8 resolution

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self._samples: list[np.ndarray] = []
        self._gain: np.ndarray | None = None  # last lighting correction, reused if no background is visible

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // self.DIFF_SCALE, h // self.DIFF_SCALE), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (0, 0), 1.5)

    def matched(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The plate, relit to match the live frame.

        Two things make the plate disagree with the live video: the webcam's
        auto-exposure / white balance, and the person themselves, who blocks light
        and casts a soft shadow on the wall around them. Either one shows up as a
        silhouette when the person vanishes. So the correction is measured right at
        the edge of the person and spread smoothly inward, falling back to the
        overall scene correction deep inside large regions.
        """
        h, w = frame.shape[:2]
        size = (w // self.GAIN_SCALE, h // self.GAIN_SCALE)
        f = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        p = cv2.resize(self.image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        log_r = np.log(np.clip((f + 4) / (p + 4), 0.4, 2.5))

        # Measure everywhere except on the person, and except big changes that
        # are real scene differences (moved objects) rather than lighting.
        person = cv2.dilate((cv2.resize(mask, size) > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8))
        valid = ((person == 0) & (np.abs(log_r).max(axis=2) < 0.5)).astype(np.float32)

        if valid.mean() > 0.02:
            local_sigma, wide_sigma = max(size) / 40, max(size) / 5
            w_local = cv2.GaussianBlur(valid, (0, 0), local_sigma)
            w_wide = cv2.GaussianBlur(valid, (0, 0), wide_sigma) + 1e-4
            num = log_r * valid[..., None]
            g_local = cv2.GaussianBlur(num, (0, 0), local_sigma) / (w_local[..., None] + 1e-4)
            g_wide = cv2.GaussianBlur(num, (0, 0), wide_sigma) / w_wide[..., None]
            # Trust the local estimate near measured pixels, the wide one far from them.
            alpha = np.clip(w_local / 0.3, 0.0, 1.0)[..., None]
            gain = alpha * g_local + (1 - alpha) * g_wide
            gain = cv2.GaussianBlur(gain, (0, 0), local_sigma)
            self._gain = np.exp(gain)
        elif self._gain is None or self._gain.shape[:2] != (size[1], size[0]):
            self._gain = np.ones((size[1], size[0], 3), np.float32)

        gain = cv2.resize(self._gain, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(self.image * gain, 0, 255).astype(np.uint8)

    def difference(self, frame: np.ndarray, plate: np.ndarray | None = None) -> np.ndarray:
        """Soft mask of everything that differs from the plate: the person, their shadow,
        and anything they hold. Returns zeros if there's no plate yet."""
        h, w = frame.shape[:2]
        plate = self.image if plate is None else plate
        if plate is None or plate.shape != frame.shape:
            return np.zeros((h, w), np.float32)
        d = cv2.absdiff(self._shrink(frame), self._shrink(plate)).max(axis=2).astype(np.float32)
        m = np.clip((d - 18.0) / 18.0, 0.0, 1.0)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop sensor noise
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # fill holes
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))  # cover the soft edge so no fringe shows
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    def add_sample(self, frame: np.ndarray) -> None:
        self._samples.append(frame.copy())

    def discard_samples(self) -> None:
        self._samples.clear()

    def finish_capture(self) -> None:
        if self._samples:
            self.image = np.median(np.stack(self._samples[-15:]), axis=0).astype(np.uint8)
            self._gain = None
        self._samples.clear()
