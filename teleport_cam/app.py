"""Main loop: camera in, effects applied, preview window + virtual camera out."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .effects import EFFECTS, Effect
from .gestures import HoldDetector
from .overlay import draw_panels, fit_to_window
from .sound import SoundPlayer
from .vision import Analyzer, BackgroundPlate

WINDOW = "Teleport Cam"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
COUNTDOWN_S = 4.0
CAPTURE_S = 1.0
COOLDOWN_S = 1.0

VISIBLE, VANISHING, GONE, APPEARING = "visible", "vanishing", "gone", "appearing"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="teleport-cam", description="Gesture-triggered teleport effects for your webcam.")
    p.add_argument("--camera", type=int, default=0, help="camera index (see --list-cameras)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--effect", choices=list(EFFECTS), default="beam")
    p.add_argument("--no-vcam", action="store_true", help="preview only, don't send to the virtual camera")
    p.add_argument("--no-sound", action="store_true", help="don't play teleport sounds")
    p.add_argument("--volume", type=float, default=1.0, help="sound volume, 0.0-1.0 (default 1.0)")
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the preview window")
    p.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    p.add_argument("--debug", action="store_true", help="draw hand landmarks in the preview")
    return p.parse_args(argv)


def list_cameras() -> None:
    out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True).stdout
    names = [
        line.strip().rstrip(":") for line in out.splitlines() if line.startswith("    ") and line.strip().endswith(":")
    ]
    print("Cameras (OpenCV usually numbers them in this order, starting at 0):")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")


def open_vcam(width: int, height: int, fps: int):
    try:
        import pyvirtualcam

        cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
        print(f"Virtual camera ready: {cam.device}  -> choose it in Zoom / Teams / Slack")
        return cam
    except Exception as e:  # noqa: BLE001 - any failure just means preview-only
        print(f"Virtual camera unavailable ({e}).")
        print("  Install OBS (brew install --cask obs), open it once, click 'Start Virtual Camera'")
        print("  and approve the system extension, then click 'Stop Virtual Camera' and rerun.")
        print("  Continuing with preview only.")
        return None


class TeleportCam:
    def __init__(
        self,
        args: argparse.Namespace,
        analyzer: Analyzer | None = None,
        sound: SoundPlayer | None = None,
    ) -> None:
        self.args = args
        self.rng = np.random.default_rng()
        # Injectable so tests can run the state machine without loading ML models.
        self.analyzer = analyzer if analyzer is not None else Analyzer()
        self.plate = BackgroundPlate()
        self.effect: Effect = EFFECTS[args.effect]()
        self.sound = sound if sound is not None else SoundPlayer(enabled=not args.no_sound, volume=args.volume)
        self.state = VISIBLE
        self.state_t0 = 0.0
        self.cooldown_until = 0.0
        self.capture_t0: float | None = None
        self.vanish_gesture = HoldDetector("v_sign", 0.25)
        self.appear_gesture = HoldDetector("open_palm", 0.8)
        self.gestures: set[str] = set()
        self.show_hud = False
        self.match_lighting = True
        self.last_debug: dict[str, np.ndarray] = {}
        self.last_frame: np.ndarray | None = None
        self.last_out: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.fullscreen = False

    # ---- state changes -------------------------------------------------

    def begin_capture(self, now: float) -> None:
        self.capture_t0 = now
        self.state = VISIBLE
        print(f"Step out of the frame - capturing the background in {COUNTDOWN_S:.0f}s...")

    def vanish(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        if self.state != VISIBLE or self.plate.image is None or self.capture_t0 is not None:
            return
        self.effect.start(frame, mask, self.rng)
        self.sound.play(self.effect.name, "vanish")
        self.state, self.state_t0 = VANISHING, now

    def appear(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        if self.state != GONE:
            return
        self.effect.start(frame, mask, self.rng)
        self.sound.play(self.effect.name, "appear")
        self.state, self.state_t0 = APPEARING, now

    def set_effect(self, name: str) -> None:
        if self.state in (VANISHING, APPEARING):
            return
        self.effect = EFFECTS[name]()
        print(f"Effect: {self.effect.label}")

    # ---- per-frame -----------------------------------------------------

    def step_capture(self, frame: np.ndarray, mask: np.ndarray, now: float) -> str | None:
        """Advance the background capture; returns a preview message while it runs."""
        elapsed = now - self.capture_t0
        if elapsed >= COUNTDOWN_S - 1.0 and person_in_frame(mask):
            # Any part of the person in the plate shows up later as a ghost, so wait.
            if elapsed >= COUNTDOWN_S:
                print("Still see you - step fully out of frame.")
            self.plate.discard_samples()
            self.capture_t0 = now - (COUNTDOWN_S - 1.0)
            return "I can still see you - step fully out of frame"
        if elapsed < COUNTDOWN_S:
            return f"Step out of frame... {COUNTDOWN_S - elapsed:.0f}"
        if elapsed < COUNTDOWN_S + CAPTURE_S:
            self.plate.add_sample(frame)
            return "Capturing background..."
        self.plate.finish_capture()
        self.capture_t0 = None
        print("Background captured. Step back in. V sign = vanish (and again to come back).")
        return None

    def render(self, frame: np.ndarray, mask: np.ndarray, gestures: set[str], now: float) -> np.ndarray:
        if now >= self.cooldown_until:
            if self.vanish_gesture.update(gestures, now):
                # V sign toggles: vanish when visible, come back when gone.
                if self.state == GONE:
                    self.appear(frame, mask, now)
                else:
                    self.vanish(frame, mask, now)
            if self.appear_gesture.update(gestures, now):
                self.appear(frame, mask, now)

        if self.state == VISIBLE:
            return frame
        bg = self.plate.matched(frame, mask) if self.match_lighting else self.plate.image
        # Hide the person plus anything else that differs from the empty background
        # (their shadow, partially-detected limbs), so no ghost is left behind.
        seg = mask
        mask = np.maximum(mask, self.plate.difference(frame, bg))
        self.last_debug = {"plate_matched": bg, "segmentation": seg, "hide_mask": mask}
        if self.state == VANISHING:
            p = (now - self.state_t0) / self.effect.duration
            if p >= 1.0:
                self.state, self.cooldown_until = GONE, now + COOLDOWN_S
            else:
                return self.effect.render(frame, bg, mask, p)
        if self.state == APPEARING:
            p = 1.0 - (now - self.state_t0) / self.effect.duration
            if p <= 0.0:
                self.state, self.cooldown_until = VISIBLE, now + COOLDOWN_S
            else:
                return self.effect.render(frame, bg, mask, p)
        if self.state == GONE:
            r = mask[..., None]
            return (frame * (1 - r) + bg * r).astype(np.uint8)
        return frame

    def draw_preview(self, out: np.ndarray, hands: list, message: str | None, vcam_on: bool, now: float) -> np.ndarray:
        view = out.copy()
        h, w = view.shape[:2]
        if self.args.debug:
            for lm in hands:
                for pt in lm:
                    cv2.circle(view, (int(pt.x * w), int(pt.y * h)), 3, (0, 255, 0), -1)
        if not self.args.no_mirror:
            view = cv2.flip(view, 1)
        if not message and not self.show_hud:
            return view
        view = fit_to_window(view, WINDOW)
        lines: list[str] = []
        hold = 0.0
        if self.show_hud:
            hold = max(self.vanish_gesture.progress(now), self.appear_gesture.progress(now))
            parts = [
                self.effect.label,
                self.state,
                f"camera {'on' if vcam_on else 'off'}",
                f"sound {'on' if self.sound.enabled else 'off'}",
                f"light match {'on' if self.match_lighting else 'off'}",
            ]
            if self.gestures:
                parts.append("sees " + ", ".join(sorted(g.replace("_", " ") for g in self.gestures)))
            if self.plate.image is None and self.capture_t0 is None:
                parts.append("press B to capture background")
            lines = [
                " · ".join(parts),
                "1-3 effect   V vanish   R reappear   B background   S sound   M light   D debug   H hide   Q quit",
            ]
        return draw_panels(view, lines, message, hold)

    # ---- main loop -----------------------------------------------------

    def run(self) -> int:
        args = self.args
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        ok, frame = cap.read()
        if not ok:
            print(f"Could not read from camera {args.camera}.")
            print("  Check System Settings > Privacy & Security > Camera and allow your terminal app,")
            print("  or try another --camera index (see --list-cameras).")
            return 1
        h, w = frame.shape[:2]
        print(f"Camera {args.camera}: {w}x{h}")

        vcam = None if args.no_vcam else open_vcam(w, h, args.fps)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(w, 960), int(min(w, 960) * h / w))
        t_start = time.monotonic()
        self.begin_capture(t_start)
        print("Minimize the preview to keep it running in the background.")
        print("Keys (click the preview window first): 1-3 effect, V vanish, R reappear, B recapture background,")
        print("  S sound, M lighting match on/off, H show status, D save debug snapshot, F fullscreen, Q quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera stopped delivering frames.")
                    break
                now = time.monotonic()
                mask, gestures, hands = self.analyzer.analyze(frame, int((now - t_start) * 1000))
                self.gestures = gestures

                message = self.step_capture(frame, mask, now) if self.capture_t0 is not None else None
                out = self.render(frame, mask, gestures, now)

                if vcam is not None:
                    vcam.send(out)
                cv2.imshow(WINDOW, self.draw_preview(out, hands, message, vcam is not None, now))

                self.last_frame, self.last_out, self.last_mask = frame, out, mask
                if not self.handle_key(cv2.waitKey(1) & 0xFF, frame, mask, now):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if vcam is not None:
                vcam.close()
            self.analyzer.close()
            self.sound.stop()
            cv2.destroyAllWindows()
        return 0

    def save_debug(self) -> None:
        """Dump what the app currently sees, for diagnosing ghosts or bad masks."""
        out_dir = DEBUG_DIR / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.last_frame is None:
            return
        images = {"frame": self.last_frame, "output": self.last_out, "segmentation": self.last_mask}
        if self.plate.image is not None:
            images["plate"] = self.plate.image
        if self.state != VISIBLE:
            images.update(self.last_debug)
        for name, img in images.items():
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{name}.png"), img)
        print(f"Debug snapshot saved to {out_dir}")

    def handle_key(self, key: int, frame: np.ndarray, mask: np.ndarray, now: float) -> bool:
        if key in (ord("q"), 27):
            return False
        names = list(EFFECTS)
        if ord("1") <= key < ord("1") + len(names):
            self.set_effect(names[key - ord("1")])
        elif key == ord("v"):
            self.vanish(frame, mask, now)
        elif key == ord("r"):
            self.appear(frame, mask, now)
        elif key == ord("b"):
            self.begin_capture(now)
        elif key == ord("s"):
            print(f"Sound {'on' if self.sound.toggle() else 'off'}")
        elif key == ord("m"):
            self.match_lighting = not self.match_lighting
            print(f"Lighting match {'on' if self.match_lighting else 'off'}")
        elif key == ord("d"):
            self.save_debug()
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            )
        return True


def person_in_frame(mask: np.ndarray) -> bool:
    return float((mask[::8, ::8] > 0.5).mean()) > 0.003


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        list_cameras()
        return 0
    return TeleportCam(args).run()


if __name__ == "__main__":
    sys.exit(main())
