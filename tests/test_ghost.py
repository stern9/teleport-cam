import numpy as np

from teleport_cam.app import APPEARING, GONE, VISIBLE, TeleportCam, parse_args
from teleport_cam.sound import SoundPlayer
from teleport_cam.vision import BackgroundPlate

H, W = 360, 640


def scene(rng):
    plate = np.zeros((H, W, 3), np.uint8)
    plate[:] = (90, 120, 150)
    plate[50:300, 40:200] = (40, 160, 200)  # a bookshelf or similar
    frame = plate.astype(np.int16)
    frame[80:360, 260:400] = (60, 80, 180)  # the person
    frame[120:360, 400:480] -= 45  # their shadow on the wall
    frame += rng.normal(0, 3, frame.shape).astype(np.int16)  # sensor noise
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    # Segmentation only half-sure about the person, and misses the shadow.
    seg = np.zeros((H, W), np.float32)
    seg[80:360, 260:400] = 0.5
    return plate, frame, seg


def test_difference_catches_person_and_shadow_but_not_noise():
    rng = np.random.default_rng(0)
    plate_img, frame, _ = scene(rng)
    plate = BackgroundPlate()
    plate.add_sample(plate_img)
    plate.finish_capture()
    d = plate.difference(frame)
    assert d[200, 330] > 0.9  # person
    assert d[250, 440] > 0.9  # shadow
    assert d[20, 600] < 0.05 and d[150, 100] < 0.05  # untouched background


def make_app(plate_img: np.ndarray, effect: str = "dissolve") -> TeleportCam:
    """A TeleportCam with a captured plate and no ML models or audio."""
    app = TeleportCam(
        parse_args(["--no-vcam", "--effect", effect]), analyzer=object(), sound=SoundPlayer(enabled=False)
    )
    app.plate.add_sample(plate_img)
    app.plate.finish_capture()
    return app


def test_gone_leaves_no_ghost():
    plate_img, frame, seg = scene(np.random.default_rng(1))
    app = make_app(plate_img)
    app.state = GONE
    out = app.render(frame, seg, set(), 10.0)
    err = np.abs(out.astype(int) - plate_img).max(axis=2)
    assert err[80:360, 260:480].mean() < 4  # person + shadow region looks like the plate
    assert np.percentile(err[80:360, 260:480], 99) < 15


def test_v_sign_toggles_back_from_gone():
    frame = np.zeros((H, W, 3), np.uint8)
    app = make_app(frame)
    app.state = GONE
    mask = np.zeros((H, W), np.float32)
    app.render(frame, mask, {"v_sign"}, 1.0)
    app.render(frame, mask, {"v_sign"}, 1.3)
    assert app.state == APPEARING
    app.render(frame, mask, set(), 5.0)
    assert app.state == VISIBLE


def test_exposure_change_leaves_no_silhouette():
    """Camera brightens and shifts white balance after capture: the hidden region must
    match the live surroundings, not the plate's old exposure."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:H, 0:W]
    plate_img = np.stack([80 + xx / 8, 100 + yy / 6, 120 + 0 * xx], axis=2).astype(np.uint8)
    gain = np.array([1.15, 1.12, 1.22])  # auto-exposure + white balance drift
    live_bg = np.clip(plate_img * gain, 0, 255)
    frame = live_bg.copy()
    frame[80:360, 260:400] = (60, 80, 180)  # the person
    frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
    seg = np.zeros((H, W), np.float32)
    seg[80:360, 260:400] = 1.0

    plate = BackgroundPlate()
    plate.add_sample(plate_img)
    plate.finish_capture()

    matched = plate.matched(frame, seg)
    region = (slice(80, 360), slice(260, 400))
    err_matched = np.abs(matched[region].astype(int) - live_bg[region]).mean()
    err_raw = np.abs(plate_img[region].astype(int) - live_bg[region]).mean()
    assert err_raw > 15  # the old behaviour: a visible patch
    assert err_matched < 3

    # And the brightness shift alone must not make the whole frame count as "changed".
    d = plate.difference(frame, matched)
    assert d[20:60, 20:200].mean() < 0.05


def test_person_shadow_leaves_no_outline():
    """The person darkens the wall around them (blocked light, soft shadow). With the
    person hidden, there must be no step in brightness at the edge of the hidden area."""
    import cv2

    plate_img = np.full((H, W, 3), 150, np.uint8)
    person = np.zeros((H, W), np.uint8)
    person[100:360, 250:390] = 1
    dist = cv2.distanceTransform(1 - person, cv2.DIST_L2, 5)
    shade = 1 - 0.15 * np.exp(-dist / 60)  # up to 15% darker right next to the person
    frame = (plate_img * shade[..., None]).astype(np.uint8)
    frame[person == 1] = (60, 80, 180)
    seg = person.astype(np.float32)

    plate = BackgroundPlate()
    plate.add_sample(plate_img)
    plate.finish_capture()
    bg = plate.matched(frame, seg)
    hide = np.maximum(seg, plate.difference(frame, bg))[..., None]
    out = (frame * (1 - hide) + bg * hide).astype(np.float32).mean(axis=2)

    row = out[230]
    inside, outside = row[245:252].mean(), row[225:232].mean()  # either side of the left edge
    assert abs(inside - outside) < 4
    assert out[200:300, 280:360].std() < 4  # hidden area is smooth, no person-shaped patch
