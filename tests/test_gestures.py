from teleport_cam.gestures import HoldDetector, classify

WRIST = (0.5, 0.9)
FINGER_X = {"index": 0.44, "middle": 0.5, "ring": 0.56, "pinky": 0.62}


def hand(extended, spread=0.04, rotate=False):
    """Synthetic upright hand: 21 (x, y) landmarks."""
    pts = [WRIST, (0.42, 0.85), (0.38, 0.8), (0.34, 0.76), (0.30, 0.72)]
    for name, x in FINGER_X.items():
        if name in ("index", "middle") and name in extended:
            x += -spread if name == "index" else spread
        if name in extended:
            pts += [(FINGER_X[name], 0.7), (x, 0.6), (x, 0.5), (x, 0.4)]
        else:
            pts += [(x, 0.7), (x, 0.62), (x, 0.68), (x, 0.72)]
    if rotate:  # hand lying sideways: swap axes
        pts = [(y, x) for x, y in pts]
    return pts


def test_v_sign():
    assert classify(hand({"index", "middle"})) == "v_sign"
    assert classify(hand({"index", "middle"}, rotate=True)) == "v_sign"


def test_fingers_together_is_not_v():
    assert classify(hand({"index", "middle"}, spread=-0.03)) is None


def test_open_palm():
    assert classify(hand({"index", "middle", "ring", "pinky"})) == "open_palm"


def test_fist_and_point_are_nothing():
    assert classify(hand(set())) is None
    assert classify(hand({"index"})) is None


def test_hold_detector_fires_once_after_hold():
    d = HoldDetector("v_sign", 0.5)
    assert not d.update({"v_sign"}, 0.0)
    assert not d.update({"v_sign"}, 0.3)
    assert d.update({"v_sign"}, 0.6)
    assert not d.update({"v_sign"}, 1.0)
    assert not d.update(set(), 1.1)
    assert not d.update({"v_sign"}, 1.2)
    assert d.update({"v_sign"}, 1.8)
