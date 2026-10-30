import numpy as np
import pytest

from teleport_cam.effects import EFFECTS, mask_bbox

H, W = 180, 320


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    frame = np.full((H, W, 3), 200, np.uint8)
    bg = np.full((H, W, 3), 40, np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[40:170, 120:200] = 1.0
    return frame, bg, mask, rng


@pytest.mark.parametrize("name", list(EFFECTS))
def test_start_shows_person(name, scene):
    frame, bg, mask, rng = scene
    effect = EFFECTS[name]()
    effect.start(frame, mask, rng)
    out = effect.render(frame, bg, mask, 0.0)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - frame).max() <= 1


@pytest.mark.parametrize("name", list(EFFECTS))
def test_end_shows_background_inside_mask_only(name, scene):
    frame, bg, mask, rng = scene
    effect = EFFECTS[name]()
    effect.start(frame, mask, rng)
    out = effect.render(frame, bg, mask, 1.0)
    inside = mask > 0.5
    assert np.abs(out[inside].astype(int) - bg[inside]).max() <= 1
    assert np.abs(out[~inside].astype(int) - frame[~inside]).max() <= 1


@pytest.mark.parametrize("name", list(EFFECTS))
def test_midway_is_partial(name, scene):
    frame, bg, mask, rng = scene
    effect = EFFECTS[name]()
    effect.start(frame, mask, rng)
    out = effect.render(frame, bg, mask, 0.5)
    assert not np.array_equal(out, frame)


@pytest.mark.parametrize("name", list(EFFECTS))
def test_handles_empty_mask(name, scene):
    frame, bg, _, rng = scene
    empty = np.zeros((H, W), np.float32)
    effect = EFFECTS[name]()
    effect.start(frame, empty, rng)
    for p in (0.0, 0.5, 1.0):
        assert effect.render(frame, bg, empty, p).shape == frame.shape


def test_mask_bbox():
    m = np.zeros((H, W), np.float32)
    assert mask_bbox(m) is None
    m[40:170, 120:200] = 1
    x0, y0, x1, y1 = mask_bbox(m)
    assert x0 == 120 and y0 == 40 and 196 <= x1 <= 204 and 166 <= y1 <= 174
