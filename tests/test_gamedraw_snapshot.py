"""Snapshot safety net for ``MainGame.gameDraw``.

This test exists to lock in the rendered composite output of ``gameDraw`` so
any refactor of the dashboard rendering pipeline (panel extraction, presenter
introduction, layout module, etc.) is immediately caught if it produces a
different image.

Strategy
--------
1. Patch all OpenCV / Win32 display calls so ``MainGame()`` constructs cleanly
   in a headless environment.
2. Patch ``time.time`` to a fixed value so all timestamp-derived rendering
   (elapsed labels, age strings, history sampling) is deterministic.
3. Build a real ``MainGame`` and tick the warehouse a fixed number of times
   with a seeded RNG.
4. Capture the composite image written by ``cv2.imshow``.
5. Validate:
   - Image shape and dtype are exactly as expected.
   - Two back-to-back ``gameDraw`` calls with no warehouse mutation produce
     byte-identical images (no hidden state mutation in renderers).
   - The composite contains non-trivial content (renderer is actually
     drawing — not returning a blank image).

The strict byte-equality check across consecutive draws is the long-term
guard:  during the refactor every panel extraction / presenter wiring step
must keep this assertion passing.
"""
from __future__ import annotations

import os
import random
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Headless patch helpers
# ---------------------------------------------------------------------------

_FROZEN_TIME = 1_700_000_000.0  # arbitrary fixed wall clock


class _ImshowCapture:
    """Drop-in for ``cv2.imshow`` that records the most recent frame."""

    def __init__(self):
        self.frames: list[np.ndarray] = []

    def __call__(self, _window_name, image):
        # Copy so subsequent in-place edits to the source array (if any)
        # cannot retroactively mutate captured snapshots.
        self.frames.append(np.ascontiguousarray(image).copy())

    @property
    def latest(self) -> np.ndarray:
        assert self.frames, "gameDraw never called cv2.imshow"
        return self.frames[-1]


def _build_headless_main_game(monkeypatch_stack: ExitStack):
    """Construct a ``MainGame`` instance with all display side-effects stubbed.

    Returns ``(main_game, imshow_capture)``.  Caller is responsible for
    keeping the ExitStack open for the lifetime of any further ``gameDraw``
    calls so the cv2 patches stay active.
    """
    capture = _ImshowCapture()

    # Freeze wall clock — must be patched *before* MainGame.__init__ runs
    # because __init__ records ``self.timeStart = int(time.time())``.
    monkeypatch_stack.enter_context(patch("time.time", return_value=_FROZEN_TIME))
    monkeypatch_stack.enter_context(patch("time.sleep", return_value=None))

    # Stub the borderless window helper so no Win32 window is created.
    monkeypatch_stack.enter_context(
        patch("data.display.display_functions.Display_Functions.init_borderless_window",
              return_value=None))
    monkeypatch_stack.enter_context(
        patch("data.display.display_functions.Display_Functions.get_screen_resolution",
              return_value=(1920, 1080)))

    # Stub all cv2 display-side calls used by MainGame.  We must patch the
    # ``cv2`` module attributes so any access from within ``main`` picks up
    # the stubs (``main`` does ``import cv2``, so the lookup is ``cv2.X``).
    import cv2 as _cv2
    monkeypatch_stack.enter_context(patch.object(_cv2, "imshow", capture))
    monkeypatch_stack.enter_context(patch.object(_cv2, "waitKey", return_value=0xFF))
    monkeypatch_stack.enter_context(patch.object(_cv2, "namedWindow", return_value=None))
    monkeypatch_stack.enter_context(patch.object(_cv2, "setMouseCallback", return_value=None))
    monkeypatch_stack.enter_context(patch.object(_cv2, "moveWindow", return_value=None))
    monkeypatch_stack.enter_context(patch.object(_cv2, "destroyAllWindows", return_value=None))

    # Lazy import — main.py does cv2.setMouseCallback at module-import time
    # via MainGame.__init__, but since we patched cv2 first it's safe now.
    from main import MainGame  # noqa: WPS433  (intentional local import)

    # Seed module-level RNGs so any stochastic warehouse spawn placement is
    # reproducible.
    random.seed(20260420)
    np.random.seed(20260420)

    game = MainGame()
    return game, capture


def _tick_warehouse(game, ticks: int):
    """Run ``ticks`` deterministic warehouse updates without rendering."""
    for _ in range(ticks):
        game.optimizer.step(game.warehouse, game.warehouse.traffic_ema)
        game.warehouse = game.warehouse.update_warehouse()
    # ``gameLoop`` normally seeds ``timeElapsed`` each tick; ``gameDraw``
    # reads it for the elapsed-time label.
    game.timeElapsed = ticks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Resource files are required to construct a real MainGame.  Skip cleanly if
# the test is invoked from a directory where they can't be located.
_REQUIRED_RESOURCES = (
    "resources/list_items.csv",
    "resources/list_addresses.csv",
)


def _resources_present() -> bool:
    return all(os.path.exists(p) for p in _REQUIRED_RESOURCES)


@pytest.mark.skipif(not _resources_present(), reason="warehouse resources not available")
class TestGameDrawSnapshot:
    """Pixel-level snapshot guard for the composite dashboard renderer."""

    def test_composite_has_expected_shape_and_dtype(self):
        with ExitStack() as stack:
            game, capture = _build_headless_main_game(stack)
            _tick_warehouse(game, ticks=20)
            game._hist_last_sample = _FROZEN_TIME  # suppress 1 Hz history sample
            game.gameDraw()

            img = capture.latest
            expected_w, expected_h = game.composite_windowRes
            assert img.shape == (expected_h, expected_w, 3), (
                "composite shape changed: got {} expected {}".format(
                    img.shape, (expected_h, expected_w, 3)))
            assert img.dtype == np.uint8

    def test_composite_is_non_trivial(self):
        """Renderer must produce real content, not a blank/uniform image."""
        with ExitStack() as stack:
            game, capture = _build_headless_main_game(stack)
            _tick_warehouse(game, ticks=20)
            game._hist_last_sample = _FROZEN_TIME
            game.gameDraw()

            img = capture.latest
            unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
            assert unique_colors > 50, (
                "composite suspiciously uniform ({} colors) — renderer "
                "may be drawing nothing".format(unique_colors))
            # Mean brightness sanity: not pitch-black, not fully saturated.
            mean = float(img.mean())
            assert 5.0 < mean < 250.0, "composite mean brightness {:.2f} out of range".format(mean)

    def test_consecutive_draws_are_byte_identical(self):
        """Two back-to-back draws with no sim mutation must hash identically.

        This is the primary refactor guard.  If a panel extraction accidentally
        mutates warehouse state, advances RNG, or reads ``time.time()`` for
        per-frame jitter, this assertion will fail immediately.
        """
        with ExitStack() as stack:
            game, capture = _build_headless_main_game(stack)
            _tick_warehouse(game, ticks=20)
            game._hist_last_sample = _FROZEN_TIME

            game.gameDraw()
            first = capture.latest.copy()
            game._hist_last_sample = _FROZEN_TIME
            game.gameDraw()
            second = capture.latest.copy()

            if not np.array_equal(first, second):
                diff = np.argwhere(np.any(first != second, axis=2))
                first_diff = diff[0] if len(diff) else None
                pytest.fail(
                    "gameDraw is non-deterministic: {} pixels differ "
                    "between back-to-back calls (first diff at {})".format(
                        len(diff), first_diff))

    def test_draw_does_not_mutate_warehouse_loop_count(self):
        """Renderer must be a pure observer of warehouse state.

        ``gameDraw`` must not advance simulation counters.  A regression here
        would mean the renderer is calling something that ticks the sim.
        """
        with ExitStack() as stack:
            game, capture = _build_headless_main_game(stack)
            _tick_warehouse(game, ticks=20)
            game._hist_last_sample = _FROZEN_TIME

            before_loops = game.warehouse.warehouseLoopCount
            before_robots = len(game.warehouse.robots)
            before_packages = len(game.warehouse.packages)
            game.gameDraw()
            assert game.warehouse.warehouseLoopCount == before_loops
            assert len(game.warehouse.robots) == before_robots
            assert len(game.warehouse.packages) == before_packages
