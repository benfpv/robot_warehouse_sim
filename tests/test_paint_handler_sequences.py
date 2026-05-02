"""High-value PaintHandler sequence and UX tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from data.paint.paint_handler import PaintHandler
from data.constants import ZONE_NONE, ZONE_IMPORT


def _cell_to_pixel(gx, gy, gw, gh, zone_view_rect):
    """Map a grid cell to a pixel in the paintable panel area."""
    vx0, vy0, vw, vh = zone_view_rect
    map_w = vw - PaintHandler.BRUSH_W
    map_h = vh - PaintHandler.BUTTON_H
    px = vx0 + int((gx + 0.5) * map_w / gw)
    py = vy0 + int((gy + 0.5) * map_h / gh)
    return px, py


class TestPaintStrokeQuality:
    """Fast drags should remain gapless and deterministic."""

    def test_left_drag_uses_bresenham_gapless_line(self):
        gw, gh = 10, 10
        zone_map = np.zeros((gh, gw), dtype=np.uint8)
        wh = SimpleNamespace(zoneMap=zone_map, pending_zone_changes=set())
        ph = PaintHandler(wh, (gw, gh), (0, 0, 100, 100))
        ph.paint_zone = ZONE_IMPORT
        ph.brush_size = 1

        sx, sy = _cell_to_pixel(1, 1, gw, gh, ph.zone_view_rect)
        ex, ey = _cell_to_pixel(5, 1, gw, gh, ph.zone_view_rect)

        ph.on_mouse(cv2.EVENT_LBUTTONDOWN, sx, sy, cv2.EVENT_FLAG_LBUTTON, None)
        ph.on_mouse(cv2.EVENT_MOUSEMOVE, ex, ey, cv2.EVENT_FLAG_LBUTTON, None)
        ph.on_mouse(cv2.EVENT_LBUTTONUP, ex, ey, 0, None)

        for x in range(1, 6):
            assert wh.zoneMap[1, x] == ZONE_IMPORT
            assert (x, 1) in wh.pending_zone_changes

    def test_right_drag_erases_and_tracks_changes(self):
        gw, gh = 8, 8
        zone_map = np.full((gh, gw), ZONE_IMPORT, dtype=np.uint8)
        wh = SimpleNamespace(zoneMap=zone_map, pending_zone_changes=set())
        ph = PaintHandler(wh, (gw, gh), (0, 0, 96, 96))

        sx, sy = _cell_to_pixel(2, 2, gw, gh, ph.zone_view_rect)
        ex, ey = _cell_to_pixel(4, 2, gw, gh, ph.zone_view_rect)

        ph.on_mouse(cv2.EVENT_RBUTTONDOWN, sx, sy, cv2.EVENT_FLAG_RBUTTON, None)
        ph.on_mouse(cv2.EVENT_MOUSEMOVE, ex, ey, cv2.EVENT_FLAG_RBUTTON, None)
        ph.on_mouse(cv2.EVENT_RBUTTONUP, ex, ey, 0, None)

        for x in range(2, 5):
            assert wh.zoneMap[2, x] == ZONE_NONE
            assert (x, 2) in wh.pending_zone_changes


class TestPaintControls:
    """Brush and strategy controls should be easy to use and reliable."""

    def test_right_strip_selects_brush_then_toggles_strategy(self):
        gw, gh = 10, 10
        zone_map = np.zeros((gh, gw), dtype=np.uint8)
        strategy = SimpleNamespace(name="Zone", label="AUTO", enabled=False)
        optimizer = SimpleNamespace(strategies=[strategy])
        wh = SimpleNamespace(zoneMap=zone_map, pending_zone_changes=set())
        ph = PaintHandler(wh, (gw, gh), (0, 0, 100, 100), optimizer=optimizer)

        vx0, vy0, vw, vh = ph.zone_view_rect
        map_w = vw - ph.BRUSH_W
        map_h = vh - ph.BUTTON_H
        brush_x = vx0 + map_w

        n_slots = len(ph._BRUSH_SIZES) + len(optimizer.strategies)
        slot_h = max(map_h // n_slots, 1)

        # Brush slot 1 (size=3)
        px = brush_x + 2
        py = vy0 + slot_h + 1
        ph.on_mouse(cv2.EVENT_LBUTTONDOWN, px, py, 0, None)
        assert ph.brush_size == 3

        # Strategy slot (first strategy after brush slots)
        strat_y = vy0 + len(ph._BRUSH_SIZES) * slot_h + 1
        ph.on_mouse(cv2.EVENT_LBUTTONDOWN, px, strat_y, 0, None)
        assert strategy.enabled is True


class TestPaintHotReload:
    """Key-L hot reload should refresh maps and enqueue full reconciliation."""

    def test_handle_key_l_reloads_assets_and_marks_all_cells_changed(self):
        gw, gh = 6, 5
        new_zone = np.ones((gh, gw), dtype=np.uint8)
        wh = SimpleNamespace(
            zoneMap=np.zeros((gh, gw), dtype=np.uint8),
            pending_zone_changes=set(),
            chargerSpawnMap=[],
            robotSpawnMap=[],
            update_zone_counts=MagicMock(),
        )
        optimizer = SimpleNamespace(strategies=[], reset=MagicMock())
        ph = PaintHandler(wh, (gw, gh), (0, 0, 80, 60), optimizer=optimizer)

        with patch("data.paint.map_importer.MapImporter.load", return_value=new_zone), \
                patch("data.paint.map_importer.MapImporter.load_spawn_map", side_effect=[[(1, 1)], [(2, 2)] ]):
            consumed = ph.handle_key(ord("l"))

        assert consumed is True
        assert np.array_equal(wh.zoneMap, new_zone)
        assert len(wh.pending_zone_changes) == gw * gh
        assert wh.chargerSpawnMap == [(1, 1)]
        assert wh.robotSpawnMap == [(2, 2)]
        wh.update_zone_counts.assert_called_once()
        optimizer.reset.assert_called_once_with(wh)
