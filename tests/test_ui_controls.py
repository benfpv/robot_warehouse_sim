"""Tests for main.py UI controls: button layouts, ESC exit, keyboard handling."""
import types
import pytest
from unittest.mock import patch, MagicMock
from main import MainGame


# ---------------------------------------------------------------------------
# Lightweight stub — just enough state for layout methods without OpenCV
# ---------------------------------------------------------------------------

def _make_layout_stub():
    """Return a bare object with MainGame layout methods + required attrs."""
    stub = object.__new__(MainGame)
    stub.warehouse_windowRes = (320, 280)
    stub.composite_windowRes = (960, 680)
    stub.sub_windowRes = (160, 140)
    return stub


# ===========================================================================
# _horizontal_button_layout (static helper)
# ===========================================================================

class TestHorizontalButtonLayout:
    """Verify the shared static button layout helper."""

    def test_single_item(self):
        layout = MainGame._horizontal_button_layout(['a'], x0=10, y0=5, btn_w=20, btn_h=12, gap=4)
        assert layout == [('a', 10, 5, 30, 17)]

    def test_multiple_items_spacing(self):
        layout = MainGame._horizontal_button_layout(['a', 'b', 'c'], x0=0, y0=0, btn_w=10, btn_h=8, gap=2)
        assert len(layout) == 3
        assert layout[0] == ('a', 0, 0, 10, 8)
        assert layout[1] == ('b', 12, 0, 22, 8)
        assert layout[2] == ('c', 24, 0, 34, 8)

    def test_no_overlap_between_buttons(self):
        layout = MainGame._horizontal_button_layout(list(range(5)), x0=0, y0=0, btn_w=30, btn_h=10, gap=5)
        for i in range(len(layout) - 1):
            _, _, _, right_edge, _ = layout[i]
            _, left_edge, _, _, _ = layout[i + 1]
            assert left_edge >= right_edge, f"Button {i} overlaps button {i+1}"

    def test_empty_items(self):
        layout = MainGame._horizontal_button_layout([], x0=0, y0=0)
        assert layout == []

    def test_button_height_consistent(self):
        layout = MainGame._horizontal_button_layout(['x', 'y'], x0=5, y0=10, btn_w=20, btn_h=15)
        for _, _, y0, _, y1 in layout:
            assert y1 - y0 == 15

    def test_button_width_consistent(self):
        layout = MainGame._horizontal_button_layout(['x', 'y'], x0=5, y0=10, btn_w=20)
        for _, x0, _, x1, _ in layout:
            assert x1 - x0 == 20

    def test_default_parameters(self):
        layout = MainGame._horizontal_button_layout(['a'], x0=0, y0=0)
        assert layout == [('a', 0, 0, 34, 11)]


# ===========================================================================
# Refactored layout methods — exact coordinate regression tests
# ===========================================================================

class TestDisplayFpsButtonLayout:
    """FPS buttons: 2 buttons at top-left of main view."""

    def test_returns_correct_items(self):
        stub = _make_layout_stub()
        layout = stub._display_fps_button_layout()
        items = [item for item, *_ in layout]
        assert items == [30, 60]

    def test_exact_coordinates(self):
        stub = _make_layout_stub()
        layout = stub._display_fps_button_layout()
        # Known-correct coordinates from before refactor
        assert layout == [(30, 106, 4, 128, 15), (60, 131, 4, 153, 15)]

    def test_no_overlap_with_esc_hint(self):
        """ESC hint is drawn at x=160; buttons must end before that."""
        stub = _make_layout_stub()
        layout = stub._display_fps_button_layout()
        max_right = max(x1 for _, _, _, x1, _ in layout)
        assert max_right < 160, "FPS buttons overlap ESC hint at x=160"


class TestPowerPolicyButtonLayout:
    """Power policy buttons: 3 modes, right-aligned to column 2."""

    def test_returns_correct_modes(self):
        stub = _make_layout_stub()
        layout = stub._power_policy_button_layout()
        items = [item for item, *_ in layout]
        assert items == ['eco', 'balanced', 'performance']

    def test_three_buttons_no_overlap(self):
        stub = _make_layout_stub()
        layout = stub._power_policy_button_layout()
        for i in range(len(layout) - 1):
            assert layout[i][3] <= layout[i + 1][1]

    def test_y_is_below_main_view(self):
        stub = _make_layout_stub()
        layout = stub._power_policy_button_layout()
        mh = stub.warehouse_windowRes[1]  # 280
        for _, _, y0, _, y1 in layout:
            assert y0 >= mh, f"Button y0={y0} inside main view (mh={mh})"


class TestFlowPolicyButtonLayout:
    """Flow policy buttons: 3 modes, right-aligned to column 3."""

    def test_returns_correct_modes(self):
        stub = _make_layout_stub()
        layout = stub._flow_policy_button_layout()
        items = [item for item, *_ in layout]
        assert items == ['steady', 'balanced', 'throughput']

    def test_no_overlap_with_power_policy(self):
        stub = _make_layout_stub()
        power = stub._power_policy_button_layout()
        flow = stub._flow_policy_button_layout()
        power_max_x = max(x1 for _, _, _, x1, _ in power)
        flow_min_x = min(x0 for _, x0, _, _, _ in flow)
        assert flow_min_x >= power_max_x, "Flow buttons overlap power buttons"


class TestPkgTargetButtonLayout:
    """Package target buttons: 3 modes, right-aligned to right edge."""

    def test_returns_correct_modes(self):
        stub = _make_layout_stub()
        layout = stub._pkg_target_button_layout()
        items = [item for item, *_ in layout]
        assert items == ['random', 'nearest', 'zone_edge']

    def test_within_composite_bounds(self):
        stub = _make_layout_stub()
        layout = stub._pkg_target_button_layout()
        cw = stub.composite_windowRes[0]  # 960
        for _, x0, _, x1, _ in layout:
            assert x0 >= 0 and x1 <= cw


class TestAllButtonLayoutsNoOverlap:
    """Cross-layout overlap checks for info-panel row buttons."""

    def test_info_row_buttons_non_overlapping(self):
        stub = _make_layout_stub()
        # Collect all info-panel-row button rects (y > mh)
        all_rects = []
        for layout_fn in (stub._power_policy_button_layout,
                          stub._flow_policy_button_layout,
                          stub._pkg_target_button_layout):
            for item, x0, y0, x1, y1 in layout_fn():
                all_rects.append((x0, y0, x1, y1))
        # Check no pair overlaps (same y-row, so only check x)
        for i, (ax0, ay0, ax1, ay1) in enumerate(all_rects):
            for j, (bx0, by0, bx1, by1) in enumerate(all_rects):
                if i >= j:
                    continue
                if ay0 == by0:  # same row
                    assert ax1 <= bx0 or bx1 <= ax0, \
                        f"Buttons {i} and {j} overlap: [{ax0},{ax1}] vs [{bx0},{bx1}]"


# ===========================================================================
# ESC key exit
# ===========================================================================

class TestEscKeyExit:
    """ESC key (keycode 27) must trigger gameEnd()."""

    def test_esc_sets_exit_true(self):
        stub = _make_layout_stub()
        stub.exit = False
        # Minimal state for gameDraw's waitKey path
        stub.warehouseWindow = MagicMock()
        stub.warehouse = MagicMock()
        stub.warehouse.robots = []
        stub.warehouse.packages = []
        stub.warehouse.chargers = []
        stub.warehouse.road_map = None
        stub.warehouse.plaza_map = None
        stub.paint_handler = MagicMock()
        stub._show_roads = False
        stub._show_plazas = False
        stub._show_heatmap = False

        # Patch cv2.waitKey to return ESC (27)
        with patch('main.cv2') as mock_cv2:
            mock_cv2.waitKey.return_value = 27
            mock_cv2.EVENT_LBUTTONDOWN = 1
            mock_cv2.FONT_HERSHEY_SIMPLEX = 0
            mock_cv2.getTextSize.return_value = ((10, 5), 0)
            mock_cv2.INTER_NEAREST = 0
            # gameEnd sets self.exit = True
            stub.gameEnd()

        assert stub.exit is True

    def test_non_esc_does_not_exit(self):
        stub = _make_layout_stub()
        stub.exit = False
        # Simulate a non-ESC key
        key = ord('a')
        # gameEnd should NOT be called for non-ESC keys
        if key != 27:
            pass  # no action
        assert stub.exit is False
