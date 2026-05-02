"""High-value UI interaction sequence tests.

These tests focus on input-routing correctness and interaction semantics in
MainGame without requiring a live OpenCV window.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2

from main import MainGame


def _make_main_input_stub():
    """Create a minimal MainGame-like object for input routing tests."""
    stub = object.__new__(MainGame)
    stub.warehouse_windowRes = (320, 280)
    stub.composite_windowRes = (960, 680)
    stub.sub_windowRes = (160, 140)
    stub._show_heatmap = True
    stub._show_roads = True
    stub._show_plazas = True
    stub.paint_handler = SimpleNamespace(on_mouse=MagicMock())
    stub.warehouse = SimpleNamespace(
        _robot_target_count=2,
        set_robot_count=MagicMock(),
    )
    return stub


class TestMouseRoutingOrder:
    """Main mouse callback should preserve control-priority and delegation."""

    def test_on_mouse_stops_after_first_control_hit(self):
        stub = _make_main_input_stub()
        stub._handle_display_fps_click = MagicMock(return_value=True)
        stub._handle_power_policy_click = MagicMock(return_value=False)
        stub._handle_flow_policy_click = MagicMock(return_value=False)
        stub._handle_robot_count_click = MagicMock(return_value=False)
        stub._handle_pkg_target_mode_click = MagicMock(return_value=False)
        stub._handle_heatmap_toggle_click = MagicMock(return_value=False)

        stub._on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)

        stub._handle_display_fps_click.assert_called_once_with(cv2.EVENT_LBUTTONDOWN, 10, 10)
        stub._handle_power_policy_click.assert_not_called()
        stub._handle_flow_policy_click.assert_not_called()
        stub._handle_robot_count_click.assert_not_called()
        stub._handle_pkg_target_mode_click.assert_not_called()
        stub._handle_heatmap_toggle_click.assert_not_called()
        stub.paint_handler.on_mouse.assert_not_called()

    def test_on_mouse_delegates_to_paint_handler_when_controls_miss(self):
        stub = _make_main_input_stub()
        stub._handle_display_fps_click = MagicMock(return_value=False)
        stub._handle_power_policy_click = MagicMock(return_value=False)
        stub._handle_flow_policy_click = MagicMock(return_value=False)
        stub._handle_robot_count_click = MagicMock(return_value=False)
        stub._handle_pkg_target_mode_click = MagicMock(return_value=False)
        stub._handle_heatmap_toggle_click = MagicMock(return_value=False)

        stub._on_mouse(cv2.EVENT_MOUSEMOVE, 77, 55, cv2.EVENT_FLAG_LBUTTON, None)

        stub.paint_handler.on_mouse.assert_called_once_with(
            cv2.EVENT_MOUSEMOVE, 77, 55, cv2.EVENT_FLAG_LBUTTON, None
        )


class TestHeatmapToggleUX:
    """Heatmap/road/plaza toggles should support deterministic click toggling."""

    def test_each_toggle_flips_state_and_round_trips(self):
        stub = _make_main_input_stub()
        initial = {
            "heatmap": stub._show_heatmap,
            "roads": stub._show_roads,
            "plazas": stub._show_plazas,
        }

        layout = stub._heatmap_toggle_button_layout()
        for key, _label, bx0, by0, bx1, by1 in layout:
            cx = (bx0 + bx1) // 2
            cy = (by0 + by1) // 2

            consumed = stub._handle_heatmap_toggle_click(cv2.EVENT_LBUTTONDOWN, cx, cy)
            assert consumed is True
            assert getattr(stub, f"_show_{key}") is (not initial[key])

            # Second click returns to original state (important for UX predictability)
            consumed = stub._handle_heatmap_toggle_click(cv2.EVENT_LBUTTONDOWN, cx, cy)
            assert consumed is True
            assert getattr(stub, f"_show_{key}") is initial[key]


class TestRobotCountButtons:
    """Robot target controls should enforce +/- semantics and lower bound."""

    def test_plus_and_minus_clicks_adjust_target_safely(self):
        stub = _make_main_input_stub()

        # Click '+' when target is 2 -> expect set_robot_count(3)
        plus = next(item for item in stub._robot_count_button_layout() if item[0] == '+')
        _, bx0, by0, bx1, by1 = plus
        cx = (bx0 + bx1) // 2
        cy = (by0 + by1) // 2
        consumed = stub._handle_robot_count_click(cv2.EVENT_LBUTTONDOWN, cx, cy)
        assert consumed is True
        stub.warehouse.set_robot_count.assert_called_with(3)

        # Click '-' when target is 1 -> must not decrement below 1
        stub.warehouse._robot_target_count = 1
        stub.warehouse.set_robot_count.reset_mock()
        minus = next(item for item in stub._robot_count_button_layout() if item[0] == '-')
        _, bx0, by0, bx1, by1 = minus
        cx = (bx0 + bx1) // 2
        cy = (by0 + by1) // 2
        consumed = stub._handle_robot_count_click(cv2.EVENT_LBUTTONDOWN, cx, cy)
        assert consumed is True
        stub.warehouse.set_robot_count.assert_not_called()
