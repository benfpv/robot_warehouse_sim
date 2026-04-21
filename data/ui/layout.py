"""Layout primitives and panel geometry for the dashboard composite window.

Extracted from ``main.MainGame``.  Holds:

* ``PanelGeometry`` — named pixel constants (window/sub-view/panel/chart sizes)
  derived once at construction time.
* ``horizontal_button_layout`` — pure helper to build a row of button rects.

Keeping these here keeps ``main.py`` free of magic numbers and lets layout
math be unit-tested without instantiating ``MainGame``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelGeometry:
    """Pixel geometry for the composite dashboard window.

    All values are derived from the two free parameters
    (``warehouse_window_res`` and the sub-view scale = 0.5).  Stored as
    explicit named attributes so call sites read as
    ``geom.composite_w`` rather than ``cw // 4 * 2 - 4 - total_w``.
    """

    # Inputs
    warehouse_w: int
    warehouse_h: int

    # Sub-view tile (top-row mini panels)
    sub_w: int
    sub_h: int

    # Bottom-strip heights
    panel_h: int = 255
    chart_h: int = 145

    @property
    def composite_w(self) -> int:
        return self.warehouse_w + self.sub_w * 4

    @property
    def composite_h(self) -> int:
        return self.warehouse_h + self.panel_h + self.chart_h

    @property
    def composite_res(self) -> tuple[int, int]:
        return (self.composite_w, self.composite_h)

    @property
    def info_panel_top(self) -> int:
        """Y coordinate where the 4-column info panel begins."""
        return self.warehouse_h

    @property
    def info_col_w(self) -> int:
        """Width of one of the four equal info-panel columns."""
        return self.composite_w // 4

    @property
    def chart_top(self) -> int:
        return self.warehouse_h + self.panel_h

    @classmethod
    def from_warehouse_window(cls, warehouse_window_res: tuple[int, int]) -> "PanelGeometry":
        """Build geometry from the upscaled warehouse render resolution.

        ``warehouse_window_res`` is ``(width, height)``.  Sub-view tiles are
        half that on each axis (the historic 0.5 factor).
        """
        w, h = warehouse_window_res
        return cls(
            warehouse_w=w,
            warehouse_h=h,
            sub_w=int(w * 0.5),
            sub_h=int(h * 0.5),
        )


def horizontal_button_layout(items, x0, y0, btn_w=34, btn_h=11, gap=3):
    """Build a horizontal row of button rects from ``items``.

    Returns ``[(item, bx0, y0, bx1, y1), ...]``.  ``x0`` is the left edge of
    the first button.  Pure function — no dependencies, fully deterministic.
    """
    layout = []
    for i, item in enumerate(items):
        bx0 = x0 + i * (btn_w + gap)
        layout.append((item, bx0, y0, bx0 + btn_w, y0 + btn_h))
    return layout
