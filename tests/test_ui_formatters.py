"""Unit tests for ``data.ui.formatters`` and ``data.ui.layout``.

Pure-function coverage so any future change to display formatting or
button-row layout math is caught immediately.
"""
import pytest

from data.ui.formatters import fmt_age, fmt_elapsed, fmt_n, fmt_rate
from data.ui.layout import PanelGeometry, horizontal_button_layout


# ---------------------------------------------------------------------------
# fmt_elapsed
# ---------------------------------------------------------------------------

class TestFmtElapsed:
    @pytest.mark.parametrize("seconds, expected_substr", [
        (-5,    "0s"),         # negative clamps to zero
        (0,     "0s"),
        (1,     "1s"),
        (59,    "59s"),
        (60,    "1min"),
        (75,    "1min 15s"),
        (3600,  "1hr"),
        (3725,  "1hr 2mins"),  # picks the *two* most-significant units
        (86400, "1d"),
    ])
    def test_basic(self, seconds, expected_substr):
        assert expected_substr in fmt_elapsed(seconds)

    def test_two_unit_cap(self):
        # 1d 1hr 1min 1s -> only first two units shown
        out = fmt_elapsed(86400 + 3600 + 60 + 1)
        # Output should contain at most one space between two parts
        assert out.count(" ") == 1

    def test_year_rollover(self):
        out = fmt_elapsed(int(365.25 * 24 * 3600) * 2)
        assert "yrs" in out  # plural

    def test_decade_label(self):
        out = fmt_elapsed(int(365.25 * 24 * 3600) * 11)
        assert "decade" in out


# ---------------------------------------------------------------------------
# fmt_age
# ---------------------------------------------------------------------------

class TestFmtAge:
    @pytest.mark.parametrize("seconds, expected", [
        (0,      "0s"),
        (3,      "3s"),
        (59,     "59s"),
        (60,     "1m0s"),
        (132,    "2m12s"),
        (3600,   "1h0m"),
        (3905,   "1h5m"),
        (86400,  "1d0h"),
        (90061,  "1d1h"),
    ])
    def test_basic(self, seconds, expected):
        assert fmt_age(seconds) == expected

    def test_truncates_to_int(self):
        assert fmt_age(3.7) == "3s"


# ---------------------------------------------------------------------------
# fmt_rate
# ---------------------------------------------------------------------------

class TestFmtRate:
    def test_zero(self):
        assert fmt_rate(0) == "0/s"

    @pytest.mark.parametrize("val, suffix", [
        (5.0,        "/s"),
        (-5.0,       "/s"),
        (50,         "/s"),
        (15_000,     "K/s"),
        (-15_000,    "K/s"),
        (2_000_000,  "M/s"),
    ])
    def test_suffix_buckets(self, val, suffix):
        out = fmt_rate(val)
        assert out.endswith(suffix)

    def test_signed_output(self):
        # All non-zero magnitudes carry a sign.
        for v in [1.0, -1.0, 100, -100, 50_000, -50_000, 5_000_000]:
            assert fmt_rate(v)[0] in "+-"


# ---------------------------------------------------------------------------
# fmt_n
# ---------------------------------------------------------------------------

class TestFmtN:
    @pytest.mark.parametrize("val, expected", [
        (0,           "0"),
        (5,           "5"),
        (999,         "999"),
        (9_999,       "9999"),     # below 10k threshold -> bare int
        (10_000,      "10.0K"),
        (1_500_000,   "1.5M"),
    ])
    def test_buckets(self, val, expected):
        assert fmt_n(val) == expected


# ---------------------------------------------------------------------------
# horizontal_button_layout
# ---------------------------------------------------------------------------

class TestHorizontalButtonLayout:
    def test_empty_items(self):
        assert horizontal_button_layout([], x0=0, y0=0) == []

    def test_single_button_geometry(self):
        out = horizontal_button_layout(["A"], x0=10, y0=20, btn_w=30, btn_h=12)
        assert out == [("A", 10, 20, 40, 32)]

    def test_three_buttons_no_overlap(self):
        out = horizontal_button_layout(["a", "b", "c"], x0=0, y0=0, btn_w=10, btn_h=5, gap=2)
        # Each pair must satisfy bx0_next >= bx1_prev + gap
        for prev, nxt in zip(out, out[1:]):
            assert nxt[1] >= prev[3] + 2

    def test_returns_item_first(self):
        out = horizontal_button_layout(["x"], x0=0, y0=0)
        assert out[0][0] == "x"


# ---------------------------------------------------------------------------
# PanelGeometry
# ---------------------------------------------------------------------------

class TestPanelGeometry:
    def test_from_warehouse_window_default_sizes(self):
        # The historical (320, 280) warehouse window must produce the legacy
        # (960, 680) composite size — exact regression check.
        g = PanelGeometry.from_warehouse_window((320, 280))
        assert (g.sub_w, g.sub_h) == (160, 140)
        assert g.composite_w == 960
        assert g.composite_h == 680
        assert g.composite_res == (960, 680)

    def test_info_panel_top_matches_warehouse_height(self):
        g = PanelGeometry.from_warehouse_window((320, 280))
        assert g.info_panel_top == 280

    def test_info_col_w_quartile(self):
        g = PanelGeometry.from_warehouse_window((320, 280))
        assert g.info_col_w == 240

    def test_chart_top_offset(self):
        g = PanelGeometry.from_warehouse_window((320, 280))
        assert g.chart_top == 280 + 255

    def test_immutable(self):
        g = PanelGeometry.from_warehouse_window((320, 280))
        with pytest.raises((AttributeError, Exception)):  # frozen dataclass
            g.warehouse_w = 999  # type: ignore[misc]
