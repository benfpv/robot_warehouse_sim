"""Robot Warehouse Simulator — entry point and composite dashboard.

Runs the simulation loop at 40 Hz and renders a 960×680 composite OpenCV
window with 8 sub-views, a 4-column info panel, and a 4-chart strip.
All display, mouse, and keyboard interaction is handled here.
"""
import os
import logging
import numpy as np
import math
import random
import time
import cv2

from data.importer import Importer
from data.functions import Functions
from data.constants import (
    POWER_POLICY_MODES, FLOW_POLICY_MODES, PKG_TARGET_MODES,
)
from data.log_setup import configure_logging

from data.warehouse.warehouse import Warehouse
from data.draw.draw_warehouse import Draw_Warehouse
from data.display.display_functions import Display_Functions
from data.paint.paint_handler import PaintHandler
from data.paint.map_importer import MapImporter
from data.paint.warehouse_optimizer import WarehouseOptimizer
from data.paint.zone_strategy import ZoneStrategy
from data.ui.formatters import fmt_age, fmt_elapsed, fmt_n, fmt_rate
from data.ui.info_panel import draw_info_panel
from data.ui.layout import PanelGeometry, horizontal_button_layout
from data.ui.presenter import FleetSnapshot, build_fleet_snapshot
from data.ui.styles import (
    POWER_POLICY_STYLE, POWER_POLICY_REASON, POWER_POLICY_HINT,
    FLOW_POLICY_STYLE, FLOW_POLICY_REASON, FLOW_POLICY_HINT,
    PKG_TARGET_STYLE, PKG_TARGET_REASON,
)

log = logging.getLogger('main')


class MainGame:
    """Top-level simulation controller.

    Owns the Warehouse instance, the composite display window, the
    paint/zone-edit handler, the optimizer, heatmaps, chart histories,
    and all rendering logic.  ``gameStart()`` enters the main loop which
    alternates between ``gameLoop()`` (sim tick) and ``gameDraw()``
    (rate-limited display repaint).
    """
    _DISPLAY_FPS_OPTIONS = [30, 60]
    _POWER_POLICY_MODES = POWER_POLICY_MODES
    _FLOW_POLICY_MODES = FLOW_POLICY_MODES
    _PKG_TARGET_MODES = PKG_TARGET_MODES

    def __init__(self):
        log.info("--- MainGame Init ---")
        self.timeStart = int(time.time())
        self.exit = False
        self._init_constants()
        self.initWarehouseWindow()
        self._init_imports()
        self._init_warehouse()
        self._init_optimizer_and_paint()
        self._init_maps()
        # Temporary draw
        self.warehouseWindow = self.warehouse_windowArray.copy()

    def _init_constants(self):
        """User-tunable display/sim constants. No I/O, no allocations."""
        self.warehouse_res = (80, 70)
        self.warehouse_windowBackgroundColour = [20, 20, 20]
        self.warehouse_windowRes = (320, 280)  # upsized resolution (width, height)
        self.sim_frametime  = 1 / 40   # 40 sim ticks/sec
        self.draw_frametime = 1 / 15   # 15 display fps (reduced to save power)
        self._last_draw_time = 0.0
        self.frametime_cap = self.sim_frametime  # kept for hist_sample_every

    def _init_imports(self):
        """Load CSV resources (items, addresses)."""
        self.itemsList = Importer.init_import_csv_as_list('resources/list_items.csv')
        self.itemsList = Importer.init_objectify_items_list(self.itemsList)
        self.addressesList = Importer.init_import_csv_as_list('resources/list_addresses.csv')
        self.addressesList = Importer.init_objectify_addresses_list(self.addressesList)

    def _init_warehouse(self):
        """Construct the simulation warehouse."""
        self.warehouse = Warehouse(
            self.warehouse_res,
            self.warehouse_windowBackgroundColour,
            self.warehouse_windowCenter,
            self.warehouse_windowArray,
            self.itemsList,
            self.addressesList,
            robotsMaxQuantity=18,
        )

    def _init_optimizer_and_paint(self):
        """Strategy-pattern optimizer + zone-paint handler + mouse callback."""
        self.optimizer = WarehouseOptimizer(tick_rate=int(1 / self.sim_frametime))
        self.optimizer.register(ZoneStrategy(self.warehouse_res))
        # Zone painting — targeting the zone-map sub-view (top-right panel)
        # zone-map panel rect in composite: x = mw + sw*2, y = 0, w = sw, h = sh
        _zone_view_rect = (
            self.warehouse_windowRes[0] + self.sub_windowRes[0] * 2,  # x0
            0,                                                          # y0
            self.sub_windowRes[0],                                      # pw
            self.sub_windowRes[1],                                      # ph
        )
        self.paint_handler = PaintHandler(self.warehouse, self.warehouse_res, _zone_view_rect, self.optimizer)
        cv2.setMouseCallback("warehouse", self._on_mouse)

    def _init_maps(self):
        """Auto-generate examples + auto-load user-supplied PNG maps (zone/charger/robot)."""
        # Zone map
        _example_path = "resources/zone_map_example.png"
        if not os.path.exists(_example_path):
            MapImporter.generate_example_png(_example_path, *self.warehouse_res)
        _map_path = "resources/zone_map.png"
        if os.path.exists(_map_path):
            gw, gh = self.warehouse_res
            _loaded = MapImporter.load(_map_path, gw, gh)
            if _loaded is not None:
                self.warehouse.zoneMap = _loaded
                self.warehouse.zoneMap_original = _loaded.copy()
                self.warehouse.update_zone_counts()
                log.info("Zone map applied from '%s'.", _map_path)
        # Charger spawn map
        _charger_example_path = "resources/charger_map_example.png"
        if not os.path.exists(_charger_example_path):
            MapImporter.generate_charger_map_example(_charger_example_path, *self.warehouse_res)
        _charger_map_path = "resources/charger_map.png"
        if os.path.exists(_charger_map_path):
            gw, gh = self.warehouse_res
            _charger_coords = MapImporter.load_spawn_map(_charger_map_path, gw, gh)
            if _charger_coords:                          # non-empty list = valid map
                self.warehouse.chargerSpawnMap = _charger_coords
                log.info("Charger spawn map applied from '%s' (%d cells).",
                         _charger_map_path, len(_charger_coords))
        # Robot spawn map
        _robot_example_path = "resources/robot_map_example.png"
        if not os.path.exists(_robot_example_path):
            MapImporter.generate_robot_map_example(_robot_example_path, *self.warehouse_res)
        _robot_map_path = "resources/robot_map.png"
        if os.path.exists(_robot_map_path):
            gw, gh = self.warehouse_res
            _robot_coords = MapImporter.load_spawn_map(_robot_map_path, gw, gh)
            if _robot_coords:
                self.warehouse.robotSpawnMap = _robot_coords
                log.info("Robot spawn map applied from '%s' (%d cells).",
                         _robot_map_path, len(_robot_coords))

    def initWarehouseWindow(self):
        """Compose-window setup: geometry, history buffers, UI state, OS window."""
        self._init_window_arrays()
        self._init_geometry()
        self._init_history_buffers()
        self._init_ui_state()
        self._init_os_window()
        return self

    def _init_window_arrays(self):
        self.warehouse_windowCenter = Functions.get_screencenter(self.warehouse_res)
        self.warehouse_windowArray = Functions.get_screenarray_colour(self.warehouse_res, self.warehouse_windowBackgroundColour)

    def _init_geometry(self):
        """Centralised pixel geometry — single source of truth for panel sizes."""
        self.geom = PanelGeometry.from_warehouse_window(self.warehouse_windowRes)
        self.sub_windowRes = (self.geom.sub_w, self.geom.sub_h)
        self.panel_h = self.geom.panel_h
        self.chart_h = self.geom.chart_h
        self.composite_windowRes = self.geom.composite_res

    def _init_history_buffers(self):
        """Rolling history arrays for charts (1800 samples @ 1/sec ≈ 30 min)."""
        _H = 1800
        self._hist_imported = np.zeros(_H, dtype=np.float32)
        self._hist_exported = np.zeros(_H, dtype=np.float32)
        self._hist_overdue  = np.zeros(_H, dtype=np.float32)
        self._hist_late     = np.zeros(_H, dtype=np.float32)  # cumulative ever-overdue count
        self._late_total    = 0
        self._hist_batt       = np.zeros(_H, dtype=np.float32)
        self._hist_robots     = np.zeros(_H, dtype=np.float32)
        self._hist_working    = np.zeros(_H, dtype=np.float32)
        self._hist_charging   = np.zeros(_H, dtype=np.float32)
        self._hist_idle       = np.zeros(_H, dtype=np.float32)
        # Flow-control history arrays
        self._hist_import_cap   = np.zeros(_H, dtype=np.float32)
        self._hist_throughput   = np.zeros(_H, dtype=np.float32)
        self._hist_delivery     = np.zeros(_H, dtype=np.float32)
        self._hist_overdue_pct  = np.zeros(_H, dtype=np.float32)
        self._hist_tick     = 0
        self._hist_count    = 0          # samples actually filled so far
        self._hist_min_win  = 30         # minimum visible window (seconds)
        self._hist_last_sample = 0.0     # wall-clock time of last 1 Hz sample
        # Rate-of-change tracking (snapshot every 1s sample)
        self._prev_stats = None   # dict of key→value at last sample
        self._rate_stats = {}     # dict of key→delta/sec

    def _init_ui_state(self):
        """Optimizer log + heatmap overlay toggles."""
        # Optimizer action log: ring buffer of (timestamp, action_str), newest last
        self._opt_log: list = []
        self._last_seen_opt_str = ''
        # Heatmap overlay toggles
        self._show_heatmap = True
        self._show_roads = True
        self._show_plazas = True

    def _init_os_window(self):
        """Create the borderless OpenCV window centred on the user's screen."""
        screen_w, screen_h = Display_Functions.get_screen_resolution()
        cx = (screen_w - self.composite_windowRes[0]) // 2
        cy = (screen_h - self.composite_windowRes[1]) // 2
        Display_Functions.init_borderless_window("warehouse", self.composite_windowRes, [cx, cy])
        log.info('- warehouse_res: %s, arrayShape: %s, composite_windowRes: %s, screen_center_pos: [%d,%d]',
                 self.warehouse_res, np.shape(self.warehouse_windowArray), self.composite_windowRes, cx, cy)

    # ── Mouse callback ──────────────────────────────────────────────────

    def _on_mouse(self, event, px, py, flags, param):
        """Composite cv2 mouse callback. Handles dashboard controls, then delegates."""
        if self._handle_display_fps_click(event, px, py):
            return
        if self._handle_power_policy_click(event, px, py):
            return
        if self._handle_flow_policy_click(event, px, py):
            return
        if self._handle_robot_count_click(event, px, py):
            return
        if self._handle_pkg_target_mode_click(event, px, py):
            return
        if self._handle_heatmap_toggle_click(event, px, py):
            return
        self.paint_handler.on_mouse(event, px, py, flags, param)

    def _set_display_fps(self, fps):
        if fps not in self._DISPLAY_FPS_OPTIONS:
            return
        self.draw_frametime = 1.0 / float(fps)
        log.info('Display FPS -> %s', fps)

    # Backwards-compatible re-exports — delegate to data.ui.layout /
    # data.ui.formatters so existing code paths and tests continue to work.
    _horizontal_button_layout = staticmethod(horizontal_button_layout)

    def _display_fps_button_layout(self):
        # Anchored to top-left of the main view so it does not overlap info-panel controls.
        return self._horizontal_button_layout(self._DISPLAY_FPS_OPTIONS, x0=106, y0=4, btn_w=22)

    def _handle_display_fps_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for fps, bx0, by0, bx1, by1 in self._display_fps_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                self._set_display_fps(fps)
                return True
        return False

    def _set_power_policy_mode(self, mode):
        if not self.warehouse.set_power_policy_mode(mode):
            return
        log.info('Power Policy -> %s (%s)', mode, POWER_POLICY_REASON[mode])

    def _set_flow_policy_mode(self, mode):
        if not self.warehouse.set_flow_policy_mode(mode):
            return
        log.info('Flow Policy -> %s (%s)', mode, FLOW_POLICY_REASON[mode])

    def _power_policy_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        btn_w = 34
        total_w = btn_w * len(self._POWER_POLICY_MODES) + 3 * (len(self._POWER_POLICY_MODES) - 1)
        x0 = col_w * 2 - 4 - total_w
        return self._horizontal_button_layout(self._POWER_POLICY_MODES, x0, mh + 3, btn_w=btn_w)

    def _flow_policy_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        btn_w = 34
        total_w = btn_w * len(self._FLOW_POLICY_MODES) + 3 * (len(self._FLOW_POLICY_MODES) - 1)
        x0 = col_w * 3 - 4 - total_w
        return self._horizontal_button_layout(self._FLOW_POLICY_MODES, x0, mh + 3, btn_w=btn_w)

    def _handle_power_policy_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for mode, bx0, by0, bx1, by1 in self._power_policy_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                self._set_power_policy_mode(mode)
                return True
        return False

    def _handle_flow_policy_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for mode, bx0, by0, bx1, by1 in self._flow_policy_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                self._set_flow_policy_mode(mode)
                return True
        return False

    def _set_pkg_target_mode(self, mode):
        if not self.warehouse.set_package_target_mode(mode):
            return
        mode = getattr(self.warehouse, '_pkg_target_mode', mode)
        log.info('Package target mode -> %s (%s)',
                 mode, PKG_TARGET_REASON.get(mode, 'user override'))

    def _pkg_target_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        btn_w = 38
        total_w = btn_w * len(self._PKG_TARGET_MODES) + 3 * (len(self._PKG_TARGET_MODES) - 1)
        x0 = cw - 4 - total_w
        return self._horizontal_button_layout(self._PKG_TARGET_MODES, x0, mh + 3, btn_w=btn_w)

    def _handle_pkg_target_mode_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for mode, bx0, by0, bx1, by1 in self._pkg_target_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                self._set_pkg_target_mode(mode)
                return True
        return False

    def _robot_count_button_layout(self):
        """Return [('-', bx0, by0, bx1, by1), ('+', bx0, by0, bx1, by1)].

        Layout: [-] <num> [+]  anchored to top-right of the ROBOTS column.
        """
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        pad_x = 4
        btn_w = 14
        btn_h = 11
        num_w = 18           # space for the target number between buttons
        gap = 2
        total_w = btn_w * 2 + num_w + gap * 2
        x0 = col_w - pad_x - total_w
        y0 = mh + 3
        return [
            ('-', x0,                              y0, x0 + btn_w,                          y0 + btn_h),
            ('+', x0 + btn_w + gap + num_w + gap,  y0, x0 + btn_w * 2 + num_w + gap * 2,   y0 + btn_h),
        ]

    def _handle_robot_count_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for action, bx0, by0, bx1, by1 in self._robot_count_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                target = self.warehouse._robot_target_count
                if action == '+':
                    self.warehouse.set_robot_count(target + 1)
                    log.info('Robot target -> %d (+1)', target + 1)
                elif action == '-' and target > 1:
                    self.warehouse.set_robot_count(target - 1)
                    log.info('Robot target -> %d (-1)', target - 1)
                return True
        return False

    # ── Heatmap overlay toggle buttons ──────────────────────────────────

    def _heatmap_toggle_button_layout(self):
        """Return [(key, label, bx0, by0, bx1, by1), ...] for heatmap/road/plaza toggles.

        Anchored to the bottom of the TRAFFIC HEATMAP panel, evenly spaced.
        """
        mw = self.warehouse_windowRes[0]  # 320
        sw = self.sub_windowRes[0]        # 160
        sh = self.sub_windowRes[1]        # 140
        btn_h = 11
        pad_x = 4
        pad_y = 3
        gap = 3
        # Panel bottom: y = 2*sh (second row ends there)
        y1 = 2 * sh - pad_y
        y0 = y1 - btn_h
        x_left = mw + sw * 2 + pad_x
        x_right = mw + sw * 3 - pad_x
        usable = x_right - x_left - gap * 2
        btn_w = usable // 3
        return [
            ('heatmap', 'HM',  x_left,                          y0, x_left + btn_w,              y1),
            ('roads',   'RD',  x_left + btn_w + gap,            y0, x_left + btn_w * 2 + gap,    y1),
            ('plazas',  'PLZ', x_left + btn_w * 2 + gap * 2,    y0, x_right,                     y1),
        ]

    def _handle_heatmap_toggle_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for key, _lbl, bx0, by0, bx1, by1 in self._heatmap_toggle_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                if key == 'roads':
                    self._show_roads = not self._show_roads
                elif key == 'plazas':
                    self._show_plazas = not self._show_plazas
                elif key == 'heatmap':
                    self._show_heatmap = not self._show_heatmap
                return True
        return False

    # ── Formatting helpers ──────────────────────────────────────────────
    # Pure formatters live in data.ui.formatters; expose as static methods
    # for backwards compatibility with existing call sites and tests.
    _fmt_elapsed = staticmethod(fmt_elapsed)
    _fmt_age     = staticmethod(fmt_age)
    _fmt_rate    = staticmethod(fmt_rate)
    _n           = staticmethod(fmt_n)

    def gameLoop(self, loop_count):
        """Execute one simulation tick and optionally repaint.

        Sequence: optimizer step → warehouse.update_warehouse() →
        accumulate robot heatmaps (fast + slow, fleet-scaled half-lives) →
        rate-limited gameDraw().  Sleeps to maintain the target sim_frametime.
        """
        timeLoopStart = time.time()

        self.timeElapsed = int(time.time() - self.timeStart)

        if self.timeElapsed > 24000:
            self.gameEnd()
            return self

        # Update Existing
        self.optimizer.step(self.warehouse, self.warehouse.traffic_ema)
        self.warehouse = self.warehouse.update_warehouse()

        # Draw (rate-limited independently of sim)
        now = time.time()
        if now - self._last_draw_time >= self.draw_frametime:
            self._last_draw_time = now
            self.gameDraw()

        frameTime = round(time.time() - timeLoopStart, 3)

        if (frameTime < self.sim_frametime):
            time.sleep(self.sim_frametime - frameTime)

        return self, frameTime

    def gameDraw(self):
        """Build and display the composite dashboard window.

        Layout (960×680 px):
          Row 0: Main view (320×280) | Chargers | Packages | Zone Map | Deadlines+Fleet Batt
          Row 1: (main view cont.)   | Robots   | Pkg Tgts | Heatmap  | Flow Ctrl
          Info panel: ROBOTS | CHARGERS | SIM STATS | PACKAGES (4×240 px)
          Chart strip: Exports&Overdue | Flow Control | Zone Bars | Fleet Health
        """
        # Draw main view
        self.warehouseWindow = self.warehouse_windowArray.copy()
        self.warehouseWindow = Draw_Warehouse.draw_warehouseZones(self.warehouseWindow, self.warehouse.zoneMap, {1: self.warehouse.colourOfImportAreas, 2: self.warehouse.colourOfStorageAreas, 3: self.warehouse.colourOfExportAreas})
        self.warehouseWindow = Draw_Warehouse.draw_chargers(self.warehouseWindow, self.warehouse)
        self.warehouseWindow = Draw_Warehouse.draw_packages(self.warehouseWindow, self.warehouse)
        self.warehouseWindow = Draw_Warehouse.draw_robots(self.warehouseWindow, self.warehouse)

        # Build composite window
        mw, mh = self.warehouse_windowRes   # 320, 280
        sw, sh = self.sub_windowRes         # 160, 140
        ph = self.panel_h                   # 165
        ch = self.chart_h                   # 95
        cw = mw + sw * 4                    # 960
        zone_colours = {1: self.warehouse.colourOfImportAreas, 2: self.warehouse.colourOfStorageAreas, 3: self.warehouse.colourOfExportAreas}
        wh = self.warehouse
        composite = np.zeros((mh + ph + ch, cw, 3), dtype=np.uint8)

        # Main view (left column)
        composite[0:mh, 0:mw] = cv2.resize(self.warehouseWindow, (mw, mh), interpolation=cv2.INTER_NEAREST)

        self._draw_main_view_overlays(composite, mw, mh)

        self._draw_subviews(composite, mw, mh, sw, sh)

        # ── Slot (row 1, col 3): DEADLINES (top) + FLEET BATT (bottom) ──────────────────
        # Panel is 160×140 px: top 70 px → deadline urgency histogram,
        # 1 px separator, bottom 69 px → fleet battery bar chart.
        _DL_H = 70
        _FB_H = sh - _DL_H - 1   # 69 px

        self._draw_deadlines_histogram(composite, mw, sw, _DL_H)

        # 1 px separator between DEADLINES and FLEET BATT
        composite[_DL_H, mw+sw*3:mw+sw*4] = (45, 45, 45)

        self._draw_fleet_battery(composite, mw, sw, sh, _DL_H, _FB_H)

        self._draw_flow_control(composite, mw, sw, sh)

        # Zone map (top-right) — reserve BUTTON_H at bottom and BRUSH_W on right
        _btn_h   = self.paint_handler.BUTTON_H
        _brush_w = self.paint_handler.BRUSH_W
        _map_w   = sw - _brush_w
        _map_h   = sh - _btn_h
        # Brighten zone colours for the sub-view (main view uses very dark colours
        # so packages/robots are visible on top; the zone map has no entities so
        # can afford much higher contrast against the dark background).
        _zm_cols = {k: tuple(min(int(c * 5), 160) for c in v) for k, v in zone_colours.items()}
        zoneMap_display = Draw_Warehouse.draw_zoneMap_display(self.warehouse.zoneMap, _zm_cols)
        composite[0:_map_h, mw+sw*2:mw+sw*2+_map_w] = cv2.resize(zoneMap_display, (_map_w, _map_h), interpolation=cv2.INTER_NEAREST)

        # Zone utilization % legend on ZONE MAP — fixed stacked column, bottom-left of map area
        _zs = next((s for s in self.optimizer.strategies if s.name == 'Zone'), None)
        _util_overlay_active = _zs and _zs._total >= _zs.WARMUP_TICKS
        # Append new optimizer actions to the log ring buffer
        if _zs and _zs._last_action_str and _zs._last_action_str != self._last_seen_opt_str:
            self._last_seen_opt_str = _zs._last_action_str
            self._opt_log.append((time.time(), _zs._last_action_str))
            if len(self._opt_log) > 20:
                self._opt_log.pop(0)
        if _util_overlay_active:
            _ufont = cv2.FONT_HERSHEY_SIMPLEX
            _ufs   = 0.28
            _ulent = [
                (1, 'IMP', self.warehouse.colourOfImportAreas),
                (2, 'STO', self.warehouse.colourOfStorageAreas),
                (3, 'EXP', self.warehouse.colourOfExportAreas),
            ]
            # Measure max line width for background rect
            _u_lh = 11
            _u_pad = 3
            _u_lines = []
            for _zid_u, _abbr_u, _zcol_u in _ulent:
                _upct = int(_zs._util_ema[_zid_u] * 100)
                _u_lines.append((_abbr_u, _upct, _zcol_u, _zs._util_ema[_zid_u]))
            _u_max_w = max(
                cv2.getTextSize('{} {}%'.format(a, p), _ufont, _ufs, 1)[0][0]
                for a, p, _, __ in _u_lines
            )
            _zmap_x0 = mw + sw * 2
            _u_x0 = _zmap_x0 + _u_pad
            _u_y0 = _map_h - len(_u_lines) * _u_lh - _u_pad
            # Semi-transparent dark background
            _ubg_x1 = _u_x0 + _u_max_w + _u_pad * 2 + 6
            _ubg_y1 = _map_h
            _u_roi = composite[_u_y0:_ubg_y1, _u_x0 - _u_pad:_ubg_x1]
            composite[_u_y0:_ubg_y1, _u_x0 - _u_pad:_ubg_x1] = (_u_roi * 0.3).astype(np.uint8)
            for _i, (_abbr_u, _upct, _zcol_u, _uval) in enumerate(_u_lines):
                _uy = _u_y0 + (_i + 1) * _u_lh - 1
                # Colour swatch
                _bright_u = tuple(min(int(c * 2.2), 255) for c in _zcol_u)
                cv2.rectangle(composite, (_u_x0, _uy - 7), (_u_x0 + 4, _uy - 2), _bright_u, -1)
                # Value colour by threshold
                if _uval > _zs.HIGH_THRESH:
                    _ucol = (60, 60, 220)    # red — bottleneck
                elif _uval < _zs.LOW_THRESH:
                    _ucol = (50, 200, 220)   # yellow — under-used
                else:
                    _ucol = (170, 170, 170)  # neutral grey
                cv2.putText(composite, '{} {}%'.format(_abbr_u, _upct),
                            (_u_x0 + 7, _uy), _ufont, _ufs, _ucol, 1)

        self._draw_subview_titles(composite, mw, sw, sh, _util_overlay_active)

        self._draw_heatmap_toggles(composite, _tfont=cv2.FONT_HERSHEY_SIMPLEX)

        self._draw_zone_legend(composite, mw, mh)

        # SIM STATS data (rendered in info panel below)
        now_t = time.time()
        r = self._rate_stats  # may be empty dict at startup
        _n = self._n

        # Youngest / oldest packages by age (with pkg number)
        if wh.packages:
            youngest_p = min(wh.packages, key=lambda p: now_t - p.createdAt)
            oldest_p   = max(wh.packages, key=lambda p: now_t - p.createdAt)
            young_pkg = "P{} {}".format(youngest_p.packageNumber, self._fmt_age(now_t - youngest_p.createdAt))
            old_pkg   = "P{} {}".format(oldest_p.packageNumber,   self._fmt_age(now_t - oldest_p.createdAt))
        else:
            young_pkg = "-"
            old_pkg   = "-"

        def _r(key):
            v = r.get(key, 0)
            return self._fmt_rate(v)

        # Fleet summary (single-pass aggregate, reused by INFO PANEL)
        fleet = build_fleet_snapshot(wh.robots)
        # Optimizer / zone utilization state (reuse cached _zs from overlay block)
        if _zs and _zs._total >= _zs.WARMUP_TICKS:
            _util_str = "I:{} S:{} E:{}".format(
                int(_zs._util_ema[1] * 100),
                int(_zs._util_ema[2] * 100),
                int(_zs._util_ema[3] * 100))
            if _zs._last_action_str:
                _opt_age = now_t - _zs._last_action_time
                _opt_str = "{} {:.0f}s".format(_zs._last_action_str, _opt_age)
            else:
                _opt_str = "no change yet"
        elif _zs:
            _rem = max(0, _zs.WARMUP_TICKS - _zs._total)
            _util_str = "warmup {}t".format(_rem)
            _opt_str  = "off" if not _zs.enabled else "waiting"
        else:
            _util_str = "---"
            _opt_str  = "---"

        _pkg_target_mode = getattr(wh, '_pkg_target_mode', 'random')
        _pkg_style = PKG_TARGET_STYLE.get(_pkg_target_mode, PKG_TARGET_STYLE['random'])
        if _pkg_target_mode == 'nearest':
            _target_str = 'nearest (min dist)'
            _target_col = _pkg_style['text']
        elif _pkg_target_mode == 'zone_edge':
            _target_str = 'edge (pipeline)'
            _target_col = _pkg_style['text']
        else:
            _target_str = 'random (spread)'
            _target_col = _pkg_style['text']

        _flow_mode = getattr(wh, '_flow_policy_mode', 'balanced')
        _flow_style = FLOW_POLICY_STYLE.get(_flow_mode, FLOW_POLICY_STYLE['balanced'])
        _flow_col = _flow_style['text']

        _power_mode = getattr(wh, '_power_policy_mode', 'balanced')
        _power_style = POWER_POLICY_STYLE.get(_power_mode, POWER_POLICY_STYLE['balanced'])
        _power_col = _power_style['text']

        # Determine colour for optimizer lines based on state
        _opt_active  = _zs and _zs.enabled and _zs._total >= _zs.WARMUP_TICKS
        _util_col    = (80, 210, 180) if _opt_active else (120, 120, 120)   # teal / dim
        _opt_col     = (80, 210, 180) if (_opt_active and _zs._last_action_str) else (120, 120, 120)

        # (label, value, optional custom colour)
        stats_lines = [
            ("elapsed",  self._fmt_elapsed(self.timeElapsed), None),
            ("loops",    "{} ({})".format(_n(wh.warehouseLoopCount),    _r("loops")), None),
            ("exported", "{} ({})".format(_n(wh.packageExportRollingCount), _r("export")), None),
            ("pkgs",     "{}/{} ({})".format(_n(wh.packagesInWarehouseCount), _n(wh.packagesMaxQuantity), _r("pkgs")), None),
            ("robots",   "idle:{}/{} b:{}%".format(fleet.n_idle, fleet.n, int(fleet.avg_battery)), None),
            ("import",   "{}/{} ({})".format(_n(wh.packagesInImportCount),  _n(wh.numberOfImportSlots),  _r("import")), None),
            ("storage",  "{}/{} ({})".format(_n(wh.packagesInStorageCount), _n(wh.numberOfStorageSlots), _r("storage")), None),
            ("export",   "{}/{} ({})".format(_n(wh.packagesInExportCount),  _n(wh.numberOfExportSlots),  _r("expzone")), None),
            ("util",     _util_str, _util_col),
            ("opt",      _opt_str,  _opt_col),
            ("power",    _power_mode, _power_col),
            ("flow",     "{} c:{}".format(_flow_mode, int(getattr(wh, '_flow_import_cap', 0))), _flow_col),
            ("target",   _target_str, _target_col),
        ]

        self._draw_info_panel(composite, mw, mh, sw, sh, ph, cw, stats_lines, now_t, fleet)

        # Update rolling chart histories (sample once per second via wall-clock)
        self._sample_chart_histories()

        # Draw charts strip
        composite[mh + ph, :] = 55  # chart separator
        self._draw_charts(composite, mh + ph + 1, ch - 1, cw)

        # Re-draw separators and column dividers last so they sit on top of chart content
        col_w = cw // 4
        composite[mh, :]                         = 60
        composite[mh + ph, :]                    = 55
        composite[mh:mh + ph, col_w]             = 40
        composite[mh:mh + ph, col_w * 2]         = 40
        composite[mh:mh + ph, col_w * 3]         = 40

        self._draw_panel_outlines(composite, mw, mh, sw, sh, ph, ch, cw)

        # Zone-map panel controls (zone buttons at bottom, brush sizes on right)
        self.paint_handler.draw_buttons(composite, cv2.FONT_HERSHEY_SIMPLEX)

        self._finalize_and_display(composite)
        return self

    def _draw_subviews(self, composite, mw, mh, sw, sh):
        """Render binary grids, robot paths, traffic heatmap, and package targets."""
        def place_binary(arr, row, col):
            img = cv2.resize((arr * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            composite[row*sh:(row+1)*sh, mw+col*sw:mw+(col+1)*sw] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        place_binary(self.warehouse.chargersInWarehouse,        0, 0)
        place_binary(self.warehouse.packagesInWarehouse,        0, 1)
        place_binary(self.warehouse.robotsInWarehouse,          1, 0)

        _sub_ox = mw
        _sub_oy = sh
        _sub_sx = sw / max(self.warehouse_res[0], 1)
        _sub_sy = sh / max(self.warehouse_res[1], 1)
        _sub_cell = min(_sub_sx, _sub_sy)
        _sub_r    = max(int(_sub_cell * 0.55), 1)

        # Subtle dark-grey road underlay on robots panel
        _road_rb = self.warehouse.road_map
        if _road_rb is not None and _road_rb.any():
            _road_rb_rsz = cv2.resize(_road_rb, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _rb_panel = composite[sh:2*sh, mw:mw+sw]
            _rb_road_mask = _road_rb_rsz > 0
            _rb_panel[_rb_road_mask] = np.clip(
                _rb_panel[_rb_road_mask].astype(np.int16) + 30, 0, 255).astype(np.uint8)

        # Speed-profile legend
        _sp_x0 = _sub_ox + sw - 50
        _sp_y0 = _sub_oy + sh - 10
        _sp_w = 36
        _sp_h = 5
        cv2.rectangle(composite, (_sp_x0 - 1, _sp_y0 - 1), (_sp_x0 + _sp_w + 1, _sp_y0 + _sp_h + 1), (18, 18, 18), -1)
        for _gi in range(_sp_w):
            _gt = _gi / max(_sp_w - 1, 1)
            _gt = max(0.0, min(1.0, _gt))
            if _gt <= 0.5:
                _gs = _gt * 2.0
                _gc = (int(60 - 10 * _gs), int(80 + 120 * _gs), int(220))
            else:
                _gs = (_gt - 0.5) * 2.0
                _gc = (int(50 + 180 * _gs), int(200 + 10 * _gs), int(220 - 150 * _gs))
            cv2.line(composite, (_sp_x0 + _gi, _sp_y0), (_sp_x0 + _gi, _sp_y0 + _sp_h - 1), _gc, 1)
        cv2.putText(composite, 'spd', (_sp_x0 + _sp_w + 2, _sp_y0 + _sp_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (100, 100, 100), 1)

        def _speed_to_bgr(t):
            t = max(0.0, min(1.0, t))
            if t <= 0.5:
                s = t * 2.0
                return (int(60 - 10 * s), int(80 + 120 * s), int(220))
            s = (t - 0.5) * 2.0
            return (int(50 + 180 * s), int(200 + 10 * s), int(220 - 150 * s))

        for _rb in self.warehouse.robots:
            _path_cells = [list(_rb.xyLocation)]
            _rb_path = list(getattr(_rb, 'path', []))
            if _rb_path:
                _path_cells.extend([list(_p) for _p in _rb_path])
            elif getattr(_rb, 'xyLocationTarget', None) and list(_rb.xyLocationTarget) != list(_rb.xyLocation):
                _path_cells.append(list(_rb.xyLocationTarget))
            elif getattr(_rb, 'pathTarget', None):
                _path_cells.append(list(_rb.pathTarget))
            elif getattr(_rb, 'actionQueue', None):
                _path_cells.append(list(_rb.actionQueue[0][0]))

            _rb_plan = list(getattr(_rb, 'pathPlan', []))
            _rb_max_v = max(float(getattr(_rb, 'maxVelocity', 0.6)), 0.01)
            if len(_path_cells) >= 2:
                for _seg_i in range(len(_path_cells) - 1):
                    _p0 = _path_cells[_seg_i]
                    _p1 = _path_cells[_seg_i + 1]
                    _sx0 = int(_p0[0] * _sub_sx + _sub_sx * 0.5) + _sub_ox
                    _sy0 = int(_p0[1] * _sub_sy + _sub_sy * 0.5) + _sub_oy
                    _sx1 = int(_p1[0] * _sub_sx + _sub_sx * 0.5) + _sub_ox
                    _sy1 = int(_p1[1] * _sub_sy + _sub_sy * 0.5) + _sub_oy
                    _plan_idx = _seg_i
                    if _rb_plan and 0 <= _plan_idx < len(_rb_plan):
                        _seg_spd = _rb_plan[_plan_idx].get('speed', _rb_max_v * 0.5)
                    else:
                        _seg_spd = _rb_max_v * 0.5
                    _t = _seg_spd / _rb_max_v
                    _seg_col = _speed_to_bgr(_t)
                    cv2.line(composite, (_sx0, _sy0), (_sx1, _sy1), _seg_col, 1, cv2.LINE_AA)

                for _mi in range(len(_rb_plan)):
                    if _rb_plan[_mi].get('turn_deg', 0) >= 40.0:
                        _pc = _path_cells[_mi + 1] if _mi + 1 < len(_path_cells) else None
                        if _pc is not None:
                            _mx = int(_pc[0] * _sub_sx + _sub_sx * 0.5) + _sub_ox
                            _my = int(_pc[1] * _sub_sy + _sub_sy * 0.5) + _sub_oy
                            _mt = _rb_plan[_mi].get('speed', _rb_max_v * 0.5) / _rb_max_v
                            cv2.circle(composite, (_mx, _my), max(int(_sub_cell * 0.3), 1),
                                       _speed_to_bgr(_mt), -1)

            _cx = int(_rb.xyLocation[0] * _sub_sx + _sub_sx * 0.5) + _sub_ox
            _cy = int(_rb.xyLocation[1] * _sub_sy + _sub_sy * 0.5) + _sub_oy
            cv2.circle(composite, (_cx, _cy), _sub_r + 1, (0, 0, 0), -1)
            cv2.circle(composite, (_cx, _cy), _sub_r, (220, 220, 220), -1)

        # Traffic heatmap
        if self._show_heatmap:
            _hm_src = self.warehouse.traffic_ema
            _hm_max = _hm_src.max()
            if _hm_max > 0:
                _hm_visited = _hm_src > 0
                _hm_pct = float(np.percentile(_hm_src[_hm_visited], 98)) if _hm_visited.any() else _hm_max
                _hm_clip = max(_hm_pct, _hm_max * 0.05)
                _hm_norm = np.clip((_hm_src / _hm_clip) ** 1.5 * 255, 0, 255).astype(np.uint8)
            else:
                _hm_norm = np.zeros_like(_hm_src, dtype=np.uint8)
            _hm_resized = cv2.resize(_hm_norm, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _hm_col = cv2.applyColorMap(_hm_resized, cv2.COLORMAP_JET)
            _hm_zero_mask = cv2.resize(
                (_hm_norm == 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            _hm_col[_hm_zero_mask > 0] = (0, 0, 0)
            composite[sh:2*sh, mw+sw*2:mw+sw*3] = _hm_col

            _slow_src = self.warehouse.traffic_total.astype(np.float32)
            _slow_max = _slow_src.max()
            if _slow_max > 0:
                _slow_visited = _slow_src > 0
                _slow_pct  = float(np.percentile(_slow_src[_slow_visited], 98)) if _slow_visited.any() else _slow_max
                _slow_clip = max(_slow_pct, _slow_max * 0.05)
                _slow_norm = np.clip((_slow_src / _slow_clip) ** 1.5 * 255, 0, 255).astype(np.uint8)
                _slow_rsz  = cv2.resize(_slow_norm, (sw, sh), interpolation=cv2.INTER_NEAREST)
                _slow_ov   = cv2.cvtColor(_slow_rsz, cv2.COLOR_GRAY2BGR)
                _slow_zero = cv2.resize(
                    (_slow_norm == 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                _slow_ov[_slow_zero > 0] = (0, 0, 0)
                composite[sh:2*sh, mw+sw*2:mw+sw*3] = cv2.addWeighted(
                    composite[sh:2*sh, mw+sw*2:mw+sw*3], 1.0, _slow_ov, 0.30, 0)

        # Road network overlay
        _road = self.warehouse.road_map
        if self._show_roads and _road is not None and _road.any():
            _road_rsz = cv2.resize(_road, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _panel = composite[sh:2*sh, mw+sw*2:mw+sw*3]
            _road_alpha = np.zeros((sh, sw), dtype=np.float32)
            _road_alpha[_road_rsz == 1] = 0.20
            _road_alpha[_road_rsz == 2] = 0.30
            _road_alpha[_road_rsz >= 3] = 0.42
            _alpha_3ch = _road_alpha[:, :, np.newaxis]
            _white = np.full_like(_panel, 255, dtype=np.uint8)
            _panel[:] = np.clip(
                _panel.astype(np.float32) * (1.0 - _alpha_3ch) + _white.astype(np.float32) * _alpha_3ch,
                0, 255).astype(np.uint8)
            _sx = sw / max(self.warehouse_res[0], 1)
            _sy = sh / max(self.warehouse_res[1], 1)
            _hs_list = self.warehouse.road_hotspots
            _hs_top = _hs_list[0][2] if _hs_list else 1.0
            for _hx, _hy, _hi in _hs_list:
                _px = int(_hx * _sx + _sx * 0.5)
                _py = int(_hy * _sy + _sy * 0.5) + sh
                if 0 <= _px < sw and sh <= _py < 2 * sh:
                    _ratio = _hi / max(_hs_top, 1e-9)
                    _hcol = (0, 220, 80) if _ratio >= 0.5 else (160, 160, 0)
                    cv2.circle(composite, (mw + sw * 2 + _px, _py), 1, _hcol, -1)

        # Plaza overlay
        _plaza = self.warehouse.plaza_map
        if self._show_plazas and _plaza is not None and _plaza.any():
            _plz_rsz = cv2.resize(_plaza, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _panel = composite[sh:2*sh, mw+sw*2:mw+sw*3]
            _plz_mask = _plz_rsz > 0
            _plz_tint = np.array([180, 140, 60], dtype=np.float32)
            _plz_alpha = 0.25
            _panel[_plz_mask] = np.clip(
                _panel[_plz_mask].astype(np.float32) * (1.0 - _plz_alpha)
                + _plz_tint * _plz_alpha,
                0, 255).astype(np.uint8)

        # Package targets sub-view
        sub_scale = sw // self.warehouse_res[0]
        pkg_tgt_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        Draw_Warehouse.draw_package_arrows(pkg_tgt_img, self.warehouse.packages, sub_scale)
        tgt_resized = cv2.resize((self.warehouse.packageTargetsInWarehouse * 255).astype(np.uint8),
                                  (sw, sh), interpolation=cv2.INTER_NEAREST)
        pkg_tgt_img[tgt_resized > 0] = (255, 255, 255)
        composite[sh:2*sh, mw+sw:mw+2*sw] = pkg_tgt_img

    def _sample_chart_histories(self):
        """Sample rolling chart histories once per second (wall-clock)."""
        if time.time() - self._hist_last_sample < 1.0:
            return
        self._hist_last_sample = time.time()
        wh = self.warehouse
        overdue = sum(1 for p in wh.packages if p.timeToDeadline.total_seconds() < 0)
        self._late_total += overdue
        mean_batt = float(np.mean([r.batteryPercent for r in wh.robots])) if wh.robots else 0.0
        charging    = sum(1 for r in wh.robots if r.status == "charging")
        idle_count  = sum(1 for r in wh.robots if r.status == "idle")
        working     = len(wh.robots) - charging - idle_count
        self._hist_imported = np.roll(self._hist_imported, -1); self._hist_imported[-1] = wh.packagesRollingCount
        self._hist_exported = np.roll(self._hist_exported, -1); self._hist_exported[-1] = wh.packageExportRollingCount
        self._hist_overdue  = np.roll(self._hist_overdue,  -1); self._hist_overdue[-1]  = overdue
        self._hist_late     = np.roll(self._hist_late,     -1); self._hist_late[-1]     = self._late_total
        self._hist_batt         = np.roll(self._hist_batt,         -1); self._hist_batt[-1]         = mean_batt
        self._hist_robots       = np.roll(self._hist_robots,       -1); self._hist_robots[-1]       = len(wh.robots)
        self._hist_working      = np.roll(self._hist_working,      -1); self._hist_working[-1]      = working
        self._hist_charging     = np.roll(self._hist_charging,     -1); self._hist_charging[-1]     = charging
        self._hist_idle         = np.roll(self._hist_idle,         -1); self._hist_idle[-1]         = idle_count
        self._hist_import_cap   = np.roll(self._hist_import_cap,   -1); self._hist_import_cap[-1]   = getattr(wh, '_flow_import_cap', wh.packagesMaxQuantity)
        self._hist_throughput   = np.roll(self._hist_throughput,   -1); self._hist_throughput[-1]   = getattr(wh, '_flow_throughput', 0) * 60
        self._hist_delivery     = np.roll(self._hist_delivery,     -1); self._hist_delivery[-1]     = getattr(wh, '_flow_avg_delivery', 0)
        n_pkgs = max(len(wh.packages), 1)
        self._hist_overdue_pct  = np.roll(self._hist_overdue_pct,  -1); self._hist_overdue_pct[-1]  = overdue / n_pkgs * 100
        self._hist_count    = min(self._hist_count + 1, len(self._hist_exported))
        cur = {
            "loops":    wh.warehouseLoopCount,
            "export":   wh.packageExportRollingCount,
            "pkgs":     wh.packagesInWarehouseCount,
            "robots":   len(wh.robots),
            "import":   wh.packagesInImportCount,
            "storage":  wh.packagesInStorageCount,
            "expzone":  wh.packagesInExportCount,
            "overdue":  overdue,
            "tot_imp":  wh.packagesRollingCount,
            "tot_late": self._late_total,
            "batt":     mean_batt,
            "working":  working,
            "chging":   charging,
            "idle":     idle_count,
        }
        if self._prev_stats is not None:
            self._rate_stats = {k: cur[k] - self._prev_stats.get(k, cur[k]) for k in cur}
        self._prev_stats = cur

    def _draw_charts(self, canvas, y_off, ch, cw):
        """Render four side-by-side charts into canvas starting at row y_off."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        _n   = self._n
        w4   = cw // 4
        y0, y1 = y_off, y_off + ch

        # Rate helper (mirrors the one in gameDraw)
        rs = self._rate_stats
        def _r(key):
            v = rs.get(key, 0)
            return self._fmt_rate(v)

        # Dynamic window: start at min_win seconds, grow to full history length
        window = max(self._hist_min_win, self._hist_count)
        span_lbl = self._fmt_elapsed(window)

        def mini_chart(x0, x1, title, series, shared_scale=None, fixed_scales=None, rates=None):
            """Draw a line chart. *rates* is an optional dict {label: rate_string}
            appended to the value text beside each series label."""
            w, h = x1 - x0, y1 - y0
            arr = np.full((h, w, 3), (8, 8, 8), dtype=np.uint8)
            # Pre-compute shared range if requested
            s_min, s_rng = 0.0, 1.0
            if shared_scale:
                shared_views = [data[-window:] for data, _, label in series if label in shared_scale]
                if shared_views:
                    s_min = float(min(np.min(v) for v in shared_views))
                    s_max = float(max(np.max(v) for v in shared_views))
                    s_rng = max(s_max - s_min, 1.0)
            # --- Pass 1: draw chart lines first ---
            for data, col, label in series:
                view = data[-window:]
                if fixed_scales and label in fixed_scales:
                    dmin, dmax = fixed_scales[label]
                    rng = max(float(dmax) - float(dmin), 1.0)
                    dmin = float(dmin)
                elif shared_scale and label in shared_scale:
                    dmin, rng = s_min, s_rng
                else:
                    dmin = float(np.min(view))
                    rng  = max(float(np.max(view)) - dmin, 1.0)
                n  = len(view)
                xs = np.linspace(1, w - 2, n).astype(int)
                ys = np.clip(
                    (h - 3 - ((view.astype(float) - dmin) / rng * (h - 8))).astype(int),
                    1, h - 2)
                for i in range(n - 1):
                    cv2.line(arr, (int(xs[i]), int(ys[i])), (int(xs[i+1]), int(ys[i+1])), col, 1)
            # --- Pass 2: per-line tight backdrop behind text, then draw text on top ---
            def _backdrop(ty, txt, fs=0.3):
                """Darken a tight rect behind a text line before drawing it."""
                (tw, th), baseline = cv2.getTextSize(txt, font, fs, 1)
                bx0 = max(2, 4 - 2)
                by0 = max(1, ty - th - 1)
                bx1 = min(w - 1, 4 + tw + 2)
                by1 = min(h - 1, ty + baseline + 1)
                roi = arr[by0:by1, bx0:bx1]
                roi[:] = (roi.astype(np.int16) * 3 // 10).clip(0, 255).astype(np.uint8)
            title_str = "{} ({})".format(title, span_lbl)
            _backdrop(11, title_str, 0.30)
            cv2.putText(arr, title_str, (4, 11), font, 0.30, (185, 185, 185), 1)
            label_y = 25
            for data, col, label in series:
                view = data[-window:]
                val_str = "{}: {}".format(label, _n(int(view[-1])))
                if rates and label in rates:
                    val_str += " ({})".format(rates[label])
                # Clip label to chart width before drawing
                max_lbl_w = w - 8
                (lw, _), _ = cv2.getTextSize(val_str, font, 0.3, 1)
                if lw > max_lbl_w:
                    paren = val_str.rfind(" (")
                    if paren != -1:
                        val_str = val_str[:paren]
                    (lw, _), _ = cv2.getTextSize(val_str, font, 0.3, 1)
                    if lw > max_lbl_w:
                        while len(val_str) > 1:
                            val_str = val_str[:-1]
                            (lw, _), _ = cv2.getTextSize(val_str + "\u2026", font, 0.3, 1)
                            if lw <= max_lbl_w:
                                val_str += "\u2026"
                                break
                _backdrop(label_y, val_str)
                cv2.putText(arr, val_str, (4, label_y), font, 0.3, col, 1)
                label_y += 10
            canvas[y0:y1, x0:x1] = arr

        # Chart 1: Imported/Exports/Tot Late share one y-scale; Overdue (instantaneous) on its own
        mini_chart(0, w4, "Imports, Exports & Overdue",
                   [(self._hist_imported, ( 60, 200,  60), "Imported"),   # green  – matches import zone
                    (self._hist_exported, ( 60,  60, 200), "Exported"),   # red    – matches export zone
                    (self._hist_late,     ( 40, 160, 240), "Tot Late"),   # orange – cumulative overdue
                    (self._hist_overdue,  (180, 100, 240), "Overdue")],   # pink   – instantaneous overdue
                   shared_scale={"Imported", "Exported", "Tot Late"},
                   rates={"Imported": _r("tot_imp"), "Exported": _r("export"), "Tot Late": _r("tot_late"), "Overdue": _r("overdue")})

        # Chart 2: Flow Control (adaptive import throttling)
        mini_chart(w4, w4 * 2, "Flow Control",
                   [(self._hist_import_cap,  ( 80, 190,  80), "Cap"),
                    (self._hist_throughput,   (200, 180,  60), "Thr/m"),
                    (self._hist_delivery,     ( 50, 160, 220), "Deliv"),
                    (self._hist_overdue_pct,  ( 80,  80, 210), "Ovrd%")],
                   fixed_scales={"Ovrd%": (0, 100)})

        # Chart 3: Zone capacity fill bars
        self._draw_zone_bars(canvas, w4 * 2, y0, w4 * 3, y1,
                             zone_rates={"Imp": _r("import"), "Sto": _r("storage"), "Exp": _r("expzone")},
                             span_lbl=span_lbl)

        # Chart 4: Fleet — Batt% on fixed scale; Working/Charging/Idle share robot-count scale
        mini_chart(w4 * 3, cw, "Fleet Health",
                   [(self._hist_batt,        ( 50, 200, 200), "Batt%"),
                    (self._hist_working,     ( 60, 180,  60), "Working"),
                    (self._hist_charging,    ( 60, 220, 240), "Chging"),
                    (self._hist_idle,        (130, 130, 130), "Idle")],
                   shared_scale={"Working", "Chging", "Idle"},
                   fixed_scales={"Batt%": (0, 100)},
                   rates={"Batt%": _r("batt"), "Working": _r("working"), "Chging": _r("chging"), "Idle": _r("idle")})

    def _draw_zone_bars(self, canvas, x0, y0, x1, y1, zone_rates=None, span_lbl=""):
        """Render zone capacity fill-bars into canvas[y0:y1, x0:x1]."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        w, h  = x1 - x0, y1 - y0
        arr   = np.full((h, w, 3), (8, 8, 8), dtype=np.uint8)
        # Title — match mini_chart style: backdrop + same font/colour
        _title_str = "Zone Capacity ({})".format(span_lbl) if span_lbl else "Zone Capacity"
        (tw, th), baseline = cv2.getTextSize(_title_str, font, 0.30, 1)
        bx0b = max(2, 2)
        by0b = max(1, 11 - th - 1)
        bx1b = min(w - 1, 4 + tw + 2)
        by1b = min(h - 1, 11 + baseline + 1)
        roi = arr[by0b:by1b, bx0b:bx1b]
        roi[:] = (roi.astype(np.int16) * 3 // 10).clip(0, 255).astype(np.uint8)
        cv2.putText(arr, _title_str, (4, 11), font, 0.30, (185, 185, 185), 1)
        wh = self.warehouse
        zones = [
            ("Imp",     wh.packagesInImportCount,  wh.numberOfImportSlots,  (200, 130,  60)),
            ("Sto",     wh.packagesInStorageCount, wh.numberOfStorageSlots, ( 80, 180, 200)),
            ("Exp",     wh.packagesInExportCount,  wh.numberOfExportSlots,  ( 80, 120, 220)),
        ]
        lbl_w     = 30                         # wide enough for "Sto" at fs=0.3
        bar_max_w = w - lbl_w - 48 - 4          # leaves right margin for count/rate
        bar_h     = 14
        bar_slot  = 26                           # bar_h + gap; 3 slots = 78px, fits in 94px with header
        for i, (name, used, total, col) in enumerate(zones):
            by   = 15 + i * bar_slot
            frac = min(used / max(total, 1), 1.0)
            fill = int(frac * bar_max_w)
            # track
            cv2.rectangle(arr, (lbl_w, by), (lbl_w + bar_max_w, by + bar_h), (25, 25, 25), -1)
            # fill — colour shifts green→yellow→red with occupancy
            if frac < 0.6:   fc = ( 80, 190,  80)
            elif frac < 0.85: fc = ( 50, 200, 200)
            else:              fc = ( 80,  80, 200)
            if fill > 0:
                cv2.rectangle(arr, (lbl_w, by), (lbl_w + fill, by + bar_h), fc, -1)
            # percentage text centred in bar
            pct_str = "{:.0f}%".format(frac * 100)
            (tw, _), _ = cv2.getTextSize(pct_str, font, 0.3, 1)
            cv2.putText(arr, pct_str, (lbl_w + bar_max_w // 2 - tw // 2, by + bar_h - 4),
                        font, 0.3, (200, 200, 200), 1)
            # zone label left; count + rate as a right-aligned pair centred beside bar
            cv2.putText(arr, name, (2, by + bar_h - 4), font, 0.3, col, 1)
            count_str = "{}/{}".format(used, total)
            (ctw, _), _ = cv2.getTextSize(count_str, font, 0.28, 1)
            cv2.putText(arr, count_str, (w - 3 - ctw, by + bar_h // 2 - 1),
                        font, 0.28, (120, 120, 120), 1)
            if zone_rates and name in zone_rates:
                rate_str = zone_rates[name]
                (rtw, _), _ = cv2.getTextSize(rate_str, font, 0.25, 1)
                cv2.putText(arr, rate_str, (w - 3 - rtw, by + bar_h // 2 + 10),
                            font, 0.25, (75, 75, 75), 1)

        # ── Summary section below the bars ──
        _sum_y = 15 + 3 * bar_slot + 2           # 2px gap below last bar

        # Total warehouse utilization bar
        _tot_used  = wh.packagesInWarehouseCount
        _tot_slots = wh.packagesMaxQuantity
        _tot_frac  = min(_tot_used / max(_tot_slots, 1), 1.0)
        _tot_bar_h = 10
        cv2.putText(arr, "Tot", (2, _sum_y + _tot_bar_h - 3), font, 0.28, (160, 160, 160), 1)
        cv2.rectangle(arr, (lbl_w, _sum_y), (lbl_w + bar_max_w, _sum_y + _tot_bar_h), (25, 25, 25), -1)
        if _tot_frac < 0.6:    _tfc = (80, 190, 80)
        elif _tot_frac < 0.85: _tfc = (50, 200, 200)
        else:                   _tfc = (80, 80, 200)
        _tot_fill = int(_tot_frac * bar_max_w)
        if _tot_fill > 0:
            cv2.rectangle(arr, (lbl_w, _sum_y), (lbl_w + _tot_fill, _sum_y + _tot_bar_h), _tfc, -1)
        _tot_txt = "{:.0f}%".format(_tot_frac * 100)
        (_ttw, _), _ = cv2.getTextSize(_tot_txt, font, 0.28, 1)
        cv2.putText(arr, _tot_txt, (lbl_w + bar_max_w // 2 - _ttw // 2, _sum_y + _tot_bar_h - 3),
                    font, 0.28, (200, 200, 200), 1)
        _tot_cnt = "{}/{}".format(_tot_used, _tot_slots)
        (_tcw, _), _ = cv2.getTextSize(_tot_cnt, font, 0.25, 1)
        cv2.putText(arr, _tot_cnt, (w - 3 - _tcw, _sum_y + _tot_bar_h // 2 + 2),
                    font, 0.25, (100, 100, 100), 1)

        # Per-zone robot counts + zone cell sizes
        _info_y = _sum_y + _tot_bar_h + 12
        _zone_ids  = [1, 2, 3]
        _zone_abbr = {1: "Imp", 2: "Sto", 3: "Exp"}
        _zone_cols = {1: (200, 130, 60), 2: (80, 180, 200), 3: (80, 120, 220)}
        # Count robots per zone
        _robots_in_zone = {1: 0, 2: 0, 3: 0}
        for rb in wh.robots:
            if rb.area == 'import':    _robots_in_zone[1] += 1
            elif rb.area == 'storage': _robots_in_zone[2] += 1
            elif rb.area == 'export':  _robots_in_zone[3] += 1
        # Zone cell counts
        _zone_cells = {1: wh.numberOfImportSlots, 2: wh.numberOfStorageSlots, 3: wh.numberOfExportSlots}
        _total_zone_cells = sum(_zone_cells.values())
        # Road cells per zone
        _road = wh.road_map
        _has_roads = _road is not None and _road.any()
        # Render: "Imp  R:3  420c 20%  | Sto  R:5  900c 45%  | ..."
        _col_span = w // 3
        for _zi, _zid in enumerate(_zone_ids):
            _zx = _zi * _col_span + 2
            _zabbr = _zone_abbr[_zid]
            _zcol  = _zone_cols[_zid]
            _rcount = _robots_in_zone[_zid]
            _cells  = _zone_cells[_zid]
            _cell_pct = int(_cells / max(_total_zone_cells, 1) * 100)
            # Line 1: zone abbr + robot count
            _l1 = "{} R:{}".format(_zabbr, _rcount)
            cv2.putText(arr, _l1, (_zx, _info_y), font, 0.25, _zcol, 1)
            # Line 2: cells + zone share %
            _l2 = "{}c {}%".format(_cells, _cell_pct)
            cv2.putText(arr, _l2, (_zx, _info_y + 10), font, 0.22, (90, 90, 90), 1)

        # Road coverage line
        if _has_roads:
            _road_cells = int(np.count_nonzero(_road))
            _grid_total = wh.zoneMap.shape[0] * wh.zoneMap.shape[1]
            _road_pct = int(_road_cells / max(_grid_total, 1) * 100)
            _road_line = "Roads: {} cells ({}%)".format(_road_cells, _road_pct)
            cv2.putText(arr, _road_line, (2, _info_y + 22), font, 0.22, (75, 75, 75), 1)
        canvas[y0:y1, x0:x1] = arr

    def _draw_main_view_overlays(self, composite, mw, mh):
        """Render road/plaza tints, robot body circles, FPS buttons, ESC hint.

        All drawing happens within the main view rect ``composite[0:mh, 0:mw]``
        plus the FPS button strip near its top edge.
        """
        # Subtle road/plaza overlay on the main view
        _mv = composite[0:mh, 0:mw]
        _rd = self.warehouse.road_map
        if self._show_roads and _rd is not None and _rd.any():
            _rd_rsz = cv2.resize(_rd, (mw, mh), interpolation=cv2.INTER_NEAREST)
            _rd_alpha = np.zeros((mh, mw), dtype=np.float32)
            _rd_alpha[_rd_rsz == 1] = 0.06    # branches: barely visible
            _rd_alpha[_rd_rsz == 2] = 0.10    # collectors: faint
            _rd_alpha[_rd_rsz >= 3] = 0.14    # arterials: subtle
            _a3 = _rd_alpha[:, :, np.newaxis]
            _tint = np.float32([200, 200, 200])  # near-white tint (BGR)
            _mv[:] = np.clip(
                _mv.astype(np.float32) * (1.0 - _a3) + _tint * _a3,
                0, 255).astype(np.uint8)
        _pz = self.warehouse.plaza_map
        if self._show_plazas and _pz is not None and _pz.any():
            _pz_rsz = cv2.resize(_pz, (mw, mh), interpolation=cv2.INTER_NEAREST)
            _pz_mask = _pz_rsz > 0
            _pz_tint = np.float32([140, 120, 60])  # warm amber (BGR)
            _mv[_pz_mask] = np.clip(
                _mv[_pz_mask].astype(np.float32) * 0.92 + _pz_tint * 0.08,
                0, 255).astype(np.uint8)

        # Robot overlay: filled circle on the main view.
        _scale_x = mw / max(self.warehouse_res[0], 1)
        _scale_y = mh / max(self.warehouse_res[1], 1)
        _cell_px = min(_scale_x, _scale_y)
        _r_body = max(int(_cell_px * 0.55), 1)
        for _rb in self.warehouse.robots:
            _cx = int(_rb.xyLocation[0] * _scale_x + _scale_x * 0.5)
            _cy = int(_rb.xyLocation[1] * _scale_y + _scale_y * 0.5)
            cv2.circle(composite, (_cx, _cy), _r_body + 1, (0, 0, 0), -1)
            cv2.circle(composite, (_cx, _cy), _r_body, (220, 220, 220), -1)

        # Display FPS controls (30/60) in main view header.
        _active_fps = int(round(1.0 / max(self.draw_frametime, 1e-9)))
        _fps_font = cv2.FONT_HERSHEY_SIMPLEX
        for fps, bx0, by0, bx1, by1 in self._display_fps_button_layout():
            _active = (_active_fps == fps)
            _bg = (24, 30, 22) if _active else (18, 18, 18)
            _edge = (90, 180, 110) if _active else (42, 42, 42)
            _txt_col = (150, 230, 170) if _active else (90, 90, 90)
            composite[by0:by1, bx0:bx1] = _bg
            composite[by0:by1, bx0:bx0 + 1] = _edge
            composite[by0:by1, bx1 - 1:bx1] = _edge
            composite[by0:by0 + 1, bx0:bx1] = _edge
            composite[by1 - 1:by1, bx0:bx1] = _edge
            _lbl = str(fps)
            (_tw, _th), _ = cv2.getTextSize(_lbl, _fps_font, 0.24, 1)
            _tx = bx0 + max((bx1 - bx0 - _tw) // 2, 1)
            _ty = by0 + (by1 - by0 + _th) // 2
            cv2.putText(composite, _lbl, (_tx, _ty), _fps_font, 0.24, _txt_col, 1)

        # ESC hint — dim label right of FPS buttons
        cv2.putText(composite, 'ESC: exit', (160, 13), _fps_font, 0.24, (70, 70, 70), 1)

    def _draw_deadlines_histogram(self, composite, mw, sw, dl_h):
        """Render the 5-band deadline-urgency histogram (top-right slot, top half).

        ``dl_h`` is the panel height in pixels.  Pure renderer: reads
        ``self.warehouse.packages`` and writes into ``composite``.
        """
        wh = self.warehouse
        bands = [
            ("OVR",   ( 60,  60, 200)),   # overdue        — red
            ("CRIT",  ( 50, 120, 230)),   # < 30 s         — orange
            ("URG",   ( 50, 200, 200)),   # < 5 min        — yellow
            ("NRML",  ( 80, 190,  80)),   # < 30 min       — green
            ("CMFT",  ( 80, 190, 140)),   # ≥ 30 min       — teal
        ]
        n_bands = len(bands)
        counts = [0, 0, 0, 0, 0]
        for _p in wh.packages:
            _td_s = _p.timeToDeadline.total_seconds()
            if   _td_s < 0:    counts[0] += 1
            elif _td_s < 30:   counts[1] += 1
            elif _td_s < 300:  counts[2] += 1
            elif _td_s < 1800: counts[3] += 1
            else:              counts[4] += 1
        total_pkg = max(sum(counts), 1)
        max_cnt   = max(counts) if wh.packages else 1
        img       = np.zeros((dl_h, sw, 3), dtype=np.uint8)
        font      = cv2.FONT_HERSHEY_SIMPLEX
        pad       = 4
        bar_w     = (sw - pad * (n_bands + 1)) // n_bands   # ~26 px
        label_h   = 22   # bottom area: band name + pct (2 rows)
        bar_y_top = 14   # leave room for the title overlay
        bar_y_bot = dl_h - label_h - 2
        chart_h   = bar_y_bot - bar_y_top
        for _bi, ((lbl, col), cnt) in enumerate(zip(bands, counts)):
            bx = pad + _bi * (bar_w + pad)
            cv2.rectangle(img, (bx, bar_y_top), (bx + bar_w, bar_y_bot), (20, 20, 20), -1)
            if cnt > 0:
                fill_h = max(2, int(cnt / max_cnt * chart_h))
                cv2.rectangle(img, (bx, bar_y_bot - fill_h), (bx + bar_w, bar_y_bot), col, -1)
            cnt_s = str(cnt) if cnt < 1000 else "{}K".format(cnt // 1000)
            (cw2, ch2), _ = cv2.getTextSize(cnt_s, font, 0.26, 1)
            cnt_col = col if cnt > 0 else (70, 70, 70)
            if cnt > 0:
                fill_top = bar_y_bot - max(2, int(cnt / max_cnt * chart_h))
                cnt_y = max(bar_y_top + ch2 + 1, fill_top - 2)
            else:
                cnt_y = bar_y_bot - 3
            cv2.putText(img, cnt_s, (bx + (bar_w - cw2) // 2, cnt_y),
                        font, 0.26, cnt_col, 1)
            (lw2, _), _ = cv2.getTextSize(lbl, font, 0.22, 1)
            lbl_col = col if cnt > 0 else (75, 75, 75)
            cv2.putText(img, lbl, (bx + (bar_w - lw2) // 2, dl_h - label_h + 10),
                        font, 0.22, lbl_col, 1)
            pct_s = "{:.0f}%".format(cnt / total_pkg * 100)
            (pw2, _), _ = cv2.getTextSize(pct_s, font, 0.20, 1)
            cv2.putText(img, pct_s, (bx + (bar_w - pw2) // 2, dl_h - 3),
                        font, 0.20, (90, 90, 90), 1)
        composite[0:dl_h, mw+sw*3:mw+sw*4] = img

    def _draw_fleet_battery(self, composite, mw, sw, sh, dl_h, fb_h):
        """Render the fleet battery bar chart (top-right slot, bottom half).

        Reads only ``self.warehouse``; writes into ``composite``.
        """
        wh = self.warehouse
        img       = np.zeros((fb_h, sw, 3), dtype=np.uint8)
        font      = cv2.FONT_HERSHEY_SIMPLEX
        chart_top = 22        # px reserved for title overlay + policy row
        chart_bot = fb_h
        chart_h   = chart_bot - chart_top
        bar_max_w = sw - 22   # right margin for optional % label on full bars
        dyn_thresh    = getattr(wh, '_charge_threshold', 30)
        thresh_target = getattr(wh, '_charge_threshold_target', dyn_thresh)
        thresh_x      = int(dyn_thresh / 100.0 * bar_max_w)
        pressure      = getattr(wh, '_charger_pressure', 0.0)
        pressure_raw  = getattr(wh, '_charger_pressure_raw', pressure)
        eco           = 1.0 - 0.3 * max(0.0, pressure - 0.5) * 2.0
        eco_save      = max(0.0, (1.0 - eco) * 100.0)
        robots_sorted = sorted(wh.robots, key=lambda r: r.batteryPercent)
        n_robots      = len(robots_sorted)
        if n_robots > 0:
            row_h_f = chart_h / n_robots
            for _fi, _fr in enumerate(robots_sorted):
                _y0 = chart_top + int(_fi * row_h_f)
                _y1 = chart_top + int((_fi + 1) * row_h_f)
                if _y1 <= _y0:
                    _y1 = _y0 + 1
                cv2.rectangle(img, (0, _y0), (sw - 1, _y1 - 1), (18, 18, 18), -1)
                if thresh_x > 0:
                    cv2.rectangle(img, (0, _y0), (thresh_x, _y1 - 1), (18, 10, 24), -1)
                _pct = _fr.batteryPercent
                _fill_w = max(1, int(_pct / 100.0 * bar_max_w))
                if _pct >= 50:
                    _col = (50, 200, 50)    # green
                elif _pct >= 20:
                    _col = (50, 200, 200)   # yellow
                else:
                    _col = (50, 50, 200)    # red
                cv2.rectangle(img, (0, _y0), (_fill_w, _y1 - 1), _col, -1)
                _ind_x  = sw - 18
                _mid_y  = (_y0 + _y1) // 2
                _bar_h  = max(_y1 - _y0, 1)
                _pip_half = max(_bar_h // 2 - 1, 1)
                if _fr.status == 'charging':
                    cv2.rectangle(img, (_ind_x, _mid_y - _pip_half),
                                  (_ind_x + 6, _mid_y + _pip_half), (200, 200, 50), -1)
                elif _fr.status == 'move to charging station':
                    cv2.rectangle(img, (_ind_x, _mid_y - _pip_half),
                                  (_ind_x + 6, _mid_y + _pip_half), (200, 200, 50), 1)
                elif _fr.batteryPercent < getattr(wh, '_limp_mode_batt_pct', 20):
                    cv2.rectangle(img, (_ind_x, _mid_y - _pip_half),
                                  (_ind_x + 6, _mid_y + _pip_half), (0, 140, 255), -1)
                elif _fr.carrying != -1:
                    cv2.rectangle(img, (_ind_x, _mid_y - _pip_half),
                                  (_ind_x + 6, _mid_y + _pip_half), (180, 180, 180), -1)
        else:
            cv2.putText(img, "no robots", (8, fb_h // 2), font, 0.32, (60, 60, 60), 1)

        # Charge threshold dashed vertical line
        for _dy in range(chart_top, fb_h, 3):
            cv2.line(img, (thresh_x, _dy), (thresh_x, min(_dy + 1, fb_h - 1)), (60, 60, 180), 1)
        cv2.putText(img, 'T', (max(thresh_x - 2, 1), chart_top - 3), font, 0.22, (90, 110, 220), 1)

        need = sum(1 for r in wh.robots if r.batteryPercent <= dyn_thresh)
        busy = sum(1 for c in wh.chargers if c.status != 'idle')
        policy_y = chart_top - 4
        _x = 2
        for _txt, _col in [
            ('PP', (120, 120, 120)),
            (' T{:.0f}/{:.0f}'.format(dyn_thresh, thresh_target), (90, 110, 220)),
            (' P{}/{}'.format(int(pressure * 100), int(pressure_raw * 100)), (60, 190, 190)),
            (' E-{}'.format(int(round(eco_save))), (120, 200, 120) if eco_save > 0 else (80, 80, 80)),
            (' N{}'.format(need), (60, 150, 230) if need > 0 else (80, 80, 80)),
            (' C{}'.format(busy), (200, 200, 50) if busy > 0 else (80, 80, 80)),
        ]:
            cv2.putText(img, _txt, (_x, policy_y), font, 0.22, _col, 1)
            (_tw, _th), _ = cv2.getTextSize(_txt, font, 0.22, 1)
            _x += _tw + 2

        # Mini legend
        lg_y = 4
        lg_items = [
            ((200, 200, 50), True,  'C'),
            ((200, 200, 50), False, 'E'),
            ((0, 140, 255),  True,  'S'),
            ((180, 180, 180), True, 'P'),
        ]
        lg_x = sw - 8 * len(lg_items) - 2
        for _lg_col, _lg_fill, _lg_ch in lg_items:
            cv2.rectangle(img, (lg_x, lg_y), (lg_x + 4, lg_y + 4),
                          _lg_col, -1 if _lg_fill else 1)
            cv2.putText(img, _lg_ch, (lg_x, lg_y + 10), font, 0.2, (120, 120, 120), 1)
            lg_x += 8

        composite[dl_h+1:sh, mw+sw*3:mw+sw*4] = img

    def _draw_flow_control(self, composite, mw, sw, sh):
        """Render the flow-control readout (row 1, col 3)."""
        wh = self.warehouse
        img      = np.zeros((sh, sw, 3), dtype=np.uint8)
        font     = cv2.FONT_HERSHEY_SIMPLEX
        cap      = int(getattr(wh, '_flow_import_cap', 0))
        n_robots = max(len(wh.robots), 1)
        ceiling  = n_robots * 10
        pipeline = getattr(wh, '_flow_pipeline_depth', 4)
        n_pkgs   = len(wh.packages)
        n_idle   = sum(1 for r in wh.robots if r.status == 'idle')
        n_overdue = sum(1 for p in wh.packages if p.timeToDeadline.total_seconds() < 0)
        throughput = getattr(wh, '_flow_throughput', 0) * 60
        delivery   = getattr(wh, '_flow_avg_delivery', 0)
        mode       = getattr(wh, '_flow_policy_mode', 'balanced')
        bar_y = 16
        bar_h = 10
        bar_w = sw - 8
        fill = int((cap / max(ceiling, 1)) * bar_w) if ceiling > 0 else 0
        cv2.rectangle(img, (4, bar_y), (4 + bar_w, bar_y + bar_h), (25, 25, 25), -1)
        if fill > 0:
            bar_col = (80, 190, 80) if cap >= n_robots * 3 else (50, 200, 200)
            cv2.rectangle(img, (4, bar_y), (4 + fill, bar_y + bar_h), bar_col, -1)
        cap_str = "cap {}/{}".format(cap, ceiling)
        (fcw, _), _ = cv2.getTextSize(cap_str, font, 0.26, 1)
        cv2.putText(img, cap_str, (4 + (bar_w - fcw) // 2, bar_y + bar_h - 2),
                    font, 0.26, (200, 200, 200), 1)
        pkg_bar_y = bar_y + bar_h + 3
        pkg_fill = int((min(n_pkgs, cap) / max(cap, 1)) * bar_w) if cap > 0 else 0
        cv2.rectangle(img, (4, pkg_bar_y), (4 + bar_w, pkg_bar_y + bar_h), (25, 25, 25), -1)
        if pkg_fill > 0:
            pkg_col = (80, 80, 200) if n_pkgs > cap else (80, 160, 80)
            cv2.rectangle(img, (4, pkg_bar_y), (4 + pkg_fill, pkg_bar_y + bar_h), pkg_col, -1)
        pkg_str = "pkgs {}/{}".format(n_pkgs, cap)
        (fpw, _), _ = cv2.getTextSize(pkg_str, font, 0.26, 1)
        cv2.putText(img, pkg_str, (4 + (bar_w - fpw) // 2, pkg_bar_y + bar_h - 2),
                    font, 0.26, (200, 200, 200), 1)
        data_y = pkg_bar_y + bar_h + 14
        idle_pct = n_idle / n_robots * 100
        ovrd_pct = n_overdue / max(n_pkgs, 1) * 100
        lines = [
            ("pipeline", "{} pkg/bot".format(pipeline), (140, 140, 140)),
            ("pkg/bot",  "{:.1f}".format(n_pkgs / n_robots), (180, 180, 180)),
            ("idle",     "{}/{} ({:.0f}%)".format(n_idle, n_robots, idle_pct),
             (50, 200, 200) if n_idle > 0 else (80, 80, 80)),
            ("overdue",  "{} ({:.0f}%)".format(n_overdue, ovrd_pct),
             (80, 80, 210) if n_overdue > 0 else (80, 80, 80)),
            ("thr/min",  "{:.1f}".format(throughput), (200, 180, 60)),
            ("deliv",    "{:.1f}s".format(delivery), (50, 160, 220) if delivery > 0 else (80, 80, 80)),
            ("mode",     mode, (120, 180, 120)),
        ]
        for _i, (lbl, val, vcol) in enumerate(lines):
            _y = data_y + _i * 11
            if _y >= sh - 4:
                break
            cv2.putText(img, lbl + ":", (4, _y), font, 0.24, (95, 95, 95), 1)
            cv2.putText(img, val,        (58, _y), font, 0.24, vcol, 1)
        composite[sh:2*sh, mw+sw*3:mw+sw*4] = img

    def _draw_subview_titles(self, composite, mw, sw, sh, util_overlay_active):
        """Draw the small grey panel-title labels overlaid on each sub-view."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        col  = (145, 145, 145)
        zonemap_title = "ZONE MAP util%" if util_overlay_active else "ZONE MAP"
        titles = [
            ("WAREHOUSE",       4,            4),
            ("CHARGERS",        mw + 4,       4),
            ("PACKAGES",        mw + sw + 4,  4),
            ("ROBOTS",          mw + 4,       sh + 4),
            ("PKG TARGETS",     mw + sw + 4,  sh + 4),
            ("TRAFFIC HEATMAP", mw + sw*2 + 4, sh + 4),
            (zonemap_title,     mw + sw*2 + 4, 4),
            ("DEADLINES",       mw + sw*3 + 4, 4),
            ("FLEET BATT",      mw + sw*3 + 4, 74),
            ("FLOW CTRL",       mw + sw*3 + 4, sh + 4),
        ]
        for txt, tx, ty in titles:
            cv2.putText(composite, txt, (tx, ty + 8), font, 0.28, col, 1)

    def _draw_heatmap_toggles(self, composite, _tfont=None):
        """Draw the HM/RD/PLZ overlay toggle buttons on the traffic-heatmap view."""
        font = _tfont if _tfont is not None else cv2.FONT_HERSHEY_SIMPLEX
        style_on  = {'bg': (24, 32, 28), 'edge': (80, 170, 120), 'text': (140, 220, 170)}
        style_off = {'bg': (18, 18, 18), 'edge': (42, 42, 42),   'text': (75, 75, 75)}
        state = {
            'heatmap': self._show_heatmap,
            'roads':   self._show_roads,
            'plazas':  self._show_plazas,
        }
        for key, label, bx0, by0, bx1, by1 in self._heatmap_toggle_button_layout():
            st = style_on if state.get(key, False) else style_off
            composite[by0:by1, bx0:bx1] = st['bg']
            composite[by0:by1, bx0:bx0 + 1] = st['edge']
            composite[by0:by1, bx1 - 1:bx1] = st['edge']
            composite[by0:by0 + 1, bx0:bx1] = st['edge']
            composite[by1 - 1:by1, bx0:bx1] = st['edge']
            (tw, th), _ = cv2.getTextSize(label, font, 0.22, 1)
            tx = bx0 + max((bx1 - bx0 - tw) // 2, 1)
            ty = by0 + (by1 - by0 + th) // 2
            cv2.putText(composite, label, (tx, ty), font, 0.22, st['text'], 1)

    def _draw_zone_legend(self, composite, mw, mh):
        """Draw the colour-keyed IMP/STO/EXP legend on the main warehouse view."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        zone_info = [
            (1, "IMP", self.warehouse.colourOfImportAreas),
            (2, "STO", self.warehouse.colourOfStorageAreas),
            (3, "EXP", self.warehouse.colourOfExportAreas),
        ]
        leg_y  = mh - 10 - len(zone_info) * 12
        bg_y0  = max(0, leg_y - 2)
        bg_y1  = min(mh, leg_y + len(zone_info) * 12 + 4)
        bg_x0, bg_x1 = 1, 46
        roi = composite[bg_y0:bg_y1, bg_x0:bg_x1]
        composite[bg_y0:bg_y1, bg_x0:bg_x1] = (roi * 0.35).astype(np.uint8)
        for i, (_zid, lbl, zcol) in enumerate(zone_info):
            ly = leg_y + i * 12 + 10
            bright = tuple(min(int(c * 2.5), 255) for c in zcol)
            cv2.rectangle(composite, (4, ly - 7), (12, ly - 1), bright, -1)
            cv2.putText(composite, lbl, (15, ly), font, 0.32, (200, 200, 200), 1)

    def _draw_panel_outlines(self, composite, mw, mh, sw, sh, ph, ch, cw):
        """Draw the subtle dark-grey outlines around every panel + chart cell."""
        oc  = (45, 45, 45)
        cw4 = cw // 4

        def outline(x0, y0, x1, y1):
            cv2.rectangle(composite, (x0, y0), (x1 - 1, y1 - 1), oc, 1)

        # Top row: main view + 4 sub-views + zone map + stats
        outline(0,           0,   mw,           mh)
        outline(mw,          0,   mw + sw,      sh)
        outline(mw + sw,     0,   mw + sw * 2,  sh)
        outline(mw,          sh,  mw + sw,      sh * 2)
        outline(mw + sw,     sh,  mw + sw * 2,  sh * 2)
        outline(mw + sw * 2, 0,   mw + sw * 3,  sh)
        outline(mw + sw * 2, sh,  mw + sw * 3,  sh * 2)
        outline(mw + sw * 3, 0,   mw + sw * 4,  sh)
        outline(mw + sw * 3, sh,  mw + sw * 4,  sh * 2)
        # Info panel (one outline for the whole strip)
        outline(0, mh, cw, mh + ph)
        # Chart strip (four charts)
        outline(0,         mh + ph, cw4,     mh + ph + ch)
        outline(cw4,       mh + ph, cw4 * 2, mh + ph + ch)
        outline(cw4 * 2,   mh + ph, cw4 * 3, mh + ph + ch)
        outline(cw4 * 3,   mh + ph, cw,      mh + ph + ch)

    def _finalize_and_display(self, composite):
        """Push the composite to the OpenCV window and dispatch the keypress."""
        cv2.imshow("warehouse", composite)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.gameEnd()
        self.paint_handler.handle_key(key)

    def _draw_info_panel(self, composite, mw, mh, sw, sh, ph, cw, stats_lines, now_t, fleet):
        """Render the four-column info panel (ROBOTS / CHARGERS / SIM STATS / PACKAGES)."""
        draw_info_panel(
            composite, mw, mh, sw, sh, ph, cw, stats_lines, now_t, fleet,
            self.warehouse,
            list(self._power_policy_button_layout()),
            list(self._flow_policy_button_layout()),
            list(self._pkg_target_button_layout()),
            list(self._robot_count_button_layout()),
        )

    def gameEnd(self):
        self.exit = True

def _shutdown(reason=''):
    """Best-effort teardown of OpenCV windows and logging handlers.

    Called from the main entrypoint's ``finally`` so the process exits
    cleanly even when the game loop raises an unhandled exception.
    """
    try:
        cv2.destroyAllWindows()
        # Drain any pending GUI events so the OS reclaims the window handle.
        for _ in range(4):
            cv2.waitKey(1)
    except Exception:
        pass
    try:
        for _h in list(logging.getLogger().handlers):
            try:
                _h.flush()
                _h.close()
            except Exception:
                pass
            logging.getLogger().removeHandler(_h)
    except Exception:
        pass
    if reason:
        log.info('-- Shutdown: %s --', reason)


if __name__ == '__main__':

    configure_logging()
    mainGame = None
    loop_count = 0
    log.info('-- Game Loop Start --')
    try:
        mainGame = MainGame()
        while not mainGame.exit:
            mainGame.gameLoop(loop_count)
            loop_count += 1
        log.info('-- Game Loop End --')
        time.sleep(2)
    except KeyboardInterrupt:
        log.info('-- Game Loop Interrupted --')
    except Exception:
        log.exception('Unhandled exception in game loop')
        raise
    finally:
        _shutdown()