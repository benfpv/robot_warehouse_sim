import os
import numpy as np
import math
import random
import time
import cv2

from data.item import *
from data.importer import *
from data.functions import *

from data.warehouse.warehouse_init import *
from data.warehouse.warehouse import *
from data.warehouse.package import *
from data.warehouse.robot import *
from data.draw.draw_warehouse import *
from data.display.display_functions import *
from data.paint.paint_handler import PaintHandler
from data.paint.map_importer import MapImporter
from data.paint.warehouse_optimizer import WarehouseOptimizer
from data.paint.zone_strategy import ZoneStrategy

class MainGame:
    _POWER_POLICY_MODES = ['eco', 'balanced', 'performance']
    _POWER_POLICY_STYLE = {
        'eco': {'label': 'ECO', 'bg': (20, 30, 24), 'edge': (80, 180, 120), 'text': (140, 220, 170)},
        'balanced': {'label': 'BAL', 'bg': (22, 24, 30), 'edge': (120, 140, 220), 'text': (180, 190, 250)},
        'performance': {'label': 'PERF', 'bg': (30, 24, 22), 'edge': (220, 140, 100), 'text': (255, 210, 170)},
    }
    _POWER_POLICY_REASON = {
        'eco': 'higher buffers, safer charging behavior',
        'balanced': 'balanced safety and throughput',
        'performance': 'lower buffers, higher utilization',
    }
    _POWER_POLICY_HINT = {
        'eco': 'safe charge',
        'balanced': 'balanced',
        'performance': 'high util',
    }
    _FLOW_POLICY_MODES = ['steady', 'balanced', 'throughput']
    _FLOW_POLICY_STYLE = {
        'steady': {'label': 'STDY', 'bg': (20, 30, 24), 'edge': (80, 180, 120), 'text': (140, 220, 170)},
        'balanced': {'label': 'BAL', 'bg': (22, 24, 30), 'edge': (120, 140, 220), 'text': (180, 190, 250)},
        'throughput': {'label': 'THRU', 'bg': (30, 24, 22), 'edge': (220, 140, 100), 'text': (255, 210, 170)},
    }
    _FLOW_POLICY_REASON = {
        'steady': 'lower occupancy target for stability',
        'balanced': 'balanced occupancy target',
        'throughput': 'higher occupancy target for throughput',
    }
    _FLOW_POLICY_HINT = {
        'steady': 'stable queue',
        'balanced': 'balanced',
        'throughput': 'max throughput',
    }
    _PKG_TARGET_MODES = ['random', 'nearest', 'zone_edge']
    _PKG_TARGET_STYLE = {
        'random': {'label': 'RND', 'hint': 'spread', 'bg': (18, 22, 24), 'edge': (48, 58, 66), 'text': (110, 120, 128)},
        'nearest': {'label': 'NEAR', 'hint': 'min dist', 'bg': (20, 30, 40), 'edge': (40, 160, 220), 'text': (100, 210, 255)},
        'zone_edge': {'label': 'EDGE', 'hint': 'pipeline', 'bg': (22, 30, 25), 'edge': (80, 180, 180), 'text': (170, 230, 230)},
    }
    _PKG_TARGET_REASON = {
        'random': 'spread load and avoid clustering',
        'nearest': 'minimize immediate travel distance',
        'zone_edge': 'stage packages for the next pipeline handoff',
    }

    def __init__(self):
        print("--- MainGame Init ---")
        self.timeStart = int(time.time())
        self.exit = False
        # User Parameters
        self.warehouse_res = (80,70)
        self.warehouse_windowBackgroundColour = [30,30,30]
        self.warehouse_windowRes = (320, 280) # upsized resolution (width, height)
        self.sim_frametime  = 1 / 40   # 40 sim ticks/sec
        self.draw_frametime = 1 / 15   # 15 display fps (reduced to save power)
        self._last_draw_time = 0.0
        self.frametime_cap = self.sim_frametime  # kept for hist_sample_every
        # Init Windows
        self.initWarehouseWindow()
        # Init Imports
        self.itemsList = Importer.init_import_csv_as_list('resources/list_items.csv')
        self.itemsList = Importer.init_objectify_items_list(self.itemsList)
        self.addressesList = Importer.init_import_csv_as_list('resources/list_addresses.csv')
        self.addressesList = Importer.init_objectify_addresses_list(self.addressesList)
        # Init Warehouse
        self.warehouse = Warehouse(self.warehouse_res, self.warehouse_windowBackgroundColour, self.warehouse_windowCenter, self.warehouse_windowArray, self.itemsList, self.addressesList)

        # Optimizer (strategy pattern — zone rebalancing first, future: charger/robot)
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

        # Map importer: generate example on first run; auto-load zone_map.png if present
        _example_path = "resources/zone_map_example.png"
        if not os.path.exists(_example_path):
            MapImporter.generate_example_png(_example_path, *self.warehouse_res)
        _map_path = "resources/zone_map.png"
        if os.path.exists(_map_path):
            gw, gh = self.warehouse_res
            _loaded = MapImporter.load(_map_path, gw, gh)
            if _loaded is not None:
                self.warehouse.zoneMap = _loaded
                self.warehouse.update_zone_counts()
                print("[main] Zone map applied from '{}'.".format(_map_path))

        # Charger spawn map: generate example on first run; auto-load charger_map.png if present
        _charger_example_path = "resources/charger_map_example.png"
        if not os.path.exists(_charger_example_path):
            MapImporter.generate_charger_map_example(_charger_example_path, *self.warehouse_res)
        _charger_map_path = "resources/charger_map.png"
        if os.path.exists(_charger_map_path):
            gw, gh = self.warehouse_res
            _charger_coords = MapImporter.load_spawn_map(_charger_map_path, gw, gh)
            if _charger_coords:                          # non-empty list = valid map
                self.warehouse.chargerSpawnMap = _charger_coords
                print("[main] Charger spawn map applied from '{}' ({} cells).".format(
                    _charger_map_path, len(_charger_coords)))

        # Robot spawn map: generate example on first run; auto-load robot_map.png if present
        _robot_example_path = "resources/robot_map_example.png"
        if not os.path.exists(_robot_example_path):
            MapImporter.generate_robot_map_example(_robot_example_path, *self.warehouse_res)
        _robot_map_path = "resources/robot_map.png"
        if os.path.exists(_robot_map_path):
            gw, gh = self.warehouse_res
            _robot_coords = MapImporter.load_spawn_map(_robot_map_path, gw, gh)
            if _robot_coords:
                self.warehouse.robotSpawnMap = _robot_coords
                print("[main] Robot spawn map applied from '{}' ({} cells).".format(
                    _robot_map_path, len(_robot_coords)))

        # Temporary draw
        self.warehouseWindow = self.warehouse_windowArray.copy()

    def initWarehouseWindow(self):
        self.warehouse_windowCenter = Functions.get_screencenter(self.warehouse_res)
        self.warehouse_windowArray = Functions.get_screenarray_colour(self.warehouse_res, self.warehouse_windowBackgroundColour)
        self.sub_windowRes = (int(self.warehouse_windowRes[0] * 0.5), int(self.warehouse_windowRes[1] * 0.5))  # 160x140
        self.panel_h = 165
        self.chart_h = 95
        self.composite_windowRes = (self.warehouse_windowRes[0] + self.sub_windowRes[0] * 4, self.warehouse_windowRes[1] + self.panel_h + self.chart_h)  # 960x540
        # Rolling history arrays for charts (1800 samples @ 1/sec ≈ 30 min)
        _H = 1800
        self._hist_imported = np.zeros(_H, dtype=np.float32)
        self._hist_exported = np.zeros(_H, dtype=np.float32)
        self._hist_overdue  = np.zeros(_H, dtype=np.float32)
        self._hist_late     = np.zeros(_H, dtype=np.float32)  # cumulative ever-overdue count
        self._late_total    = 0
        self._hist_batt       = np.zeros(_H, dtype=np.float32)
        self._hist_robots     = np.zeros(_H, dtype=np.float32)
        self._hist_need_charge = np.zeros(_H, dtype=np.float32)
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
        # Robot traffic heatmap — accumulated robotsInWarehouse, decayed each tick
        gw, gh = self.warehouse_res
        self._robot_heatmap      = np.zeros((gh, gw), dtype=np.float32)  # fast (~8 s half-life)
        self._robot_heatmap_slow = np.zeros((gh, gw), dtype=np.float32)  # slow (~5 min half-life)
        # Optimizer action log: ring buffer of (timestamp, action_str), newest last
        self._opt_log: list = []
        self._last_seen_opt_str = ''
        screen_w, screen_h = Display_Functions.get_screen_resolution()
        cx = (screen_w - self.composite_windowRes[0]) // 2
        cy = (screen_h - self.composite_windowRes[1]) // 2
        Display_Functions.init_borderless_window("warehouse", self.composite_windowRes, [cx, cy])
        print('- warehouse_res: {}, arrayShape: {}, composite_windowRes: {}, screen_center_pos: [{},{}]'.format(
            self.warehouse_res, np.shape(self.warehouse_windowArray), self.composite_windowRes, cx, cy))
        return self

    # ── Mouse callback ──────────────────────────────────────────────────

    def _on_mouse(self, event, px, py, flags, param):
        """Composite cv2 mouse callback. Handles dashboard controls, then delegates."""
        if self._handle_power_policy_click(event, px, py):
            return
        if self._handle_flow_policy_click(event, px, py):
            return
        if self._handle_pkg_target_mode_click(event, px, py):
            return
        self.paint_handler.on_mouse(event, px, py, flags, param)

    def _set_power_policy_mode(self, mode):
        if not self.warehouse.set_power_policy_mode(mode):
            return
        print('[main] Power Policy -> {} ({})'.format(
            mode, self._POWER_POLICY_REASON[mode]))

    def _set_flow_policy_mode(self, mode):
        if not self.warehouse.set_flow_policy_mode(mode):
            return
        print('[main] Flow Policy -> {} ({})'.format(
            mode, self._FLOW_POLICY_REASON[mode]))

    def _power_policy_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        pad_x = 4
        gap = 3
        btn_h = 11
        btn_w = 34
        total_w = btn_w * 3 + gap * 2
        x0 = col_w * 2 - pad_x - total_w
        y0 = mh + 3
        layout = []
        for i, mode in enumerate(self._POWER_POLICY_MODES):
            bx0 = x0 + i * (btn_w + gap)
            bx1 = bx0 + btn_w
            layout.append((mode, bx0, y0, bx1, y0 + btn_h))
        return layout

    def _flow_policy_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        pad_x = 4
        gap = 3
        btn_h = 11
        btn_w = 34
        total_w = btn_w * 3 + gap * 2
        x0 = col_w * 3 - pad_x - total_w
        y0 = mh + 3
        layout = []
        for i, mode in enumerate(self._FLOW_POLICY_MODES):
            bx0 = x0 + i * (btn_w + gap)
            bx1 = bx0 + btn_w
            layout.append((mode, bx0, y0, bx1, y0 + btn_h))
        return layout

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
        print('[main] Package target mode -> {} ({})'.format(
            mode, self._PKG_TARGET_REASON.get(mode, 'user override')))

    def _pkg_target_button_layout(self):
        mh = self.warehouse_windowRes[1]
        cw = self.composite_windowRes[0]
        col_w = cw // 4
        pad_x = 4
        gap = 3
        btn_h = 11
        btn_w = 38
        total_w = btn_w * 3 + gap * 2
        x0 = cw - pad_x - total_w
        y0 = mh + 3
        layout = []
        for i, mode in enumerate(self._PKG_TARGET_MODES):
            bx0 = x0 + i * (btn_w + gap)
            bx1 = bx0 + btn_w
            layout.append((mode, bx0, y0, bx1, y0 + btn_h))
        return layout

    def _handle_pkg_target_mode_click(self, event, px, py):
        if event != cv2.EVENT_LBUTTONDOWN:
            return False
        for mode, bx0, by0, bx1, by1 in self._pkg_target_button_layout():
            if bx0 <= px < bx1 and by0 <= py < by1:
                self._set_pkg_target_mode(mode)
                return True
        return False

    # ── Formatting helpers ──────────────────────────────────────────────

    @staticmethod
    def _fmt_elapsed(s):
        """Format seconds into a concise human-readable string.
        Picks the two most significant units."""
        if s < 0:
            return "0s"
        intervals = [
            (365.25 * 24 * 3600 * 10, "decade", "decades"),
            (365.25 * 24 * 3600,       "yr",     "yrs"),
            (30.44  * 24 * 3600,       "mo",     "mo"),
            (24 * 3600,                "d",      "d"),
            (3600,                     "hr",     "hrs"),
            (60,                       "min",    "mins"),
            (1,                        "s",      "s"),
        ]
        parts = []
        rem = float(s)
        for secs, sg, pl in intervals:
            if rem >= secs:
                n = int(rem // secs)
                rem -= n * secs
                parts.append("{}{}".format(n, pl if n != 1 else sg))
                if len(parts) == 2:
                    break
        return " ".join(parts) if parts else "0s"

    @staticmethod
    def _fmt_age(seconds):
        """Short age string: e.g. '3s', '2m12s', '1h5m'."""
        s = int(seconds)
        if s < 60:
            return "{}s".format(s)
        if s < 3600:
            return "{}m{}s".format(s // 60, s % 60)
        if s < 86400:
            return "{}h{}m".format(s // 3600, (s % 3600) // 60)
        return "{}d{}h".format(s // 86400, (s % 86400) // 3600)

    @staticmethod
    def _fmt_rate(val):
        """Format a per-second rate concisely, abbreviating large values."""
        if val == 0:
            return "0/s"
        a = abs(val)
        if a >= 1_000_000:
            return "{:+.1f}M/s".format(val / 1_000_000)
        if a >= 10_000:
            return "{:+.1f}K/s".format(val / 1_000)
        if a >= 10:
            return "{:+.0f}/s".format(val)
        return "{:+.1f}/s".format(val)

    @staticmethod
    def _n(val):
        """Abbreviate large integers for display."""
        if val >= 1_000_000:
            return "{:.1f}M".format(val / 1_000_000)
        if val >= 10_000:
            return "{:.1f}K".format(val / 1_000)
        return str(val)

    def gameLoop(self, loop_count):
        timeLoopStart = time.time()

        self.timeElapsed = int(time.time() - self.timeStart)
        #print('--- New loop --- #{}, timeElapsed: {}'.format(loop_count, self.timeElapsed))

        if self.timeElapsed > 24000:
            self.gameEnd()
            return self

        # Update Existing
        self.optimizer.step(self.warehouse, self._robot_heatmap_slow)
        self.warehouse = self.warehouse.update_warehouse()
        # Accumulate robot traffic heatmap — only moving robots, not idle/charging
        # Decay is time-anchored: per-tick factor = 0.5^(dt / half_life), so the
        # visible window stays constant regardless of sim speed or FPS setting.
        _STATIC = {'idle', 'charging'}
        self._robot_heatmap      *= 0.5 ** (self.sim_frametime / 8.0)
        self._robot_heatmap_slow *= 0.5 ** (self.sim_frametime / 300.0)  # 5-min half-life
        for _rb in self.warehouse.robots:
            if _rb.status not in _STATIC:
                _rx, _ry = _rb.xyLocation
                self._robot_heatmap[_ry, _rx]      += 1.0
                self._robot_heatmap_slow[_ry, _rx] += 1.0

        # Draw (rate-limited independently of sim)
        now = time.time()
        if now - self._last_draw_time >= self.draw_frametime:
            self._last_draw_time = now
            self.gameDraw()

        frameTime = round(time.time() - timeLoopStart, 3)

        if (frameTime < self.sim_frametime):
            time.sleep(self.sim_frametime - frameTime)
        
        if loop_count % 1 == 0:
            pass
            #print('- L#{}, t: {}, pIn: {}, pToMove: {}, rIn: {}, pMoving: {}, avg_frameTime: {}'.format(loop_count, self.timeElapsed, self.warehouse.packagesInWarehouseCount, len(self.warehouse.packagesMoveList), self.warehouse.robotsInWarehouseCount, len(self.warehouse.packagesMovingList), frameTime))

        return self, frameTime

    def gameDraw(self):
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

        # Binary sub-views: multiply by 255, resize, convert to BGR
        def place_binary(arr, row, col):
            img = cv2.resize((arr * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            composite[row*sh:(row+1)*sh, mw+col*sw:mw+(col+1)*sw] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        place_binary(self.warehouse.chargersInWarehouse,        0, 0)
        place_binary(self.warehouse.packagesInWarehouse,        0, 1)
        place_binary(self.warehouse.robotsInWarehouse,          1, 0)
        # Traffic heatmap — row 1, col 2 (dedicated panel, full sw×sh)
        # Normalization: percentile-clip at 98th percentile so isolated hotspots
        # don't crush the rest of the map to near-zero; then sqrt (gamma 0.5) to
        # stretch mid-range traffic into the visible part of the colour ramp.
        # COLORMAP_JET: dark-blue (no/low traffic) → cyan → green → yellow → red
        # (high traffic) — the "cold to hot" mapping is universally intuitive.
        # Zero-traffic cells are forced to black so they read as background, not
        # as the dark-blue end of the colour ramp.
        _hm_max = self._robot_heatmap.max()
        if _hm_max > 0:
            _hm_visited = self._robot_heatmap > 0
            _hm_pct = float(np.percentile(self._robot_heatmap[_hm_visited], 98)) if _hm_visited.any() else _hm_max
            _hm_clip = max(_hm_pct, _hm_max * 0.05)   # never clip below 5 % of true peak
            # Power 1.5 > 1.0: pushes dim/stale cells toward black, keeps active
            # hotspots bright — more spatial contrast than sqrt (which boosted mid-range).
            _hm_norm = np.clip((self._robot_heatmap / _hm_clip) ** 1.5 * 255, 0, 255).astype(np.uint8)
        else:
            _hm_norm = np.zeros_like(self._robot_heatmap, dtype=np.uint8)
        _hm_resized = cv2.resize(_hm_norm, (sw, sh), interpolation=cv2.INTER_NEAREST)
        _hm_col = cv2.applyColorMap(_hm_resized, cv2.COLORMAP_JET)
        # Mask unvisited cells back to black (they'd otherwise render as dark blue)
        _hm_zero_mask = cv2.resize(
            (_hm_norm == 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        _hm_col[_hm_zero_mask > 0] = (0, 0, 0)
        composite[sh:2*sh, mw+sw*2:mw+sw*3] = _hm_col

        # Slow-heatmap overlay — persistent route layer (~5 min half-life).
        # Rendered as a warm-white brightness boost so historically-busy corridors
        # glow even when no robot is currently there, without hiding the JET colours.
        _slow_max = self._robot_heatmap_slow.max()
        if _slow_max > 0:
            _slow_visited = self._robot_heatmap_slow > 0
            _slow_pct  = float(np.percentile(self._robot_heatmap_slow[_slow_visited], 98)) if _slow_visited.any() else _slow_max
            _slow_clip = max(_slow_pct, _slow_max * 0.05)
            _slow_norm = np.clip((self._robot_heatmap_slow / _slow_clip) ** 1.5 * 255, 0, 255).astype(np.uint8)
            _slow_rsz  = cv2.resize(_slow_norm, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _slow_ov   = cv2.cvtColor(_slow_rsz, cv2.COLOR_GRAY2BGR)
            _slow_zero = cv2.resize(
                (self._robot_heatmap_slow == 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            _slow_ov[_slow_zero > 0] = (0, 0, 0)
            composite[sh:2*sh, mw+sw*2:mw+sw*3] = cv2.addWeighted(
                composite[sh:2*sh, mw+sw*2:mw+sw*3], 1.0, _slow_ov, 0.30, 0)

        # Package targets sub-view: arrows on blank canvas, target dots stamped on top
        sub_scale = sw // self.warehouse_res[0]  # 2
        pkg_tgt_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        Draw_Warehouse.draw_package_arrows(pkg_tgt_img, self.warehouse.packages, sub_scale)
        tgt_resized = cv2.resize((self.warehouse.packageTargetsInWarehouse * 255).astype(np.uint8),
                                  (sw, sh), interpolation=cv2.INTER_NEAREST)
        pkg_tgt_img[tgt_resized > 0] = (255, 255, 255)
        composite[sh:2*sh, mw+sw:mw+2*sw] = pkg_tgt_img

        # ── Slot (row 1, col 3): DEADLINES (top) + FLEET BATT (bottom) ──────────────────
        # Panel is 160×140 px: top 70 px → deadline urgency histogram,
        # 1 px separator, bottom 69 px → fleet battery bar chart.
        _DL_H = 70
        _FB_H = sh - _DL_H - 1   # 69 px

        # ── DEADLINES histogram (top _DL_H rows) ─────────────────────────────
        # 5-band urgency histogram across all live packages.
        # Bars are scaled relative to the tallest band so the chart is
        # always readable regardless of total package count.
        _dl_bands = [
            ("OVR",   ( 60,  60, 200)),   # overdue         — red
            ("CRIT",  ( 50, 120, 230)),   # < 30 s          — orange
            ("URG",   ( 50, 200, 200)),   # < 5 min         — yellow
            ("NRML",  ( 80, 190,  80)),   # < 30 min        — green
            ("CMFT",  ( 80, 190, 140)),   # ≥ 30 min        — teal
        ]
        _dl_n      = len(_dl_bands)
        _dl_counts = [0, 0, 0, 0, 0]
        for _p in wh.packages:
            _td_s = _p.timeToDeadline.total_seconds()
            if   _td_s < 0:    _dl_counts[0] += 1
            elif _td_s < 30:   _dl_counts[1] += 1
            elif _td_s < 300:  _dl_counts[2] += 1
            elif _td_s < 1800: _dl_counts[3] += 1
            else:              _dl_counts[4] += 1
        _dl_total_pkg  = max(sum(_dl_counts), 1)
        _dl_max_cnt    = max(_dl_counts) if wh.packages else 1
        _dl_img        = np.zeros((_DL_H, sw, 3), dtype=np.uint8)
        _dl_font       = cv2.FONT_HERSHEY_SIMPLEX
        _dl_pad        = 4
        _dl_bar_w      = (sw - _dl_pad * (_dl_n + 1)) // _dl_n   # ~26 px
        _dl_label_h    = 22   # bottom area: band name + pct (2 rows)
        _dl_bar_y_top  = 14   # leave room for the title overlay
        _dl_bar_y_bot  = _DL_H - _dl_label_h - 2
        _dl_chart_h    = _dl_bar_y_bot - _dl_bar_y_top
        for _bi, ((_dl_lbl, _dl_col), _dl_cnt) in enumerate(zip(_dl_bands, _dl_counts)):
            _dl_bx = _dl_pad + _bi * (_dl_bar_w + _dl_pad)
            # track (empty bar background)
            cv2.rectangle(_dl_img,
                          (_dl_bx, _dl_bar_y_top),
                          (_dl_bx + _dl_bar_w, _dl_bar_y_bot),
                          (20, 20, 20), -1)
            # filled portion
            if _dl_cnt > 0:
                _dl_fill_h = max(2, int(_dl_cnt / _dl_max_cnt * _dl_chart_h))
                cv2.rectangle(_dl_img,
                              (_dl_bx, _dl_bar_y_bot - _dl_fill_h),
                              (_dl_bx + _dl_bar_w, _dl_bar_y_bot),
                              _dl_col, -1)
            # count above filled bar (floats above the top of the fill, or above the track)
            _dl_cnt_s = str(_dl_cnt) if _dl_cnt < 1000 else "{}K".format(_dl_cnt // 1000)
            (_cw2, _ch2), _ = cv2.getTextSize(_dl_cnt_s, _dl_font, 0.26, 1)
            _dl_cnt_col = _dl_col if _dl_cnt > 0 else (70, 70, 70)
            if _dl_cnt > 0:
                _dl_fill_top = _dl_bar_y_bot - max(2, int(_dl_cnt / _dl_max_cnt * _dl_chart_h))
                _dl_cnt_y = max(_dl_bar_y_top + _ch2 + 1, _dl_fill_top - 2)
            else:
                _dl_cnt_y = _dl_bar_y_bot - 3
            cv2.putText(_dl_img, _dl_cnt_s,
                        (_dl_bx + (_dl_bar_w - _cw2) // 2, _dl_cnt_y),
                        _dl_font, 0.26, _dl_cnt_col, 1)
            # band name
            (_lw2, _), _ = cv2.getTextSize(_dl_lbl, _dl_font, 0.22, 1)
            _dl_lbl_col = _dl_col if _dl_cnt > 0 else (75, 75, 75)
            cv2.putText(_dl_img, _dl_lbl,
                        (_dl_bx + (_dl_bar_w - _lw2) // 2, _DL_H - _dl_label_h + 10),
                        _dl_font, 0.22, _dl_lbl_col, 1)
            # pct of total
            _dl_pct_s = "{:.0f}%".format(_dl_cnt / _dl_total_pkg * 100)
            (_pw2, _), _ = cv2.getTextSize(_dl_pct_s, _dl_font, 0.20, 1)
            cv2.putText(_dl_img, _dl_pct_s,
                        (_dl_bx + (_dl_bar_w - _pw2) // 2, _DL_H - 3),
                        _dl_font, 0.20, (90, 90, 90), 1)

        composite[0:_DL_H, mw+sw*3:mw+sw*4] = _dl_img

        # 1 px separator between DEADLINES and FLEET BATT
        composite[_DL_H, mw+sw*3:mw+sw*4] = (45, 45, 45)

        # ── FLEET BATT bar chart (bottom _FB_H rows) ─────────────────────────────
        # One horizontal bar per robot, sorted ascending by battery %.  Green
        # (high) → yellow (mid) → red (low).  Gives an immediate fleet-wide
        # battery picture that the single avg:XX% stat cannot.
        _fb_img   = np.zeros((_FB_H, sw, 3), dtype=np.uint8)
        _fb_font  = cv2.FONT_HERSHEY_SIMPLEX
        _fb_chart_top = 22        # px reserved for title overlay + policy row
        _fb_chart_bot = _FB_H
        _fb_chart_h   = _fb_chart_bot - _fb_chart_top
        _fb_bar_max_w = sw - 22   # right margin for optional % label on full bars
        _fb_dyn_thresh = getattr(self.warehouse, '_charge_threshold', 30)
        _fb_thresh_target = getattr(self.warehouse, '_charge_threshold_target', _fb_dyn_thresh)
        _fb_thresh_x = int(_fb_dyn_thresh / 100.0 * _fb_bar_max_w)
        _fb_pressure = getattr(self.warehouse, '_charger_pressure', 0.0)
        _fb_pressure_raw = getattr(self.warehouse, '_charger_pressure_raw', _fb_pressure)
        _fb_eco = 1.0 - 0.3 * max(0.0, _fb_pressure - 0.5) * 2.0
        _fb_eco_save = max(0.0, (1.0 - _fb_eco) * 100.0)
        robots_sorted_batt = sorted(wh.robots, key=lambda r: r.batteryPercent)
        _fb_n = len(robots_sorted_batt)
        if _fb_n > 0:
            _fb_row_h_f = _fb_chart_h / _fb_n
            for _fi, _fr in enumerate(robots_sorted_batt):
                _fb_y0 = _fb_chart_top + int(_fi * _fb_row_h_f)
                _fb_y1 = _fb_chart_top + int((_fi + 1) * _fb_row_h_f)
                if _fb_y1 <= _fb_y0:
                    _fb_y1 = _fb_y0 + 1
                # Track background
                cv2.rectangle(_fb_img, (0, _fb_y0), (sw - 1, _fb_y1 - 1), (18, 18, 18), -1)
                if _fb_thresh_x > 0:
                    cv2.rectangle(_fb_img, (0, _fb_y0), (_fb_thresh_x, _fb_y1 - 1), (18, 10, 24), -1)
                # Bar fill — colour by battery level
                _fb_pct = _fr.batteryPercent
                _fb_fill_w = max(1, int(_fb_pct / 100.0 * _fb_bar_max_w))
                if _fb_pct >= 50:
                    _fb_col = (50, 200, 50)    # green
                elif _fb_pct >= 20:
                    _fb_col = (50, 200, 200)   # yellow
                else:
                    _fb_col = (50, 50, 200)    # red
                cv2.rectangle(_fb_img, (0, _fb_y0), (_fb_fill_w, _fb_y1 - 1), _fb_col, -1)
                # ── Right-edge status indicators ──
                # Small coloured marker on the right side of each bar to show
                # the robot's power/activity state at a glance.
                _fb_ind_x = sw - 18          # right margin area
                _fb_mid_y = (_fb_y0 + _fb_y1) // 2
                _fb_bar_h = max(_fb_y1 - _fb_y0, 1)
                if _fr.status == 'charging':
                    # Solid cyan pip — actively docked and charging
                    _pip_half = max(_fb_bar_h // 2 - 1, 1)
                    cv2.rectangle(_fb_img, (_fb_ind_x, _fb_mid_y - _pip_half),
                                  (_fb_ind_x + 6, _fb_mid_y + _pip_half), (200, 200, 50), -1)
                elif _fr.status == 'move to charging station':
                    # Hollow cyan pip — en route to charger
                    _pip_half = max(_fb_bar_h // 2 - 1, 1)
                    cv2.rectangle(_fb_img, (_fb_ind_x, _fb_mid_y - _pip_half),
                                  (_fb_ind_x + 6, _fb_mid_y + _pip_half), (200, 200, 50), 1)
                elif _fr.batteryPercent < getattr(self.warehouse, '_limp_mode_batt_pct', 20):
                    # Orange pip — limp mode
                    _pip_half = max(_fb_bar_h // 2 - 1, 1)
                    cv2.rectangle(_fb_img, (_fb_ind_x, _fb_mid_y - _pip_half),
                                  (_fb_ind_x + 6, _fb_mid_y + _pip_half), (0, 140, 255), -1)
                elif _fr.carrying != -1:
                    # Small white pip — carrying a package (higher drain)
                    _pip_half = max(_fb_bar_h // 2 - 1, 1)
                    cv2.rectangle(_fb_img, (_fb_ind_x, _fb_mid_y - _pip_half),
                                  (_fb_ind_x + 6, _fb_mid_y + _pip_half), (180, 180, 180), -1)
        else:
            # No robots yet — placeholder message
            cv2.putText(_fb_img, "no robots", (8, _FB_H // 2), _fb_font, 0.32, (60, 60, 60), 1)

        # Charge threshold line — vertical dashed line at dynamic threshold across full chart height.
        # Threshold adapts to charger pressure (30%-50%). Any bar below = robot seeking charger.
        for _dy in range(_fb_chart_top, _FB_H, 3):
            cv2.line(_fb_img, (_fb_thresh_x, _dy), (_fb_thresh_x, min(_dy + 1, _FB_H - 1)), (60, 60, 180), 1)
        cv2.putText(_fb_img, 'T', (max(_fb_thresh_x - 2, 1), _fb_chart_top - 3), _fb_font, 0.22, (90, 110, 220), 1)

        _fb_need = sum(1 for r in wh.robots if r.batteryPercent <= _fb_dyn_thresh)
        _fb_busy = sum(1 for c in wh.chargers if c.status != 'idle')
        _policy_y = _fb_chart_top - 4
        _x = 2
        for _txt, _col in [
            ('PP', (120, 120, 120)),
            (' T{:.0f}/{:.0f}'.format(_fb_dyn_thresh, _fb_thresh_target), (90, 110, 220)),
            (' P{}/{}'.format(int(_fb_pressure * 100), int(_fb_pressure_raw * 100)), (60, 190, 190)),
            (' E-{}'.format(int(round(_fb_eco_save))), (120, 200, 120) if _fb_eco_save > 0 else (80, 80, 80)),
            (' N{}'.format(_fb_need), (60, 150, 230) if _fb_need > 0 else (80, 80, 80)),
            (' C{}'.format(_fb_busy), (200, 200, 50) if _fb_busy > 0 else (80, 80, 80)),
        ]:
            cv2.putText(_fb_img, _txt, (_x, _policy_y), _fb_font, 0.22, _col, 1)
            (_tw, _th), _ = cv2.getTextSize(_txt, _fb_font, 0.22, 1)
            _x += _tw + 2

        # Mini legend in title row (right side): coloured squares + 1-char labels
        _lg_y = 4
        _lg_items = [
            ((200, 200, 50), True,  'C'),   # Charging (filled cyan)
            ((200, 200, 50), False, 'E'),   # En-route (hollow cyan)
            ((0, 140, 255),  True,  'S'),   # Saving mode (orange)
            ((180, 180, 180), True, 'P'),   # Package (white)
        ]
        _lg_x = sw - 8 * len(_lg_items) - 2
        for _lg_col, _lg_fill, _lg_ch in _lg_items:
            cv2.rectangle(_fb_img, (_lg_x, _lg_y), (_lg_x + 4, _lg_y + 4),
                          _lg_col, -1 if _lg_fill else 1)
            cv2.putText(_fb_img, _lg_ch, (_lg_x, _lg_y + 10),
                        _fb_font, 0.2, (120, 120, 120), 1)
            _lg_x += 8

        composite[_DL_H+1:sh, mw+sw*3:mw+sw*4] = _fb_img

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

        # Sub-view title overlays (drawn onto composite after all sub-views are placed)
        _tfont = cv2.FONT_HERSHEY_SIMPLEX
        _tcol  = (145, 145, 145)
        _zonemap_title   = "ZONE MAP util%" if _util_overlay_active else "ZONE MAP"
        _titles = [
            ("WAREHOUSE",    4,            4),
            ("CHARGERS",     mw + 4,       4),
            ("PACKAGES",     mw+sw+4,      4),
            ("ROBOTS",       mw + 4,       sh + 4),
            ("PKG TARGETS",  mw+sw+4,      sh + 4),
            ("TRAFFIC HEATMAP", mw+sw*2+4, sh + 4),
            (_zonemap_title, mw+sw*2+4,    4),
            ("DEADLINES",    mw+sw*3+4,    4),
            ("FLEET BATT",   mw+sw*3+4,    74),
        ]
        for _ttxt, _tx, _ty in _titles:
            cv2.putText(composite, _ttxt, (_tx, _ty + 8), _tfont, 0.28, _tcol, 1)

        # Zone labels — colour-keyed legend in corner of each view
        _zfont = cv2.FONT_HERSHEY_SIMPLEX
        _zone_info = [
            (1, "IMP", self.warehouse.colourOfImportAreas),
            (2, "STO", self.warehouse.colourOfStorageAreas),
            (3, "EXP", self.warehouse.colourOfExportAreas),
        ]

        # Legend box on main view (bottom-left corner) – semi-transparent bg
        _leg_y = mh - 10 - len(_zone_info) * 12
        _lbg_y0 = max(0, _leg_y - 2)
        _lbg_y1 = min(mh, _leg_y + len(_zone_info) * 12 + 4)
        _lbg_x0, _lbg_x1 = 1, 46
        _roi = composite[_lbg_y0:_lbg_y1, _lbg_x0:_lbg_x1]
        composite[_lbg_y0:_lbg_y1, _lbg_x0:_lbg_x1] = (_roi * 0.35).astype(np.uint8)
        for _li, (_zid, _zlbl, _zcol) in enumerate(_zone_info):
            _ly = _leg_y + _li * 12 + 10
            _bright = tuple(min(int(c * 2.5), 255) for c in _zcol)
            cv2.rectangle(composite, (4, _ly - 7), (12, _ly - 1), _bright, -1)
            cv2.putText(composite, _zlbl, (15, _ly), _zfont, 0.32, (200, 200, 200), 1)

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

        # Fleet summary
        _idle_r = sum(1 for rb in wh.robots if rb.status == 'idle')
        _avg_b  = int(sum(rb.batteryPercent for rb in wh.robots) / max(len(wh.robots), 1))
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
        _pkg_style = self._PKG_TARGET_STYLE.get(_pkg_target_mode, self._PKG_TARGET_STYLE['random'])
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
        _flow_style = self._FLOW_POLICY_STYLE.get(_flow_mode, self._FLOW_POLICY_STYLE['balanced'])
        _flow_col = _flow_style['text']

        _power_mode = getattr(wh, '_power_policy_mode', 'balanced')
        _power_style = self._POWER_POLICY_STYLE.get(_power_mode, self._POWER_POLICY_STYLE['balanced'])
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
            ("robots",   "idle:{}/{} b:{}%".format(_idle_r, len(wh.robots), _avg_b), None),
            ("import",   "{}/{} ({})".format(_n(wh.packagesInImportCount),  _n(wh.numberOfImportSlots),  _r("import")), None),
            ("storage",  "{}/{} ({})".format(_n(wh.packagesInStorageCount), _n(wh.numberOfStorageSlots), _r("storage")), None),
            ("export",   "{}/{} ({})".format(_n(wh.packagesInExportCount),  _n(wh.numberOfExportSlots),  _r("expzone")), None),
            ("util",     _util_str, _util_col),
            ("opt",      _opt_str,  _opt_col),
            ("power",    _power_mode, _power_col),
            ("flow",     _flow_mode, _flow_col),
            ("target",   _target_str, _target_col),
        ]
        # (SIM STATS rendering moved to info panel column 3)

        # ═══════════════════════ INFO PANEL ═══════════════════════
        # 960×165px area below the main view (y = mh..mh+ph = 280..445)
        # 4 columns of 240px: ROBOTS | CHARGERS | SIM STATS | PACKAGES
        composite[mh, :] = 60  # thin separator line
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.26
        ft    = 1
        lh    = 10
        pad_x = 4
        col_w = cw // 4
        white = (220, 220, 220)
        gray  = (140, 140, 140)
        dim   = (85, 85, 85)
        sec_col = (95, 95, 95)     # section-separator label colour

        def put(text, x, y, col, max_x=None):
            nonlocal composite
            if max_x is not None and x < max_x:
                max_w = max_x - x - 2
                (tw, _), _ = cv2.getTextSize(text, font, fs, ft)
                if tw > max_w:
                    while len(text) > 1:
                        text = text[:-1]
                        (tw, _), _ = cv2.getTextSize(text + "\u2026", font, fs, ft)
                        if tw <= max_w:
                            text += "\u2026"
                            break
            cv2.putText(composite, text, (x, y), font, fs, col, ft)
            (tw, _), _ = cv2.getTextSize(text, font, fs, ft)
            return x + tw

        # ── Pixel geometry ──
        _hdr_y      = mh + 10           # header text baseline
        _underline  = mh + 11           # underline row
        _data_start = mh + 22           # first data row baseline (gap avoids overlap)
        _split      = 7                 # rows in each section
        _div_y      = _data_start + _split * lh - lh // 2  # horizontal divider

        # Column dividers (no underline — mh separator is enough)
        composite[mh:mh + ph, col_w]     = 40
        composite[mh:mh + ph, col_w * 2] = 40
        composite[mh:mh + ph, col_w * 3] = 40

        # ── Lookup tables ──
        robot_priority = {"drop off target package": 0, "pick up target package": 1,
                          "move to target dropoff location": 2, "move to target pickup location": 3,
                          "move to charging station": 4, "charging": 5, "idle": 6}
        robot_short = {
            "drop off target package":         "DROPOFF",
            "pick up target package":          "PICKUP",
            "move to target dropoff location": ">>DROP",
            "move to target pickup location":  ">>PICK",
            "move to charging station":        ">>CHRG",
            "charging":                        "CHARGING",
            "idle":                            "IDLE",
        }
        robot_col_map = {
            "DROPOFF": (60, 180, 255), "PICKUP": (60, 180, 255),
            ">>DROP": (80, 220, 150), ">>PICK": (80, 220, 150),
            ">>CHRG": (50, 200, 200), "CHARGING": (50, 200, 200),
            "IDLE": (75,  75,  75),
        }
        pkg_short = {"idle": "idle", "move planned": "planned", "carried": "carried", "error": "ERROR"}
        zone_col = {"import": (200, 160, 80), "storage": (80, 180, 200), "export": (80, 120, 220)}
        charger_priority = {"charging": 0, "charging planned": 1, "idle": 2}

        def batt_col(pct):
            if pct > 50: return (80, 190, 80)
            if pct > 20: return (50, 200, 200)
            return (80, 80, 200)

        def chg_col(c):
            if c.status == "charging":         return (130, 235, 235)
            if c.status == "charging planned": return (65, 145, 145)
            return (45, 90, 90)

        def dl_col(td):
            if td < 0:   return (80, 80, 210)
            if td < 60:  return (70, 130, 230)
            if td < 300: return (50, 200, 200)
            return gray

        now_t = time.time()

        # ────────────── ROBOTS ──────────────
        _nr = len(self.warehouse.robots)
        x = put("ROBOTS ({})  ".format(_nr), pad_x, _hdr_y, white)
        put("status | battery", x, _hdr_y, sec_col)

        # Top 7: by task priority (busiest first)
        _robots_by_status = sorted(self.warehouse.robots, key=lambda r: robot_priority.get(r.status, 7))
        for ri in range(min(_split, len(_robots_by_status))):
            r = _robots_by_status[ri]
            y = _data_start + ri * lh
            if y >= mh + ph - 2: break
            sh_s = robot_short.get(r.status, r.status[:4].upper())
            x = pad_x
            x = put("R{} ".format(r.robotNumber), x, y, gray)
            x = put("{} ".format(sh_s), x, y, robot_col_map.get(sh_s, gray))
            if r.carrying != -1:
                x = put("P{} ".format(_n(r.carrying)), x, y, (80, 160, 220))
            batt = int(r.batteryPercent)
            x = put("{}% ".format(batt), x, y, batt_col(batt))
            # Zone
            _rzone = r.area[:4] if r.area else "?"
            _rtgt = r.areaTarget[:4] if r.areaTarget and r.areaTarget != r.area else ""
            if _rtgt:
                x = put("{}>{}  ".format(_rzone, _rtgt), x, y, zone_col.get(r.area, dim))
            else:
                x = put("{}  ".format(_rzone), x, y, zone_col.get(r.area, dim))
            age = now_t - r.createdAt
            put(self._fmt_age(int(age)), x, y, dim, max_x=col_w)

        # Divider
        composite[_div_y:_div_y + 1, 0:col_w] = 40

        # Bottom 7: by lowest battery
        _robots_by_batt = sorted(self.warehouse.robots, key=lambda r: r.batteryPercent)
        for bi in range(min(_split, len(_robots_by_batt))):
            r = _robots_by_batt[bi]
            y = _div_y + 8 + bi * lh
            if y >= mh + ph - 2: break
            sh_s = robot_short.get(r.status, r.status[:4].upper())
            x = pad_x
            x = put("R{} ".format(r.robotNumber), x, y, (60, 200, 200))
            batt = int(r.batteryPercent)
            x = put("{}% ".format(batt), x, y, batt_col(batt))
            drn = r.batteryDepletingRate * r.batteryDrainMultiplier
            x = put("drain:{:.2f} ".format(drn), x, y, dim)
            x = put("{} ".format(sh_s), x, y, robot_col_map.get(sh_s, gray))
            if r.carrying != -1:
                x = put("P{} ".format(_n(r.carrying)), x, y, (80, 160, 220))
            _rzone = r.area[:4] if r.area else "?"
            put("{}".format(_rzone), x, y, zone_col.get(r.area, dim), max_x=col_w)

        # ────────────── CHARGERS ──────────────
        _nc = len(self.warehouse.chargers)
        _n_busy = sum(1 for c in self.warehouse.chargers if c.status != 'idle')
        put("CHARGERS ({})".format(_nc), col_w + pad_x, _hdr_y, white)
        _pp_mode = getattr(self.warehouse, '_power_policy_mode', 'balanced')
        _pp_style = self._POWER_POLICY_STYLE.get(_pp_mode, self._POWER_POLICY_STYLE['balanced'])
        for mode, bx0, by0, bx1, by1 in self._power_policy_button_layout():
            style = self._POWER_POLICY_STYLE[mode]
            active = (mode == _pp_mode)
            composite[by0:by1, bx0:bx1] = style['bg'] if active else (18, 18, 18)
            composite[by0:by1, bx0:bx0 + 1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by1, bx1 - 1:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by0 + 1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by1 - 1:by1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            _label = style['label']
            _fs_btn = 0.23
            (_tw, _th), _ = cv2.getTextSize(_label, font, _fs_btn, 1)
            _tx = bx0 + max((bx1 - bx0 - _tw) // 2, 1)
            _ty = by0 + (by1 - by0 + _th) // 2
            cv2.putText(composite, _label, (_tx, _ty), font, _fs_btn,
                        style['text'] if active else (85, 85, 85), 1)
        put("mode: {} ({})".format(_pp_mode.upper(), self._POWER_POLICY_HINT.get(_pp_mode, _pp_mode)),
            col_w + pad_x, _data_start - 1, _pp_style['text'], max_x=col_w * 2)
        _chg_data_start = _data_start + lh
        _chg_div_y = _div_y + lh

        # Charger→robot object mapping for battery display
        _crmap_obj = {}
        for _cr in self.warehouse.robots:
            for _ct in _cr.actionQueue:
                if _ct[1] in ("move to charging station", "charging"):
                    _crmap_obj[tuple(_ct[0])] = _cr

        # Top section: all chargers sorted by status
        _chargers_sorted = sorted(self.warehouse.chargers, key=lambda c: charger_priority.get(c.status, 3))
        for ci in range(min(_split, len(_chargers_sorted))):
            c = _chargers_sorted[ci]
            y = _chg_data_start + ci * lh
            if y >= mh + ph - 2: break
            x = col_w + pad_x
            x = put("C{} ".format(c.chargerNumber), x, y, tuple(c.colour))
            _st = c.status
            if _st == "charging":
                x = put("CHARGING ", x, y, chg_col(c))
            elif _st == "charging planned":
                x = put("PLANNED  ", x, y, chg_col(c))
            else:
                x = put("IDLE     ", x, y, chg_col(c))
            _rob_obj = _crmap_obj.get(tuple(c.xyLocation))
            if _rob_obj is not None:
                x = put("R{} ".format(_rob_obj.robotNumber), x, y, (80, 220, 150))
                _rb = int(_rob_obj.batteryPercent)
                put("{}%".format(_rb), x, y, batt_col(_rb), max_x=col_w * 2)

        # If chargers don't fill 7 rows, show remaining chargers beyond 7
        for ci2 in range(_split, min(len(_chargers_sorted), _split * 2)):
            c = _chargers_sorted[ci2]
            y = _chg_data_start + ci2 * lh
            if y >= _chg_div_y: break
            x = col_w + pad_x
            x = put("C{} ".format(c.chargerNumber), x, y, tuple(c.colour))
            _st = c.status
            if _st == "charging":
                x = put("CHARGING ", x, y, chg_col(c))
            elif _st == "charging planned":
                x = put("PLANNED  ", x, y, chg_col(c))
            else:
                x = put("IDLE     ", x, y, chg_col(c))
            _rob_obj = _crmap_obj.get(tuple(c.xyLocation))
            if _rob_obj is not None:
                x = put("R{} ".format(_rob_obj.robotNumber), x, y, (80, 220, 150))
                _rb = int(_rob_obj.batteryPercent)
                put("{}%".format(_rb), x, y, batt_col(_rb), max_x=col_w * 2)

        # Divider
        composite[_chg_div_y:_chg_div_y + 1, col_w:col_w * 2] = 40

        # Bottom section: fleet & power summary
        _avg_batt  = sum(r.batteryPercent for r in self.warehouse.robots) / max(_nr, 1)
        _pressure  = getattr(self.warehouse, '_charger_pressure', 0.0)
        _pressure_raw = getattr(self.warehouse, '_charger_pressure_raw', _pressure)
        _threshold = getattr(self.warehouse, '_charge_threshold', 30)
        _threshold_target = getattr(self.warehouse, '_charge_threshold_target', _threshold)
        _n_idle_r   = sum(1 for r in self.warehouse.robots if r.status == 'idle')
        _n_chrging  = sum(1 for r in self.warehouse.robots if r.status == 'charging')
        _n_enroute  = sum(1 for r in self.warehouse.robots if r.status == 'move to charging station')
        _n_working  = _nr - _n_idle_r - _n_chrging - _n_enroute
        _avg_drain  = sum(r.batteryDepletingRate * r.batteryDrainMultiplier for r in self.warehouse.robots) / max(_nr, 1)

        _fleet_lines = [
            ("avg batt",   "{:.0f}%".format(_avg_batt),     batt_col(int(_avg_batt))),
            ("avg drain",  "{:.3f}/t".format(_avg_drain),    dim),
            ("policy",     "Power/{}".format(getattr(self.warehouse, '_power_policy_mode', 'balanced')[:4]), _pp_style['text']),
            ("pressure",   "{:.0f}/{:.0f}%".format(_pressure * 100, _pressure_raw * 100), (50, 200, 200) if _pressure > 0.5 else gray),
            ("threshold",  "{:.0f}/{:.0f}%".format(_threshold, _threshold_target), gray),
            ("working",    str(_n_working),                  (80, 220, 150)),
            ("charging",   "{} + {} en-rt".format(_n_chrging, _n_enroute), (50, 200, 200)),
            ("idle",       str(_n_idle_r),                   (75, 75, 75)),
        ]
        for fi, (fl, fv, fc) in enumerate(_fleet_lines):
            y = _chg_div_y + 8 + fi * lh
            if y >= mh + ph - 2: break
            x = col_w + pad_x
            x = put("{}: ".format(fl), x, y, sec_col)
            put(fv, x, y, fc, max_x=col_w * 2)

        # ────────────── SIM STATS ──────────────
        _sc2 = col_w * 2
        _sc3 = col_w * 3
        put("SIM STATS", _sc2 + pad_x, _hdr_y, white)
        _fp_mode = getattr(self.warehouse, '_flow_policy_mode', 'balanced')
        _fp_style = self._FLOW_POLICY_STYLE.get(_fp_mode, self._FLOW_POLICY_STYLE['balanced'])
        for mode, bx0, by0, bx1, by1 in self._flow_policy_button_layout():
            style = self._FLOW_POLICY_STYLE[mode]
            active = (mode == _fp_mode)
            composite[by0:by1, bx0:bx1] = style['bg'] if active else (18, 18, 18)
            composite[by0:by1, bx0:bx0 + 1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by1, bx1 - 1:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by0 + 1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by1 - 1:by1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            _label = style['label']
            _fs_btn = 0.23
            (_tw, _th), _ = cv2.getTextSize(_label, font, _fs_btn, 1)
            _tx = bx0 + max((bx1 - bx0 - _tw) // 2, 1)
            _ty = by0 + (by1 - by0 + _th) // 2
            cv2.putText(composite, _label, (_tx, _ty), font, _fs_btn,
                        style['text'] if active else (85, 85, 85), 1)
        put("mode: {} ({})".format(_fp_mode.upper(), self._FLOW_POLICY_HINT.get(_fp_mode, _fp_mode)),
            _sc2 + pad_x, _data_start - 1, _fp_style['text'], max_x=_sc3)
        _sim_data_start = _data_start + lh
        _swhite = (200, 200, 200)
        _sdim   = (95, 95, 95)
        _s_val_x = _sc2 + 52  # label : value offset
        for si, (lbl, val, vcol) in enumerate(stats_lines):
            y = _sim_data_start + si * lh
            if y >= mh + ph - 2:
                break
            put(lbl + ":", _sc2 + pad_x, y, _sdim)
            put(val, _s_val_x, y, vcol or _swhite, max_x=_sc3)

        # ────────────── PACKAGES ──────────────
        _np = len(self.warehouse.packages)
        _nm = self.warehouse.packagesMaxQuantity
        put("PKGS ({}/{})".format(_np, _nm), col_w * 3 + pad_x, _hdr_y, white)
        _active_mode = getattr(self.warehouse, '_pkg_target_mode', 'random')
        _pm_style = self._PKG_TARGET_STYLE.get(_active_mode, self._PKG_TARGET_STYLE['random'])
        for mode, bx0, by0, bx1, by1 in self._pkg_target_button_layout():
            style = self._PKG_TARGET_STYLE[mode]
            active = (mode == _active_mode)
            composite[by0:by1, bx0:bx1] = style['bg'] if active else (18, 18, 18)
            composite[by0:by1, bx0:bx0 + 1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by1, bx1 - 1:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by0:by0 + 1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            composite[by1 - 1:by1, bx0:bx1] = style['edge'] if active else (42, 42, 42)
            _label = style['label']
            _fs_btn = 0.23
            (_tw, _th), _ = cv2.getTextSize(_label, font, _fs_btn, 1)
            _tx = bx0 + max((bx1 - bx0 - _tw) // 2, 1)
            _ty = by0 + (by1 - by0 + _th) // 2
            cv2.putText(composite, _label, (_tx, _ty), font, _fs_btn,
                        style['text'] if active else (85, 85, 85), 1)
        put("mode: {} ({})".format(_active_mode.upper(), _pm_style['hint']),
            col_w * 3 + pad_x, _data_start - 1, _pm_style['text'], max_x=cw)

        def _draw_pkg(x_start, y, p, max_end):
            td = p.timeToDeadline.total_seconds()
            dl_str = "LATE" if td < 0 else self._fmt_age(int(td))
            age = now_t - p.createdAt
            age_str = self._fmt_age(int(age))
            tgt = p.areaTarget if p.areaTarget not in ("none", "") else ""
            st = pkg_short.get(p.status, p.status[:3])
            x = x_start
            # ID
            x = put("P{} ".format(_n(p.packageNumber)), x, y, tuple(p.colour))
            # Deadline + Age
            x = put("{} ".format(dl_str), x, y, dl_col(td))
            x = put("{} ".format(age_str), x, y, (200, 160, 80))
            # Zone > target
            x = put("{} ".format(p.area[:4]), x, y, zone_col.get(p.area, gray))
            if tgt:
                x = put(">{} ".format(tgt[:4]), x, y, zone_col.get(tgt, gray))
            # Carrier / status
            if p.carrier != -1:
                x = put("R{} ".format(p.carrier), x, y, (80, 160, 220))
            x = put("{} ".format(st), x, y, dim)
            # Weight
            _wt = getattr(p.itemValues, 'weight', None) if p.itemValues else None
            if _wt:
                x = put("{}kg ".format(_wt), x, y, (140, 140, 100))
            # Item name
            _iname = getattr(p.itemValues, 'name', '') if p.itemValues else ''
            if _iname:
                x = put("{} ".format(_iname), x, y, (120, 140, 120), max_x=max_end)
            # Destination city (fills remaining space)
            _city = getattr(p.addressTo, 'city', '') if hasattr(p, 'addressTo') and p.addressTo else ''
            if _city and x < max_end - 10:
                put(_city, x, y, (110, 110, 130), max_x=max_end)

        # All packages by deadline (most urgent first) — fills entire column
        _pkgs_deadline = sorted(self.warehouse.packages, key=lambda p: p.timeToDeadline)
        _pkg_rows_start = _data_start + lh
        _max_pkg_rows = _split * 2
        for pi in range(_max_pkg_rows):
            if pi >= len(_pkgs_deadline): break
            y = _pkg_rows_start + pi * lh
            if y >= mh + ph - 2: break
            _draw_pkg(col_w * 3 + pad_x, y, _pkgs_deadline[pi], cw)

        # Update rolling chart histories (sample once per second via wall-clock)
        if time.time() - self._hist_last_sample >= 1.0:
            self._hist_last_sample = time.time()
            overdue = sum(1 for p in self.warehouse.packages if p.timeToDeadline.total_seconds() < 0)
            self._late_total += overdue   # accumulate: each 1-s sample adds current late count
            mean_batt = float(np.mean([r.batteryPercent for r in self.warehouse.robots])) if self.warehouse.robots else 0.0
            _charge_cutoff = getattr(self.warehouse, '_charge_threshold', 20)
            need_charge = sum(1 for r in self.warehouse.robots if r.batteryPercent <= _charge_cutoff)
            charging    = sum(1 for r in self.warehouse.robots if r.status == "charging")
            idle_count  = sum(1 for r in self.warehouse.robots if r.status == "idle")
            self._hist_imported = np.roll(self._hist_imported, -1); self._hist_imported[-1] = self.warehouse.packagesRollingCount
            self._hist_exported = np.roll(self._hist_exported, -1); self._hist_exported[-1] = self.warehouse.packageExportRollingCount
            self._hist_overdue  = np.roll(self._hist_overdue,  -1); self._hist_overdue[-1]  = overdue
            self._hist_late     = np.roll(self._hist_late,     -1); self._hist_late[-1]     = self._late_total
            self._hist_batt         = np.roll(self._hist_batt,         -1); self._hist_batt[-1]         = mean_batt
            self._hist_robots       = np.roll(self._hist_robots,       -1); self._hist_robots[-1]       = len(self.warehouse.robots)
            self._hist_need_charge  = np.roll(self._hist_need_charge,  -1); self._hist_need_charge[-1]  = need_charge
            self._hist_charging     = np.roll(self._hist_charging,     -1); self._hist_charging[-1]     = charging
            self._hist_idle         = np.roll(self._hist_idle,         -1); self._hist_idle[-1]         = idle_count
            # Flow control histories
            self._hist_import_cap   = np.roll(self._hist_import_cap,   -1); self._hist_import_cap[-1]   = getattr(wh, '_flow_import_cap', wh.packagesMaxQuantity)
            self._hist_throughput   = np.roll(self._hist_throughput,   -1); self._hist_throughput[-1]   = getattr(wh, '_flow_throughput', 0) * 60  # per minute for readability
            self._hist_delivery     = np.roll(self._hist_delivery,     -1); self._hist_delivery[-1]     = getattr(wh, '_flow_avg_delivery', 0)
            n_pkgs = max(len(wh.packages), 1)
            self._hist_overdue_pct  = np.roll(self._hist_overdue_pct,  -1); self._hist_overdue_pct[-1]  = overdue / n_pkgs * 100
            self._hist_count    = min(self._hist_count + 1, len(self._hist_exported))
            # Rate-of-change: compare with previous snapshot
            wh = self.warehouse
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
                "need_chg": need_charge,
                "chging":   charging,
                "idle":     idle_count,
            }
            if self._prev_stats is not None:
                self._rate_stats = {k: cur[k] - self._prev_stats.get(k, cur[k]) for k in cur}
            self._prev_stats = cur

        # Draw charts strip
        composite[mh + ph, :] = 55  # chart separator
        self._draw_charts(composite, mh + ph + 1, ch - 1, cw)

        # Re-draw separators and column dividers last so they sit on top of chart content
        composite[mh, :]                         = 60
        composite[mh + ph, :]                    = 55
        composite[mh:mh + ph, col_w]             = 40
        composite[mh:mh + ph, col_w * 2]         = 40
        composite[mh:mh + ph, col_w * 3]         = 40

        # ── Panel outlines ─────────────────────────────────────────────
        _oc  = (45, 45, 45)   # subtle dark-grey border colour
        _cw4 = cw // 4        # chart column width (mirrors w4 in _draw_charts)
        def _outline(x0, y0, x1, y1):
            cv2.rectangle(composite, (x0, y0), (x1 - 1, y1 - 1), _oc, 1)
        # Top row: main view + 4 sub-views + zone map + stats
        _outline(0,          0,   mw,          mh)           # main warehouse view
        _outline(mw,         0,   mw + sw,     sh)           # chargers binary
        _outline(mw + sw,    0,   mw + sw * 2, sh)           # packages binary
        _outline(mw,         sh,  mw + sw,     sh * 2)       # robots binary
        _outline(mw + sw,    sh,  mw + sw * 2, sh * 2)       # package targets
        _outline(mw + sw*2,  0,   mw + sw * 3, sh)           # zone map
        _outline(mw + sw*2,  sh,  mw + sw * 3, sh * 2)       # traffic heatmap
        _outline(mw + sw*3,  0,   mw + sw * 4, sh)            # deadlines + fleet batt
        _outline(mw + sw*3,  sh,  mw + sw * 4, sh * 2)       # (empty)
        # Info panel (one outline for the whole strip)
        _outline(0,           mh,       cw,           mh + ph)      # info panel
        # Chart strip (four charts)
        _outline(0,           mh + ph,  _cw4,         mh + ph + ch) # chart 1
        _outline(_cw4,        mh + ph,  _cw4 * 2,     mh + ph + ch) # chart 2
        _outline(_cw4 * 2,    mh + ph,  _cw4 * 3,     mh + ph + ch) # chart 3
        _outline(_cw4 * 3,    mh + ph,  cw,            mh + ph + ch) # chart 4

        # Zone-map panel controls (zone buttons at bottom, brush sizes on right)
        self.paint_handler.draw_buttons(composite, _zfont)

        cv2.imshow("warehouse", composite)
        key = cv2.waitKey(1) & 0xFF
        self.paint_handler.handle_key(key)
        return self

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
                             zone_rates={"Imp": _r("import"), "Sto": _r("storage"), "Exp": _r("expzone")})

        # Chart 4: Fleet — Batt% on fixed scale; Robots/NeedCharge/Charging/Idle share robot-count scale
        mini_chart(w4 * 3, cw, "Fleet Health",
                   [(self._hist_batt,        ( 50, 200, 200), "Batt%"),
                    (self._hist_robots,      (200, 140,  70), "Robots"),
                    (self._hist_need_charge, ( 30,  80, 230), "NeedChg"),
                    (self._hist_charging,    ( 60, 220,  80), "Chging"),
                    (self._hist_idle,        ( 90,  90,  90), "Idle")],
                   shared_scale={"Robots", "NeedChg", "Chging", "Idle"},
                   fixed_scales={"Batt%": (0, 100)},
                   rates={"Batt%": _r("batt"), "NeedChg": _r("need_chg"), "Chging": _r("chging"), "Idle": _r("idle")})

    def _draw_zone_bars(self, canvas, x0, y0, x1, y1, zone_rates=None):
        """Render zone capacity fill-bars into canvas[y0:y1, x0:x1]."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        w, h  = x1 - x0, y1 - y0
        arr   = np.full((h, w, 3), (8, 8, 8), dtype=np.uint8)
        cv2.putText(arr, "Zone Capacity", (4, 11), font, 0.32, (145, 145, 145), 1)
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
        canvas[y0:y1, x0:x1] = arr

    def gameEnd(self):
        self.exit = True

if __name__ == '__main__':

    mainGame = MainGame()
    loop_count = 0

    print('-- Game Loop Start --')
    while not mainGame.exit:
        mainGame.gameLoop(loop_count)
        loop_count += 1
    print('-- Game Loop End --')

    time.sleep(2)
    cv2.destroyAllWindows()