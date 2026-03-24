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

class MainGame:
    def __init__(self):
        print("--- MainGame Init ---")
        self.timeStart = int(time.time())
        self.exit = False
        # User Parameters
        self.warehouse_res = (80,70)
        self.warehouse_windowBackgroundColour = [50,50,50]
        self.warehouse_windowRes = (320, 280) # upsized resolution (width, height)
        self.sim_frametime  = 1 / 40   # 40 sim ticks/sec
        self.draw_frametime = 1 / 20   # 20 display fps
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

        # Zone painting — targeting the zone-map sub-view (top-right panel)
        # zone-map panel rect in composite: x = mw + sw*2, y = 0, w = sw, h = sh
        _zone_view_rect = (
            self.warehouse_windowRes[0] + self.sub_windowRes[0] * 2,  # x0
            0,                                                          # y0
            self.sub_windowRes[0],                                      # pw
            self.sub_windowRes[1],                                      # ph
        )
        self.paint_handler = PaintHandler(self.warehouse, self.warehouse_res, _zone_view_rect)
        cv2.setMouseCallback("warehouse", self.paint_handler.on_mouse)

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
        self.chart_h = 130
        self.composite_windowRes = (self.warehouse_windowRes[0] + self.sub_windowRes[0] * 3, self.warehouse_windowRes[1] + self.panel_h + self.chart_h)  # 800x575
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
        self._hist_tick     = 0
        self._hist_count    = 0          # samples actually filled so far
        self._hist_min_win  = 30         # minimum visible window (seconds)
        self._hist_sample_every = max(1, round(1.0 / self.draw_frametime))  # draws per second
        # Rate-of-change tracking (snapshot every 1s sample)
        self._prev_stats = None   # dict of key→value at last sample
        self._rate_stats = {}     # dict of key→delta/sec
        screen_w, screen_h = Display_Functions.get_screen_resolution()
        cx = (screen_w - self.composite_windowRes[0]) // 2
        cy = (screen_h - self.composite_windowRes[1]) // 2
        Display_Functions.init_borderless_window("warehouse", self.composite_windowRes, [cx, cy])
        print('- warehouse_res: {}, arrayShape: {}, composite_windowRes: {}, screen_center_pos: [{},{}]'.format(
            self.warehouse_res, np.shape(self.warehouse_windowArray), self.composite_windowRes, cx, cy))
        return self

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
        self.warehouse = self.warehouse.update_warehouse()

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
        ch = self.chart_h                   # 130
        cw = mw + sw * 3                    # 800
        zone_colours = {1: self.warehouse.colourOfImportAreas, 2: self.warehouse.colourOfStorageAreas, 3: self.warehouse.colourOfExportAreas}
        composite = np.zeros((mh + ph + ch, cw, 3), dtype=np.uint8)

        # Main view (left column)
        composite[0:mh, 0:mw] = cv2.resize(self.warehouseWindow, (mw, mh), interpolation=cv2.INTER_NEAREST)

        # Binary sub-views: multiply by 255, resize, convert to BGR
        def place_binary(arr, row, col):
            img = cv2.resize((arr * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            composite[row*sh:(row+1)*sh, mw+col*sw:mw+(col+1)*sw] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        place_binary(self.warehouse.chargersInWarehouse,       0, 0)
        place_binary(self.warehouse.packagesInWarehouse,        0, 1)
        place_binary(self.warehouse.robotsInWarehouse,          1, 0)
        # Package targets sub-view: arrows on blank canvas, target dots stamped on top
        sub_scale = sw // self.warehouse_res[0]  # 2
        pkg_tgt_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        Draw_Warehouse.draw_package_arrows(pkg_tgt_img, self.warehouse.packages, sub_scale)
        tgt_resized = cv2.resize((self.warehouse.packageTargetsInWarehouse * 255).astype(np.uint8),
                                  (sw, sh), interpolation=cv2.INTER_NEAREST)
        pkg_tgt_img[tgt_resized > 0] = (255, 255, 255)
        composite[sh:2*sh, mw+sw:mw+2*sw] = pkg_tgt_img

        # Zone map (top-right)
        zoneMap_display = Draw_Warehouse.draw_zoneMap_display(self.warehouse.zoneMap, zone_colours)
        composite[0:sh, mw+sw*2:mw+sw*3] = cv2.resize(zoneMap_display, (sw, sh), interpolation=cv2.INTER_NEAREST)

        # Sub-view title overlays (drawn onto composite after all sub-views are placed)
        _tfont = cv2.FONT_HERSHEY_SIMPLEX
        _tcol  = (110, 110, 110)
        _titles = [
            ("WAREHOUSE",   4,        4),
            ("CHARGERS",    mw + 4,   4),
            ("PACKAGES",    mw+sw+4,  4),
            ("ROBOTS",      mw + 4,   sh + 4),
            ("PKG TARGETS", mw+sw+4,  sh + 4),
            ("ZONE MAP",    mw+sw*2+4, 4),
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

        # Legend box on zone sub-view (bottom-left corner) – semi-transparent bg
        _leg_y2 = sh - 10 - len(_zone_info) * 11
        _zbg_y0 = max(0, _leg_y2 - 2)
        _zbg_y1 = min(sh, _leg_y2 + len(_zone_info) * 11 + 3)
        _ox = mw + sw * 2
        _zbg_x0, _zbg_x1 = _ox + 1, _ox + 42
        _zroi = composite[_zbg_y0:_zbg_y1, _zbg_x0:_zbg_x1]
        composite[_zbg_y0:_zbg_y1, _zbg_x0:_zbg_x1] = (_zroi * 0.35).astype(np.uint8)
        for _li, (_zid, _zlbl, _zcol) in enumerate(_zone_info):
            _ly = _leg_y2 + _li * 11 + 9
            _bright = tuple(min(int(c * 2.5), 255) for c in _zcol)
            cv2.rectangle(composite, (_ox + 3, _ly - 6), (_ox + 9, _ly - 1), _bright, -1)
            cv2.putText(composite, _zlbl, (_ox + 12, _ly), _zfont, 0.25, (200, 200, 200), 1)

        # Bottom-right sub-view: sim stats
        stats_img = np.zeros((sh, sw, 3), dtype=np.uint8)
        sfont = cv2.FONT_HERSHEY_SIMPLEX
        sfs   = 0.28
        scol  = (110, 110, 110)
        swhite = (200, 200, 200)
        cv2.putText(stats_img, "SIM STATS", (4, 10), sfont, 0.30, scol, 1)

        now_t = time.time()
        wh = self.warehouse
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

        stats_lines = [
            ("elapsed",  self._fmt_elapsed(self.timeElapsed)),
            ("loops",    "{} ({})".format(_n(wh.warehouseLoopCount),    _r("loops"))),
            ("exported", "{} ({})".format(_n(wh.packageExportRollingCount), _r("export"))),
            ("pkgs",     "{}/{} ({})".format(_n(wh.packagesInWarehouseCount), _n(wh.packagesMaxQuantity), _r("pkgs"))),
            ("robots",   "{}".format(len(wh.robots))),
            ("chargers", "{}".format(len(wh.chargers))),
            ("import",   "{}/{} ({})".format(_n(wh.packagesInImportCount),  _n(wh.numberOfImportSlots),  _r("import"))),
            ("storage",  "{}/{} ({})".format(_n(wh.packagesInStorageCount), _n(wh.numberOfStorageSlots), _r("storage"))),
            ("export",   "{}/{} ({})".format(_n(wh.packagesInExportCount),  _n(wh.numberOfExportSlots),  _r("expzone"))),
            ("newest",   young_pkg),
            ("oldest",   old_pkg),
        ]
        lbl_x = 4
        val_x = 48
        for si, (lbl, val) in enumerate(stats_lines):
            sy2 = 22 + si * 11
            if sy2 > sh - 4:
                break
            cv2.putText(stats_img, lbl + ":", (lbl_x, sy2), sfont, sfs, scol, 1)
            max_val_w = sw - val_x - 2
            def _fits(s):
                (w2, _), _ = cv2.getTextSize(s, sfont, sfs, 1)
                return w2 <= max_val_w
            # Cascade: full → drop rate suffix → char-trim with ellipsis
            if not _fits(val):
                # Try stripping trailing ( ... ) rate annotation
                paren = val.rfind(" (")
                if paren != -1:
                    val = val[:paren]
            if not _fits(val):
                # Last resort: trim characters and append ellipsis
                while len(val) > 1:
                    val = val[:-1]
                    if _fits(val + "\u2026"):
                        val = val + "\u2026"
                        break
            cv2.putText(stats_img, val, (val_x, sy2), sfont, sfs, swhite, 1)
        composite[sh:2*sh, mw+sw*2:mw+sw*3] = stats_img

        # Info panel
        composite[mh, :] = 60  # thin separator line
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.36
        ft    = 1
        lh    = 14
        pad_x = 6
        col_w = cw // 3
        white = (220, 220, 220)
        gray  = (150, 150, 150)

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

        robot_priority = {"drop off target package": 0, "pick up target package": 1,
                          "move to target dropoff location": 2, "move to target pickup location": 3,
                          "move to charging station": 4, "charging": 5, "idle": 6}
        sorted_robots_active = sorted(self.warehouse.robots, key=lambda r: robot_priority.get(r.status, 7))[:7]
        sorted_robots_batt   = sorted(self.warehouse.robots, key=lambda r: r.batteryPercent)[:3]

        charger_priority = {"charging": 0, "charging planned": 1, "idle": 2}
        sorted_chargers = sorted(self.warehouse.chargers, key=lambda c: charger_priority.get(c.status, 3))[:10]

        sorted_packages = sorted(self.warehouse.packages, key=lambda p: p.timeToDeadline)[:10]

        # Headers with live entity counts / max
        put("ROBOTS ({})".format(len(self.warehouse.robots)),                                          pad_x,             mh + lh + 2, white)
        put("CHARGERS ({})".format(len(self.warehouse.chargers)),                                      col_w + pad_x,     mh + lh + 2, white)
        put("PACKAGES ({}/{})".format(len(self.warehouse.packages), self.warehouse.packagesMaxQuantity), col_w * 2 + pad_x, mh + lh + 2, white)
        composite[mh + lh + 5:mh + lh + 6, :] = 45   # underline below column headers
        composite[mh:, col_w]     = 40
        composite[mh:, col_w * 2] = 40

        robot_status_short = {
            "drop off target package":         "dropoff",
            "pick up target package":          "pickup",
            "move to target dropoff location": "->dropoff",
            "move to target pickup location":  "->pickup",
            "move to charging station":        "->charger",
            "charging":                        "charging",
            "idle":                            "idle",
        }
        pkg_status_short = {
            "idle": "idle", "move planned": "planned",
            "carried": "carried", "error": "ERR",
        }
        robot_action_col = {
            "dropoff":   (60, 180, 255),   # orange  – delivering
            "pickup":    (60, 180, 255),   # orange  – picking up
            "->dropoff": (80, 220, 150),   # lime    – in transit
            "->pickup":  (80, 220, 150),   # lime    – in transit
            "->charger": (50, 200, 200),   # yellow  – battery concern
            "charging":  (50, 200, 200),   # yellow
            "idle":      (90,  90,  90),   # dim
        }
        zone_col = {
            "import":  (200, 160,  80),    # warm blue
            "storage": ( 80, 180, 200),    # amber
            "export":  ( 80, 120, 220),    # orange-red (base; overridden per package)
        }

        def batt_col(pct):
            if pct > 50: return ( 80, 190,  80)   # green
            if pct > 20: return ( 50, 200, 200)   # yellow
            return              ( 80,  80, 200)   # red

        def charger_status_col(c):
            if c.status == "charging":         return (130, 235, 235)   # bright teal
            if c.status == "charging planned": return ( 65, 145, 145)   # mid teal
            return                                    ( 45,  90,  90)   # dim teal – idle

        def pkg_deadline_col(td):
            if td < 0:   return ( 80,  80, 210)   # red   – overdue
            if td < 60:  return ( 70, 130, 230)   # orange
            if td < 300: return ( 50, 200, 200)   # yellow
            return gray

        def draw_robot_row(x, y, r, id_col):
            batt  = int(r.batteryPercent)
            short = robot_status_short.get(r.status, r.status[:9])
            x = put("R{} ".format(r.robotNumber), x, y, id_col)
            x = put("{} ".format(short), x, y, robot_action_col.get(short, gray))
            if r.carrying != -1:
                x = put("[P{}] ".format(_n(r.carrying)), x, y, (80, 160, 220))
            put("{}%".format(batt), x, y, batt_col(batt), max_x=col_w)

        # Divider between top-7 active and top-3 lowest-battery rows
        sep_y = mh + lh + 2 + 7 * lh + lh // 2
        composite[sep_y:sep_y + 1, 0:col_w] = 50

        for i in range(10):
            y = mh + lh + 2 + (i + 1) * lh
            if y >= mh + ph - 2:
                break

            if i < 7:
                if i < len(sorted_robots_active):
                    draw_robot_row(pad_x, y, sorted_robots_active[i], gray)
            else:
                bi = i - 7
                if bi < len(sorted_robots_batt):
                    draw_robot_row(pad_x, y, sorted_robots_batt[bi], (60, 200, 200))

            if i < len(sorted_chargers):
                c = sorted_chargers[i]
                x = col_w + pad_x
                x = put("C{} ".format(c.chargerNumber), x, y, tuple(c.colour))
                put(c.status, x, y, charger_status_col(c), max_x=col_w * 2)

            if i < len(sorted_packages):
                p = sorted_packages[i]
                td = p.timeToDeadline.total_seconds()
                if td < 0:
                    td_str = "LATE "
                else:
                    td_str = self._fmt_age(int(td)) + " "
                tgt    = p.areaTarget if p.areaTarget not in ("none", "") else ""
                arrow  = "->{} ".format(tgt[:4]) if tgt else " "
                st_str = "[{}]".format(pkg_status_short.get(p.status, p.status[:6]))
                x = col_w * 2 + pad_x
                x = put("P{} ".format(_n(p.packageNumber)), x, y, tuple(p.colour))
                x = put(p.area[:4], x, y, zone_col.get(p.area, gray))
                x = put(arrow, x, y, gray)
                x = put(td_str, x, y, pkg_deadline_col(td))
                put(st_str, x, y, gray, max_x=cw)

        # Update rolling chart histories (sample once per second)
        self._hist_tick += 1
        if self._hist_tick >= self._hist_sample_every:
            self._hist_tick = 0
            overdue = sum(1 for p in self.warehouse.packages if p.timeToDeadline.total_seconds() < 0)
            self._late_total += overdue   # accumulate: each 1-s sample adds current late count
            mean_batt = float(np.mean([r.batteryPercent for r in self.warehouse.robots])) if self.warehouse.robots else 0.0
            need_charge = sum(1 for r in self.warehouse.robots if r.batteryPercent <= 20)
            charging    = sum(1 for r in self.warehouse.robots if r.status == "charging")
            self._hist_imported = np.roll(self._hist_imported, -1); self._hist_imported[-1] = self.warehouse.packagesRollingCount
            self._hist_exported = np.roll(self._hist_exported, -1); self._hist_exported[-1] = self.warehouse.packageExportRollingCount
            self._hist_overdue  = np.roll(self._hist_overdue,  -1); self._hist_overdue[-1]  = overdue
            self._hist_late     = np.roll(self._hist_late,     -1); self._hist_late[-1]     = self._late_total
            self._hist_batt         = np.roll(self._hist_batt,         -1); self._hist_batt[-1]         = mean_batt
            self._hist_robots       = np.roll(self._hist_robots,       -1); self._hist_robots[-1]       = len(self.warehouse.robots)
            self._hist_need_charge  = np.roll(self._hist_need_charge,  -1); self._hist_need_charge[-1]  = need_charge
            self._hist_charging     = np.roll(self._hist_charging,     -1); self._hist_charging[-1]     = charging
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
        composite[mh + lh + 5 : mh + lh + 6, :] = 45
        composite[mh:, col_w]                    = 40
        composite[mh:, col_w * 2]                = 40

        # ── Panel outlines ─────────────────────────────────────────────
        _oc  = (45, 45, 45)   # subtle dark-grey border colour
        _cw3 = cw // 3        # chart column width (mirrors w3 in _draw_charts)
        def _outline(x0, y0, x1, y1):
            cv2.rectangle(composite, (x0, y0), (x1 - 1, y1 - 1), _oc, 1)
        # Top row: main view + 4 sub-views + zone map + stats
        _outline(0,          0,   mw,          mh)           # main warehouse view
        _outline(mw,         0,   mw + sw,     sh)           # chargers binary
        _outline(mw + sw,    0,   mw + sw * 2, sh)           # packages binary
        _outline(mw,         sh,  mw + sw,     sh * 2)       # robots binary
        _outline(mw + sw,    sh,  mw + sw * 2, sh * 2)       # package targets
        _outline(mw + sw*2,  0,   mw + sw * 3, sh)           # zone map
        _outline(mw + sw*2,  sh,  mw + sw * 3, sh * 2)       # sim stats
        # Info panel (one outline for the whole strip)
        _outline(0,           mh,       cw,           mh + ph)      # info panel
        # Chart strip (three charts)
        _outline(0,           mh + ph,  _cw3,         mh + ph + ch) # chart 1
        _outline(_cw3,        mh + ph,  _cw3 * 2,     mh + ph + ch) # chart 2
        _outline(_cw3 * 2,    mh + ph,  cw,            mh + ph + ch) # chart 3

        # Paint mode HUD — bottom-right corner of main view
        self.paint_handler.draw_hud(composite, _zfont)

        cv2.imshow("warehouse", composite)
        key = cv2.waitKey(1) & 0xFF
        self.paint_handler.handle_key(key)
        return self

    def _draw_charts(self, canvas, y_off, ch, cw):
        """Render three side-by-side charts into canvas starting at row y_off."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        _n   = self._n
        w3   = cw // 3
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
            arr[[0, -1], :] = (55, 55, 55)
            arr[:, [0, -1]] = (55, 55, 55)
            cv2.putText(arr, "{} ({})".format(title, span_lbl), (4, 11), font, 0.30, (110, 110, 110), 1)
            # Pre-compute shared range if requested
            s_min, s_rng = 0.0, 1.0
            if shared_scale:
                shared_views = [data[-window:] for data, _, label in series if label in shared_scale]
                if shared_views:
                    s_min = float(min(np.min(v) for v in shared_views))
                    s_max = float(max(np.max(v) for v in shared_views))
                    s_rng = max(s_max - s_min, 1.0)
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
                cv2.putText(arr, val_str, (4, label_y), font, 0.3, col, 1)
                label_y += 10
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
            canvas[y0:y1, x0:x1] = arr

        # Chart 1: Imported/Exports/Tot Late share one y-scale; Overdue (instantaneous) on its own
        mini_chart(0, w3, "Imports, Exports & Overdue",
                   [(self._hist_imported, ( 60, 200,  60), "Imported"),   # green  – matches import zone
                    (self._hist_exported, ( 60,  60, 200), "Exported"),   # red    – matches export zone
                    (self._hist_late,     ( 40, 160, 240), "Tot Late"),   # orange – cumulative overdue
                    (self._hist_overdue,  (180, 100, 240), "Overdue")],   # pink   – instantaneous overdue
                   shared_scale={"Imported", "Exported", "Tot Late"},
                   rates={"Imported": _r("tot_imp"), "Exported": _r("export"), "Tot Late": _r("tot_late"), "Overdue": _r("overdue")})

        # Chart 2: Zone capacity fill bars
        self._draw_zone_bars(canvas, w3, y0, w3 * 2, y1,
                             zone_rates={"Import": _r("import"), "Storage": _r("storage"), "Export": _r("expzone")})

        # Chart 3: Fleet — Batt% on fixed scale; Robots/NeedCharge/Charging share robot-count scale
        mini_chart(w3 * 2, cw, "Fleet Health",
                   [(self._hist_batt,        ( 50, 200, 200), "Batt%"),
                    (self._hist_robots,      (200, 140,  70), "Robots"),
                    (self._hist_need_charge, ( 30,  80, 230), "NeedChg"),
                    (self._hist_charging,    ( 60, 220,  80), "Chging")],
                   shared_scale={"Robots", "NeedChg", "Chging"},
                   fixed_scales={"Batt%": (0, 100)},
                   rates={"Batt%": _r("batt"), "NeedChg": _r("need_chg"), "Chging": _r("chging")})

    def _draw_zone_bars(self, canvas, x0, y0, x1, y1, zone_rates=None):
        """Render zone capacity fill-bars into canvas[y0:y1, x0:x1]."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        w, h  = x1 - x0, y1 - y0
        arr   = np.full((h, w, 3), (8, 8, 8), dtype=np.uint8)
        arr[[0, -1], :] = (55, 55, 55)
        arr[:, [0, -1]] = (55, 55, 55)
        cv2.putText(arr, "Zone Capacity", (4, 11), font, 0.32, (110, 110, 110), 1)
        wh = self.warehouse
        zones = [
            ("Import",  wh.packagesInImportCount,  wh.numberOfImportSlots,  (200, 130,  60)),
            ("Storage", wh.packagesInStorageCount, wh.numberOfStorageSlots, ( 80, 180, 200)),
            ("Export",  wh.packagesInExportCount,  wh.numberOfExportSlots,  ( 80, 120, 220)),
        ]
        lbl_w     = 56                         # wide enough for "Storage" at fs=0.3
        bar_max_w = w - lbl_w - 56 - 4          # 266-56-56-4=150; leaves 56px right margin
        bar_h     = 18
        bar_slot  = 37                           # bar_h + gap; 3 slots = 111px, fits in 129px with header
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