import cv2
import numpy as np

# Zone constants (must match warehouse.py)
ZONE_NONE    = 0
ZONE_IMPORT  = 1
ZONE_STORAGE = 2
ZONE_EXPORT  = 3


class PaintHandler:
    """Handles user zone-painting input and on-screen zone-map panel controls.

    Painting mutates warehouse.zoneMap and populates warehouse.pending_zone_changes
    for reconciliation on the next sim tick.

    Controls:
        Left-drag                 — paint active zone
        Right-drag                — erase (zone 0 / neutral)
        Zone buttons (bottom)     — select zone / erase
        Brush buttons (right top) — select brush size (1 / 3 / 5)
        Strategy toggles (right bottom) — enable/disable optimizer strategies
        Key L                     — hot-reload all map PNGs from disk
    """

    ZONE_NAMES = {
        ZONE_NONE:    'ERASE',
        ZONE_IMPORT:  'IMPORT',
        ZONE_STORAGE: 'STORAGE',
        ZONE_EXPORT:  'EXPORT',
    }
    # BGR colours for the HUD label — brightened versions of zone area colours
    ZONE_COLS = {
        ZONE_NONE:    ( 90,  90,  90),
        ZONE_IMPORT:  ( 60, 200,  60),
        ZONE_STORAGE: ( 80, 180, 200),
        ZONE_EXPORT:  ( 60,  60, 200),
    }

    # Height (px) of the zone-button strip at the bottom of the zone-map panel.
    BUTTON_H = 16
    # Width (px) of the vertical brush-size strip on the right of the zone-map panel.
    BRUSH_W  = 16
    # Zone/erase buttons (bottom strip, full panel width) — (zone_id, short label)
    _BUTTONS = [
        (ZONE_NONE,    'CLR'),
        (ZONE_IMPORT,  'IMP'),
        (ZONE_STORAGE, 'STO'),
        (ZONE_EXPORT,  'EXP'),
    ]
    # Brush sizes (right vertical strip) — (grid-cell side length, label)
    _BRUSH_SIZES = [
        (1, '1'),
        (3, '3'),
        (5, '5'),
    ]

    def __init__(self, warehouse, warehouse_res, zone_view_rect, optimizer=None):
        """
        Args:
            warehouse:       Warehouse instance (accessed for zoneMap / pending_zone_changes).
            warehouse_res:   (gw, gh) grid cell dimensions.
            zone_view_rect:  (x0, y0, pw, ph) — pixel rect of the zone-map sub-view inside
                             the composite window.  Painting is restricted to this region.
            optimizer:       Optional WarehouseOptimizer instance (strategies are toggled via
                             the right-side brush strip).
        """
        self.warehouse      = warehouse
        self.warehouse_res  = warehouse_res   # grid dims (gw, gh)
        self.zone_view_rect = zone_view_rect  # (x0, y0, pw, ph) of zone-map panel
        self.optimizer      = optimizer       # WarehouseOptimizer or None
        self.paint_zone     = ZONE_IMPORT     # active zone; 0 = erase
        self.brush_size     = 1              # current brush side length (1, 3, or 5)
        self._last_left_cell  = None          # (gx, gy) of last L-button painted cell
        self._last_right_cell = None          # (gx, gy) of last R-button erased cell

    # ── Input ──────────────────────────────────────────────────────────

    def on_mouse(self, event, px, py, flags, param):
        """cv2 mouse callback — paints grid cells or selects zone / brush via
        the button strip (bottom) and brush strip (right) of the zone-map panel."""
        vx0, vy0, vw, vh = self.zone_view_rect
        map_w   = vw - self.BRUSH_W          # paintable map width  (px)
        map_h   = vh - self.BUTTON_H          # paintable map height (px)
        brush_x = vx0 + map_w                 # left edge of brush strip
        btn_y0  = vy0 + map_h                 # top edge of zone button strip

        # ── Right strip click (brush sizes top, strategy toggles bottom) ─
        n_strats  = len(self.optimizer.strategies) if self.optimizer else 0
        n_brushes = len(self._BRUSH_SIZES)
        n_slots   = n_brushes + n_strats
        slot_h    = max(map_h // max(n_slots, 1), 1)
        strat_y0  = vy0 + n_brushes * slot_h  # where strategy toggles start

        if (event == cv2.EVENT_LBUTTONDOWN
                and brush_x <= px < vx0 + vw
                and vy0 <= py < btn_y0):
            if py < strat_y0:
                # Brush size click
                idx = int((py - vy0) * n_brushes / max(n_brushes * slot_h, 1))
                self.brush_size = self._BRUSH_SIZES[max(0, min(idx, n_brushes - 1))][0]
            elif self.optimizer and n_strats > 0:
                # Strategy toggle click
                idx = int((py - strat_y0) * n_strats / max(map_h - n_brushes * slot_h, 1))
                idx = max(0, min(idx, n_strats - 1))
                s = self.optimizer.strategies[idx]
                s.enabled = not s.enabled
                print('[PaintHandler] {} optimizer {}.'.format(
                    s.name, 'ON' if s.enabled else 'OFF'))
            return

        # ── Zone button strip click (full-width bottom strip) ─────────
        if (event == cv2.EVENT_LBUTTONDOWN
                and vx0 <= px < vx0 + vw
                and btn_y0 <= py < vy0 + vh):
            n = len(self._BUTTONS)
            idx = int((px - vx0) * n / vw)
            self.paint_zone = self._BUTTONS[max(0, min(idx, n - 1))][0]
            return

        # ── Paint / erase in the map area ────────────────────────────
        is_paint = (event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_LBUTTON))
        is_erase = (event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_RBUTTON))

        # Reset stroke tracking when the button is released
        if event == cv2.EVENT_LBUTTONUP:
            self._last_left_cell = None
            return
        if event == cv2.EVENT_RBUTTONUP:
            self._last_right_cell = None
            return

        if not (is_paint or is_erase):
            return
        if not (vx0 <= px < brush_x and vy0 <= py < btn_y0):
            # Leaving the paint area resets the relevant button's last position
            if is_paint: self._last_left_cell  = None
            if is_erase: self._last_right_cell = None
            return
        gw, gh = self.warehouse_res
        gx = int((px - vx0) * gw / map_w)
        gy = int((py - vy0) * gh / map_h)
        if not (0 <= gx < gw and 0 <= gy < gh):
            return
        zone_id = self.paint_zone if is_paint else ZONE_NONE

        # Paint every cell along the line from the last position to the current one
        # (Bresenham) so fast mouse movement never leaves gaps.
        if is_paint:
            start = self._last_left_cell if self._last_left_cell is not None else (gx, gy)
            self._last_left_cell = (gx, gy)
        else:
            start = self._last_right_cell if self._last_right_cell is not None else (gx, gy)
            self._last_right_cell = (gx, gy)
        r = self.brush_size // 2
        for cx, cy in self._bresenham(start[0], start[1], gx, gy):
            for bx in range(cx - r, cx + r + 1):
                for by in range(cy - r, cy + r + 1):
                    if 0 <= bx < gw and 0 <= by < gh:
                        self.warehouse.zoneMap[by][bx] = zone_id
                        self.warehouse.pending_zone_changes.add((bx, by))

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Yield all grid cells on the line from (x0,y0) to (x1,y1) inclusive."""
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        while True:
            yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy

    def handle_key(self, key):
        """Handle utility keypresses not covered by the on-screen buttons.

        Returns True if the key was consumed, False otherwise.
        Key L — hot-reload all map PNGs from disk.
        """
        if key == ord('l'):
            from data.paint.map_importer import MapImporter
            gw, gh = self.warehouse_res
            # Zone map
            new_map = MapImporter.load("resources/zone_map.png", gw, gh)
            if new_map is not None:
                self.warehouse.zoneMap = new_map
                self.warehouse.update_zone_counts()
                # Mark every cell changed so reconcile_zone_changes handles in-flight plans
                self.warehouse.pending_zone_changes = {
                    (x, y) for y in range(gh) for x in range(gw)
                }
                print("[PaintHandler] Zone map hot-reloaded from 'resources/zone_map.png'.")
            # Charger spawn map
            _charger_coords = MapImporter.load_spawn_map("resources/charger_map.png", gw, gh)
            if _charger_coords:
                self.warehouse.chargerSpawnMap = _charger_coords
                print("[PaintHandler] Charger spawn map hot-reloaded ({} cells).".format(
                    len(_charger_coords)))
            # Robot spawn map
            _robot_coords = MapImporter.load_spawn_map("resources/robot_map.png", gw, gh)
            if _robot_coords:
                self.warehouse.robotSpawnMap = _robot_coords
                print("[PaintHandler] Robot spawn map hot-reloaded ({} cells).".format(
                    len(_robot_coords)))
            # Reset optimizer (recompute zonable mask from new spawn maps)
            if self.optimizer:
                self.optimizer.reset(self.warehouse)
                print("[PaintHandler] Optimizer strategies reset.")
            return True
        return False

    # ── Buttons ────────────────────────────────────────────────────────

    def draw_buttons(self, composite, font):
        """Draw controls around the zone-map panel.

        Layout (within the zone_view_rect panel):
          Right  BRUSH_W px (top):    vertical brush size selectors — 1 / 3 / 5
          Right  BRUSH_W px (bottom): strategy toggle buttons — e.g. Z
          Bottom BUTTON_H px (full):  zone / erase selectors  — CLR|IMP|STO|EXP

        Args:
            composite: composite BGR image (mutated in-place).
            font:      cv2 font constant (FONT_HERSHEY_SIMPLEX).
        """
        vx0, vy0, vw, vh = self.zone_view_rect
        btn_h   = self.BUTTON_H
        brush_w = self.BRUSH_W
        map_w   = vw - brush_w
        map_h   = vh - btn_h
        btn_y0  = vy0 + map_h             # top of zone button strip
        brush_x = vx0 + map_w             # left of brush strip

        # Slot layout for right strip: brush sizes top, strategy toggles bottom
        strategies = self.optimizer.strategies if self.optimizer else []
        n_b   = len(self._BRUSH_SIZES)
        n_s   = len(strategies)
        n_tot = n_b + n_s
        slot_h = max(map_h // max(n_tot, 1), 1)

        # ── Zone / erase buttons (bottom strip, full width) ──────────
        composite[btn_y0:vy0 + vh, vx0:vx0 + vw] = (18, 18, 18)
        composite[btn_y0, vx0:vx0 + vw]           = (48, 48, 48)    # top border

        n_z = len(self._BUTTONS)
        for i, (zone_id, label) in enumerate(self._BUTTONS):
            bx0 = vx0 + i * vw // n_z
            bx1 = vx0 + (i + 1) * vw // n_z
            active = (self.paint_zone == zone_id)
            if active:
                col = self.ZONE_COLS[zone_id]
                bg  = tuple(max(int(c * 0.22), 0) for c in col) if any(col) else (34, 34, 34)
                composite[btn_y0 + 2:vy0 + vh, bx0:bx1] = bg
                composite[btn_y0:btn_y0 + 2, bx0:bx1]   = col
                text_col = tuple(min(int(c * 1.4), 255) for c in col)
            else:
                text_col = (55, 55, 55)
            if i < n_z - 1:
                composite[btn_y0:vy0 + vh, bx1 - 1:bx1] = (36, 36, 36)
            (tw, th), _ = cv2.getTextSize(label, font, 0.22, 1)
            tx = bx0 + (bx1 - bx0 - tw) // 2
            ty = btn_y0 + (btn_h + th) // 2
            cv2.putText(composite, label, (tx, ty), font, 0.22, text_col, 1)

        # ── Right vertical strip background ──────────────────────────
        composite[vy0:btn_y0, brush_x:vx0 + vw] = (18, 18, 18)
        composite[vy0:btn_y0, brush_x:brush_x + 1] = (48, 48, 48)  # left border

        # ── Brush size buttons (top of right strip) ──────────────────
        for i, (size, label) in enumerate(self._BRUSH_SIZES):
            by0 = vy0 + i * slot_h
            by1 = vy0 + (i + 1) * slot_h
            active = (self.brush_size == size)
            if active:
                composite[by0:by1, brush_x + 2:vx0 + vw] = (36, 36, 44)
                composite[by0:by1, brush_x:brush_x + 2]   = (130, 130, 170)
                text_col = (190, 190, 215)
            else:
                text_col = (55, 55, 55)
            if i < n_b - 1:
                composite[by1 - 1:by1, brush_x:vx0 + vw] = (36, 36, 36)
            (tw, th), _ = cv2.getTextSize(label, font, 0.22, 1)
            tx = brush_x + (brush_w - tw) // 2
            ty = by0 + (by1 - by0 + th) // 2
            cv2.putText(composite, label, (tx, ty), font, 0.22, text_col, 1)

        # ── Strategy toggle buttons (bottom of right strip) ──────────
        if n_s > 0:
            strat_y0 = vy0 + n_b * slot_h
            # Separator between brush and strategy sections
            composite[strat_y0 - 1:strat_y0, brush_x:vx0 + vw] = (48, 48, 48)
            strat_h = map_h - n_b * slot_h
            for i, s in enumerate(strategies):
                by0 = strat_y0 + i * strat_h // n_s
                by1 = strat_y0 + (i + 1) * strat_h // n_s
                if s.enabled:
                    composite[by0:by1, brush_x + 2:vx0 + vw] = (24, 40, 24)
                    composite[by0:by1, brush_x:brush_x + 2]   = (60, 200, 60)
                    text_col = (80, 220, 80)
                else:
                    text_col = (55, 55, 55)
                if i < n_s - 1:
                    composite[by1 - 1:by1, brush_x:vx0 + vw] = (36, 36, 36)
                lbl = s.label
                (tw, th), _ = cv2.getTextSize(lbl, font, 0.22, 1)
                tx = brush_x + (brush_w - tw) // 2
                ty = by0 + (by1 - by0 + th) // 2
                cv2.putText(composite, lbl, (tx, ty), font, 0.22, text_col, 1)
