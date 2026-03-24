import cv2
import numpy as np

# Zone constants (must match warehouse.py)
ZONE_NONE    = 0
ZONE_IMPORT  = 1
ZONE_STORAGE = 2
ZONE_EXPORT  = 3


class PaintHandler:
    """Handles user zone-painting input and the associated HUD overlay.

    Painting mutates warehouse.zoneMap and populates warehouse.pending_zone_changes
    for reconciliation on the next sim tick.

    Controls:
        Left-drag  — paint active zone
        Right-drag — erase (zone 0 / neutral)
        Keys 1/2/3 — select Import / Storage / Export
        Key  0     — select Erase mode
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

    def __init__(self, warehouse, warehouse_res, zone_view_rect):
        """
        Args:
            warehouse:       Warehouse instance (accessed for zoneMap / pending_zone_changes).
            warehouse_res:   (gw, gh) grid cell dimensions.
            zone_view_rect:  (x0, y0, pw, ph) — pixel rect of the zone-map sub-view inside
                             the composite window.  Painting is restricted to this region.
        """
        self.warehouse      = warehouse
        self.warehouse_res  = warehouse_res   # grid dims (gw, gh)
        self.zone_view_rect = zone_view_rect  # (x0, y0, pw, ph) of zone-map panel
        self.paint_zone     = ZONE_IMPORT     # active zone; 0 = erase

    # ── Input ──────────────────────────────────────────────────────────

    def on_mouse(self, event, px, py, flags, param):
        """cv2 mouse callback — paints grid cells when the cursor is over the zone-map panel."""
        is_paint = (event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_LBUTTON))
        is_erase = (event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_RBUTTON))
        if not (is_paint or is_erase):
            return
        vx0, vy0, vw, vh = self.zone_view_rect
        # Only act when the cursor is inside the zone-map panel
        if not (vx0 <= px < vx0 + vw and vy0 <= py < vy0 + vh):
            return
        gw, gh = self.warehouse_res
        gx = int((px - vx0) * gw / vw)
        gy = int((py - vy0) * gh / vh)
        if not (0 <= gx < gw and 0 <= gy < gh):
            return
        zone_id = self.paint_zone if is_paint else ZONE_NONE
        self.warehouse.zoneMap[gy][gx] = zone_id
        self.warehouse.pending_zone_changes.add((gx, gy))

    def handle_key(self, key):
        """Update active paint zone from a keypress.

        Returns True if the key was consumed, False otherwise.
        Keys: 1=Import  2=Storage  3=Export  0=Erase  L=reload zone_map.png
        """
        if   key == ord('1'): self.paint_zone = ZONE_IMPORT;  return True
        elif key == ord('2'): self.paint_zone = ZONE_STORAGE; return True
        elif key == ord('3'): self.paint_zone = ZONE_EXPORT;  return True
        elif key == ord('0'): self.paint_zone = ZONE_NONE;    return True
        elif key == ord('l'):
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
            return True
        return False

    # ── HUD ────────────────────────────────────────────────────────────

    def draw_hud(self, composite, font):
        """Draw the PAINT: <ZONE> indicator in the bottom-right corner of the zone-map panel.

        Args:
            composite: composite BGR image (mutated in-place).
            font:      cv2 font constant.
        """
        vx0, vy0, vw, vh = self.zone_view_rect
        lbl     = "PAINT: {}".format(self.ZONE_NAMES.get(self.paint_zone, '?'))
        col_raw = self.ZONE_COLS.get(self.paint_zone, (180, 180, 180))
        col     = tuple(min(int(c * 1.5), 255) for c in col_raw)
        (tw, th), _ = cv2.getTextSize(lbl, font, 0.30, 1)
        bx0 = vx0 + vw - tw - 7
        bx1 = vx0 + vw - 1
        by0 = vy0 + vh - th - 5
        by1 = vy0 + vh - 1
        roi = composite[by0:by1, bx0:bx1]
        composite[by0:by1, bx0:bx1] = (roi * 0.3).astype(np.uint8)
        cv2.putText(composite, lbl, (vx0 + vw - tw - 4, vy0 + vh - 4), font, 0.30, col, 1)
