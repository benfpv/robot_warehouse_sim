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

    def __init__(self, warehouse, warehouse_res, warehouse_window_res):
        """
        Args:
            warehouse:            Warehouse instance (accessed for zoneMap / pending_zone_changes).
            warehouse_res:        (gw, gh) grid cell dimensions.
            warehouse_window_res: (pw, ph) pixel dimensions of the main view in the composite window.
        """
        self.warehouse            = warehouse
        self.warehouse_res        = warehouse_res        # grid dims (gw, gh)
        self.warehouse_window_res = warehouse_window_res # pixel dims of main view (pw, ph)
        self.paint_zone           = ZONE_IMPORT          # active zone; 0 = erase

    # ── Input ──────────────────────────────────────────────────────────

    def on_mouse(self, event, px, py, flags, param):
        """cv2 mouse callback — translates pixel coords to a grid cell and records the paint."""
        is_paint = (event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_LBUTTON))
        is_erase = (event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE)
                    and bool(flags & cv2.EVENT_FLAG_RBUTTON))
        if not (is_paint or is_erase):
            return
        pw, ph = self.warehouse_window_res
        if not (0 <= px < pw and 0 <= py < ph):
            return
        gw, gh = self.warehouse_res
        gx = int(px * gw / pw)
        gy = int(py * gh / ph)
        if not (0 <= gx < gw and 0 <= gy < gh):
            return
        zone_id = self.paint_zone if is_paint else ZONE_NONE
        self.warehouse.zoneMap[gy][gx] = zone_id
        self.warehouse.pending_zone_changes.add((gx, gy))

    def handle_key(self, key):
        """Update active paint zone from a keypress.

        Returns True if the key was consumed, False otherwise.
        """
        if   key == ord('1'): self.paint_zone = ZONE_IMPORT;  return True
        elif key == ord('2'): self.paint_zone = ZONE_STORAGE; return True
        elif key == ord('3'): self.paint_zone = ZONE_EXPORT;  return True
        elif key == ord('0'): self.paint_zone = ZONE_NONE;    return True
        return False

    # ── HUD ────────────────────────────────────────────────────────────

    def draw_hud(self, composite, mw, mh, font):
        """Draw the PAINT: <ZONE> indicator in the bottom-right corner of the main view.

        Args:
            composite: composite BGR image (mutated in-place).
            mw, mh:    pixel width/height of the main view region.
            font:      cv2 font constant.
        """
        lbl     = "PAINT: {}".format(self.ZONE_NAMES.get(self.paint_zone, '?'))
        col_raw = self.ZONE_COLS.get(self.paint_zone, (180, 180, 180))
        col     = tuple(min(int(c * 1.5), 255) for c in col_raw)
        (tw, th), _ = cv2.getTextSize(lbl, font, 0.30, 1)
        bx0, bx1 = mw - tw - 7, mw - 1
        by0, by1 = mh - th - 5, mh - 1
        roi = composite[by0:by1, bx0:bx1]
        composite[by0:by1, bx0:bx1] = (roi * 0.3).astype(np.uint8)
        cv2.putText(composite, lbl, (mw - tw - 4, mh - 4), font, 0.30, col, 1)
