"""map_importer.py — load a user-drawn zone-map PNG into the simulation's zoneMap array.

Image format
────────────
Draw zones using broad colour families — any shade works.  Classification is
hue-based (HSV) so imprecise shades, gradients, and anti-aliasing are handled
gracefully.  Use any image editor: MS Paint, GIMP, Photoshop, etc.

    Zone       ID   Colour family     Examples that all work
    ────────   ──   ───────────────   ─────────────────────────────────────────
    Neutral     0   black / white /   background, light grey, white, black
                    grey / leave      (anything unsaturated or unlisted)
    Import      1   GREEN             lime, dark green, olive, forest green
    Storage     2   BLUE              pure blue, navy, sky blue, teal, cyan
    Export      3   RED               pure red, crimson, dark red, orange-red

Image resolution is arbitrary — it is rescaled with nearest-neighbour
interpolation so painted edges stay sharp.

Quick-start
───────────
1. Open  resources/zone_map_example.png  in any image editor.
2. Flood-fill / paint zones with any shade of green / blue / red.
3. Leave unpainted areas black (or any unsaturated colour) for neutral.
4. Save as  zone_map.png  in the project root.
5. Restart the sim (or press  L  while running) to apply.
"""

import os
import cv2
import numpy as np

# Zone constants (must match warehouse.py)
ZONE_NONE    = 0
ZONE_IMPORT  = 1
ZONE_STORAGE = 2
ZONE_EXPORT  = 3


class MapImporter:
    # BGR colours used when *rendering* zones in the example PNG
    # (pure saturated — easily reproduced in MS Paint)
    _ZONE_BGR = {
        ZONE_NONE:    (  20,  20,  20),   # near-black background
        ZONE_IMPORT:  (   0, 200,   0),   # pure green
        ZONE_STORAGE: ( 220,   0,   0),   # pure blue  (BGR — R and B swapped)
        ZONE_EXPORT:  (   0,   0, 220),   # pure red   (BGR)
    }

    # ── HSV hue-range classification ─────────────────────────────────
    # OpenCV HSV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
    #
    # Minimum saturation / value to be treated as a coloured pixel at all.
    # Anything below these → neutral (catches black, white, grey).
    _S_MIN = 60   # saturation floor  (0-255)
    _V_MIN = 40   # value (brightness) floor

    # Hue windows for each zone (inclusive, OpenCV 0-179 scale)
    # Red wraps around 0, so we store it as two sub-ranges.
    _HUE_GREEN_LO,  _HUE_GREEN_HI  = 35, 85    # green / lime / olive / teal-green
    _HUE_BLUE_LO,   _HUE_BLUE_HI   = 90, 140   # blue / navy / sky-blue / cyan / teal
    _HUE_RED_LO1,   _HUE_RED_HI1   = 0,  12    # red (low end)
    _HUE_RED_LO2,   _HUE_RED_HI2   = 158, 179  # red (wrap-around end: magenta-red)

    # ── Load ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path, gw, gh):
        """Load *path* PNG and return a uint8 (gh × gw) zoneMap, or None on failure.

        Classification is hue-based so any shade of the right colour family
        maps to the correct zone regardless of brightness or exact RGB values.
        """
        img = cv2.imread(path)
        if img is None:
            print("[MapImporter] Could not read: {}".format(path))
            return None

        # Scale to grid with nearest-neighbour to preserve painted zone edges
        img = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.int32)   # hue     0-179
        S = hsv[:, :, 1]                     # sat     0-255
        V = hsv[:, :, 2]                     # value   0-255

        coloured = (S >= cls._S_MIN) & (V >= cls._V_MIN)

        zone_map = np.zeros((gh, gw), dtype='uint8')   # default neutral

        green_mask = coloured & (H >= cls._HUE_GREEN_LO) & (H <= cls._HUE_GREEN_HI)
        blue_mask  = coloured & (H >= cls._HUE_BLUE_LO)  & (H <= cls._HUE_BLUE_HI)
        red_mask   = coloured & ((H <= cls._HUE_RED_HI1) | (H >= cls._HUE_RED_LO2))

        zone_map[green_mask] = ZONE_IMPORT
        zone_map[blue_mask]  = ZONE_STORAGE
        zone_map[red_mask]   = ZONE_EXPORT   # applied last — red wins on overlap edges

        counts = {v: int(np.count_nonzero(zone_map == v)) for v in [1, 2, 3]}
        print("[MapImporter] Loaded '{}' → {}×{} grid, slots: {}".format(path, gw, gh, counts))
        return zone_map

    # ── Example PNG generator ──────────────────────────────────────────

    @classmethod
    def generate_example_png(cls, path, gw=80, gh=70, cell_px=8):
        """Write an annotated example zone-map PNG to *path*.

        Renders the default rectangular zone layout with a colour-coded legend
        and instructions.  Open it in any image editor, flood-fill or paint
        over the zones with any shade of the indicated colour family, then
        save as resources/zone_map.png in the project root.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Default zone geometry (mirrors Warehouse_Init.init_zoneMap) ──
        pad     = 3
        scale   = 0.775
        inner_w = gw - pad * 2
        inner_h = gh - pad * 2
        zone_w  = int(inner_w * scale)
        zone_h  = int((inner_h // 3) * scale)
        x0      = pad + (inner_w - zone_w) // 2
        y_base  = pad + (inner_h - zone_h * 3) // 2
        zone_map = np.zeros((gh, gw), dtype='uint8')
        zone_map[y_base            : y_base + zone_h,      x0:x0 + zone_w] = ZONE_IMPORT
        zone_map[y_base + zone_h   : y_base + zone_h * 2,  x0:x0 + zone_w] = ZONE_STORAGE
        zone_map[y_base + zone_h*2 : y_base + zone_h * 3,  x0:x0 + zone_w] = ZONE_EXPORT

        # Canvas size
        cw, ch   = gw * cell_px, gh * cell_px   # 640 × 560 at default cell_px=8
        legend_h = 80                             # taller legend strip

        # ── Zone colours: pure saturated — easy to pick in MS Paint ──────
        # These match the hue ranges in load() — any shade of the family works.
        paint_col_bgr = {
            ZONE_NONE:    (  0,   0,   0),   # black   → neutral
            ZONE_IMPORT:  (  0, 180,   0),   # green   → import
            ZONE_STORAGE: (200,   0,   0),   # blue    → storage  (BGR)
            ZONE_EXPORT:  (  0,   0, 200),   # red     → export   (BGR)
        }

        # Build zone map image and upscale
        zone_img = np.zeros((gh, gw, 3), dtype='uint8')
        for zid, col in paint_col_bgr.items():
            zone_img[zone_map == zid] = col
        canvas = cv2.resize(zone_img, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # Full image (canvas + legend)
        img = np.full((ch + legend_h, cw, 3), 18, dtype='uint8')
        img[:ch, :cw] = canvas

        # ── Subtle 10-cell grid guide lines ───────────────────────────────
        lc = (38, 38, 38)
        for i in range(0, gw + 1, 10):
            cv2.line(img, (i * cell_px, 0), (i * cell_px, ch - 1), lc, 1)
        for i in range(0, gh + 1, 10):
            cv2.line(img, (0, i * cell_px), (cw - 1, i * cell_px), lc, 1)

        # ── Zone border + large centred label + "use ANY shade" sub-label ─
        zones_draw = [
            (ZONE_IMPORT,  y_base,              zone_h, "IMPORT",  "any shade of GREEN"),
            (ZONE_STORAGE, y_base + zone_h,     zone_h, "STORAGE", "any shade of BLUE"),
            (ZONE_EXPORT,  y_base + zone_h * 2, zone_h, "EXPORT",  "any shade of RED"),
        ]
        for zid, gy_start, zh, name, hint in zones_draw:
            col   = paint_col_bgr[zid]
            white = (230, 230, 230)
            dim   = (120, 120, 120)
            bx0 = x0 * cell_px
            bx1 = (x0 + zone_w) * cell_px - 1
            by0 = gy_start * cell_px
            by1 = (gy_start + zh) * cell_px - 1
            # Bright outline
            cv2.rectangle(img, (bx0, by0), (bx1, by1), col, 3)
            # Main zone label
            (tw, th), _ = cv2.getTextSize(name, font, 0.65, 2)
            lx = bx0 + (bx1 - bx0 - tw) // 2
            ly = by0 + (by1 - by0) // 2
            cv2.putText(img, name, (lx + 1, ly + 1), font, 0.65, (0, 0, 0), 3)
            cv2.putText(img, name, (lx,     ly),     font, 0.65, white,      2)
            # Sub-label
            (sw2, sh2), _ = cv2.getTextSize(hint, font, 0.30, 1)
            sx = bx0 + (bx1 - bx0 - sw2) // 2
            sy = ly + th + 4
            cv2.putText(img, hint, (sx, sy), font, 0.30, dim, 1)

        # "NEUTRAL" hint in background area
        cv2.putText(img, "NEUTRAL = leave black (or any grey/white/unsaturated colour)",
                    (6, 14), font, 0.28, (55, 55, 55), 1)

        # ── Legend strip ──────────────────────────────────────────────────
        # Divider
        img[ch:ch + 1, :] = 50

        lg_items = [
            (ZONE_NONE,    "Neutral",  "leave black / unpainted",  (90, 90, 90)),
            (ZONE_IMPORT,  "Import",   "any GREEN shade",           (0, 200, 0)),
            (ZONE_STORAGE, "Storage",  "any BLUE shade",            (210, 60, 0)),   # BGR
            (ZONE_EXPORT,  "Export",   "any RED shade",             (0, 0, 210)),    # BGR
        ]
        slot_w  = cw // len(lg_items)
        sw_size = 20   # swatch size in px

        for i, (zid, name, desc, bcol) in enumerate(lg_items):
            lx   = i * slot_w + 8
            ty_s = ch + 12               # swatch top
            ty_n = ch + 12 + sw_size + 10  # name baseline
            ty_d = ty_n + 16               # desc baseline

            # Filled colour swatch
            cv2.rectangle(img, (lx, ty_s), (lx + sw_size * 3, ty_s + sw_size), bcol, -1)
            cv2.rectangle(img, (lx, ty_s), (lx + sw_size * 3, ty_s + sw_size), (160, 160, 160), 1)

            bright = tuple(min(255, c + 40) for c in bcol) if zid != ZONE_NONE else (140, 140, 140)
            cv2.putText(img, name, (lx, ty_n), font, 0.38, bright, 1)
            cv2.putText(img, desc, (lx, ty_d), font, 0.27, (100, 100, 100), 1)

        # Bottom instruction line
        instr = "Paint zones then save as resources/zone_map.png | press L in sim to hot-reload"
        (iw, _), _ = cv2.getTextSize(instr, font, 0.25, 1)
        cv2.putText(img, instr, ((cw - iw) // 2, ch + legend_h - 4),
                    font, 0.25, (60, 60, 60), 1)

        # ── Write ─────────────────────────────────────────────────────────
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print("[MapImporter] Example PNG written: {}  ({}×{} px)".format(
            path, cw, ch + legend_h))

    # ── Default zone_map.png generator ────────────────────────────────

    @classmethod
    def generate_default_map(cls, path, gw=80, gh=70, cell_px=8):
        """Write a clean, annotation-free zone_map.png at *path*.

        Produces the default three-band layout as a plain coloured image with
        no text, borders, or grid lines — exactly what the HSV classifier
        expects.  Save to  resources/zone_map.png  and it will be auto-loaded
        on startup (or press L in-sim to hot-reload).
        """
        pad     = 3
        scale   = 0.775
        inner_w = gw - pad * 2
        inner_h = gh - pad * 2
        zone_w  = int(inner_w * scale)
        zone_h  = int((inner_h // 3) * scale)
        x0      = pad + (inner_w - zone_w) // 2
        y_base  = pad + (inner_h - zone_h * 3) // 2

        zone_map = np.zeros((gh, gw), dtype='uint8')
        zone_map[y_base            : y_base + zone_h,      x0:x0 + zone_w] = ZONE_IMPORT
        zone_map[y_base + zone_h   : y_base + zone_h * 2,  x0:x0 + zone_w] = ZONE_STORAGE
        zone_map[y_base + zone_h*2 : y_base + zone_h * 3,  x0:x0 + zone_w] = ZONE_EXPORT

        # Pixel colour for each zone (pure saturated — well within HSV thresholds)
        zone_img = np.zeros((gh, gw, 3), dtype='uint8')
        for zid, col in {
            ZONE_NONE:    (  0,   0,   0),
            ZONE_IMPORT:  (  0, 180,   0),   # pure green
            ZONE_STORAGE: (220,   0,   0),   # pure blue  (BGR)
            ZONE_EXPORT:  (  0,   0, 220),   # pure red   (BGR)
        }.items():
            zone_img[zone_map == zid] = col

        img = cv2.resize(zone_img, (gw * cell_px, gh * cell_px),
                         interpolation=cv2.INTER_NEAREST)

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        cw2, ch2 = gw * cell_px, gh * cell_px
        print("[MapImporter] Default map written: {}  ({}×{} px)".format(path, cw2, ch2))

    # ── Spiral / black-hole zone map generator ─────────────────────────

    @classmethod
    def generate_spiral_map(cls, path, gw=80, gh=70, cell_px=8,
                            n_turns=2.5, core_frac=0.08, outer_frac=0.88):
        """Write a spiral zone-map PNG to *path*.

        Zones cycle Import→Storage→Export spiralling outward from the centre,
        creating a black-hole / galaxy-arm effect.  The very centre and outer
        border remain neutral so robots can navigate around the edge.

        Args:
            n_turns:    how many full spiral rotations across the radius.
            core_frac:  fraction of max_r treated as neutral black-hole core.
            outer_frac: fraction of max_r beyond which cells revert to neutral.
        """
        # Build coordinate grids (float)
        ys, xs = np.mgrid[0:gh, 0:gw].astype(float)
        cx, cy = (gw - 1) / 2.0, (gh - 1) / 2.0
        dx, dy = xs - cx, ys - cy

        # Polar coords
        r     = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)          # -π .. π

        max_r = min(cx, cy)                  # largest usable radius

        # Normalise radius to [0, 1] and theta to [0, 1)
        r_norm     = r / max_r
        theta_norm = (theta + np.pi) / (2 * np.pi)   # 0 .. 1

        # Spiral phase: advances n_turns full cycles as r goes 0→1,
        # plus one extra cycle contributed by the angle.
        phase = (r_norm * n_turns + theta_norm) % 3.0

        # Assign zones based on phase thirds
        zone_map = np.zeros((gh, gw), dtype='uint8')
        zone_map[(phase >= 0) & (phase < 1)] = ZONE_IMPORT
        zone_map[(phase >= 1) & (phase < 2)] = ZONE_STORAGE
        zone_map[(phase >= 2) & (phase < 3)] = ZONE_EXPORT

        # Hollow out: core (black-hole centre) and outer border → neutral
        zone_map[r_norm <  core_frac]  = ZONE_NONE
        zone_map[r_norm >= outer_frac] = ZONE_NONE

        # Also clear a 2-cell perimeter so robots can always navigate the edge
        zone_map[:2, :]  = ZONE_NONE
        zone_map[-2:, :] = ZONE_NONE
        zone_map[:, :2]  = ZONE_NONE
        zone_map[:, -2:] = ZONE_NONE

        # Render to BGR image
        zone_img = np.zeros((gh, gw, 3), dtype='uint8')
        for zid, col in {
            ZONE_NONE:    (  0,   0,   0),
            ZONE_IMPORT:  (  0, 180,   0),   # green
            ZONE_STORAGE: (220,   0,   0),   # blue  (BGR)
            ZONE_EXPORT:  (  0,   0, 220),   # red   (BGR)
        }.items():
            zone_img[zone_map == zid] = col

        img = cv2.resize(zone_img, (gw * cell_px, gh * cell_px),
                         interpolation=cv2.INTER_NEAREST)

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        counts = {v: int(np.count_nonzero(zone_map == v)) for v in [1, 2, 3]}
        cw2, ch2 = gw * cell_px, gh * cell_px
        print("[MapImporter] Spiral map written: {}  ({}×{} px), slots: {}".format(
            path, cw2, ch2, counts))
