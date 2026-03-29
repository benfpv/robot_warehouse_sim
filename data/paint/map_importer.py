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
        print("[MapImporter] Loaded '{}' -> {}x{} grid, slots: {}".format(path, gw, gh, counts))
        return zone_map

    # ── Example PNG generator ──────────────────────────────────────────

    @classmethod
    def generate_example_png(cls, path, gw=80, gh=70, cell_px=8):
        """Write an annotated example zone-map PNG to *path*.

        Renders the default three-band zone layout with a colour-coded legend
        and instructions.  Open in any image editor, paint over the zones with
        any shade of the indicated colour family, then save as
        resources/zone_map.png in the project root.
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
                    font, 0.25, (85, 85, 85), 1)

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

    # ── Sunflower zone map generator ───────────────────────────────────

    @classmethod
    def generate_sunflower_map(cls, path, gw=80, gh=70, cell_px=8,
                               outer_frac=0.90):
        """Write a phyllotaxis / sunflower zone-map PNG to *path*.

        Seed positions follow Vogel's golden-angle formula with n=377 seeds
        (a Fibonacci number), which encodes 21 CW and 34 CCW visible spiral
        arms — the two consecutive Fibonacci numbers for this seed count.
        Each grid cell is assigned to its nearest seed (Voronoi), so every
        seed becomes one distinct rhombic section.

        Zone boundaries follow Fibonacci spiral arms via arm modulation:
          arm_mod = sin(2π·i%21/21) × sin(2π·i%34/34)
        This peaks at the centre of each rhombus and dips at arm crossings,
        causing zone boundaries to zigzag along the spiral arms.

        Zone proportions are enforced by assigning seeds in ascending gravity
        order, accumulating *grid cell counts* (not seed counts) until the
        target area fraction is met.  This corrects for inner seeds having
        larger Voronoi cells than outer seeds.

        Gravity ordering:
            Import  — outer ring / top    (≈20 % of active disk area)
            Storage — middle annular band (≈31 %)
            Export  — inner core / bottom (≈49 %)
        """
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))   # ≈ 137.508°
        n_seeds      = 377       # Fibonacci → 21-CW + 34-CCW visible spiral arms
        F_cw, F_ccw  = 21, 34

        cx, cy = (gw - 1) / 2.0, (gh - 1) / 2.0
        max_r  = min(cx, cy)

        # ── Vogel seed positions ──────────────────────────────────────────
        idx    = np.arange(0, n_seeds, dtype=float)
        r_norm = np.sqrt(idx / max(n_seeds - 1, 1))         # 0 .. 1
        theta  = idx * golden_angle
        seed_x = cx + r_norm * outer_frac * max_r * np.cos(theta)
        seed_y = cy + r_norm * outer_frac * max_r * np.sin(theta)

        # ── Voronoi assignment ────────────────────────────────────────────
        ys, xs  = np.mgrid[0:gh, 0:gw].astype(float)
        flat_x  = xs.ravel()
        flat_y  = ys.ravel()
        dx = flat_x[:, np.newaxis] - seed_x[np.newaxis, :]  # (gh*gw, n_seeds)
        dy = flat_y[:, np.newaxis] - seed_y[np.newaxis, :]
        nearest  = np.argmin(dx**2 + dy**2, axis=1).reshape(gh, gw)

        # ── In-disk mask (for area calculations) ─────────────────────────
        r_grid   = np.sqrt((xs - cx)**2 + (ys - cy)**2) / max_r
        in_disk  = r_grid < outer_frac * 1.01

        # ── Per-seed gravity: mean gravity of their in-disk cells ─────────
        # Using grid-cell-level gravity weights area correctly — inner seeds
        # naturally contribute more cells and thus more weight, matching the
        # proportion of grid cells each zone will receive.
        y_grid        = ys / (gh - 1)                       # 0=top, 1=bottom
        cell_gravity  = (1.0 - r_grid) * 0.4 + y_grid * 0.6

        flat_nearest  = nearest.ravel()
        flat_gravity  = cell_gravity.ravel()
        flat_in_disk  = in_disk.ravel()
        valid_idx     = flat_nearest[flat_in_disk]
        valid_grav    = flat_gravity[flat_in_disk]

        grav_sum   = np.bincount(valid_idx, weights=valid_grav, minlength=n_seeds)
        cell_count = np.bincount(valid_idx, minlength=n_seeds)
        seed_grav  = np.where(cell_count > 0, grav_sum / (cell_count + 1e-9), 0.0)

        # ── Fibonacci arm modulation ──────────────────────────────────────
        # sin(2π·i%F_cw/F_cw) × sin(2π·i%F_ccw/F_ccw) peaks at the centre
        # of each rhombic "seed diamond", causing zone edges to follow
        # the 21/34 Fibonacci spiral arms rather than a smooth oval.
        arm_cw  = np.sin(2.0 * np.pi * (idx % F_cw)  / F_cw)
        arm_ccw = np.sin(2.0 * np.pi * (idx % F_ccw) / F_ccw)
        score   = seed_grav + 0.08 * (arm_cw * arm_ccw)    # ∈ rough [0, 1]

        # ── Zone assignment by cumulative grid-cell area ──────────────────
        # Sort seeds ascending (low score = outer/top = Import first).
        # Walk the sorted list, accumulating their cell counts, and flip the
        # zone label when the running total crosses each target threshold.
        # This guarantees the desired grid-cell proportions regardless of
        # how cell sizes vary across radii.
        total_cells    = int(flat_in_disk.sum())
        import_ceiling = int(total_cells * 0.20)
        storage_ceil   = import_ceiling + int(total_cells * 0.31)

        seed_zone  = np.full(n_seeds, ZONE_EXPORT, dtype='uint8')
        cumcells   = 0
        for si in np.argsort(score):
            nc = cell_count[si]
            if cumcells < import_ceiling:
                seed_zone[si] = ZONE_IMPORT
            elif cumcells < storage_ceil:
                seed_zone[si] = ZONE_STORAGE
            # else: remains ZONE_EXPORT (default)
            cumcells += nc

        # ── Build zone_map, mask outer fringe and perimeter ──────────────
        zone_map = seed_zone[nearest]
        zone_map[~in_disk] = ZONE_NONE
        zone_map[:2, :]  = ZONE_NONE
        zone_map[-2:, :] = ZONE_NONE
        zone_map[:, :2]  = ZONE_NONE
        zone_map[:, -2:] = ZONE_NONE

        # ── Render ────────────────────────────────────────────────────────
        zone_img = np.zeros((gh, gw, 3), dtype='uint8')
        for zid, col in {
            ZONE_NONE:    (  0,   0,   0),
            ZONE_IMPORT:  (  0, 180,   0),
            ZONE_STORAGE: (220,   0,   0),
            ZONE_EXPORT:  (  0,   0, 220),
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
        print("[MapImporter] Sunflower map written: {}  ({}×{} px), slots: {}".format(
            path, cw2, ch2, counts))

    # ── Spiral / black-hole zone map generator ─────────────────────────

    @classmethod
    def generate_spiral_map(cls, path, gw=80, gh=70, cell_px=8,
                            n_turns=3.0, core_frac=0.11, outer_frac=0.90):
        """Write a gravity-optimized spiral zone-map PNG to *path*.

        Flow mirrors gravity pulling packages inward and downward:

            Import  — outer ring / top   (packages arrive from outside/above)
            Storage — middle spiral bands (transit layer)
            Export  — inner core / bottom (gravity well — packages settle here)

        The spiral rotates clockwise so arms sweep downward.  A vertical
        gravity bias shifts the zone boundaries progressively toward export
        at the bottom of the warehouse, reinforcing the gravitational pull.
        """
        ys, xs = np.mgrid[0:gh, 0:gw].astype(float)
        cx, cy = (gw - 1) / 2.0, (gh - 1) / 2.0
        dx, dy = xs - cx, ys - cy

        r     = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)          # -π .. π

        max_r = min(cx, cy)

        r_norm = r / max_r

        # Clockwise angle: negate so arms rotate CW (gravity spiral inward)
        theta_cw = ((-theta) % (2 * np.pi)) / (2 * np.pi)   # 0 .. 1, clockwise

        # Vertical gravity bias: 0 at top → gravity_strength at bottom.
        # Nudges the phase downward so the bottom of the warehouse naturally
        # settles into the export zone without needing to be near the centre.
        gravity_strength = 0.85
        gravity_bias = (ys / (gh - 1)) * gravity_strength

        # Phase encoding (mod 3):
        #   0 .. 1  → Export  (low phase = inner / bottom = gravity sink)
        #   1 .. 2  → Storage (mid)
        #   2 .. 3  → Import  (high phase = outer / top = arrival zone)
        phase = (r_norm * n_turns + theta_cw + gravity_bias) % 3.0

        zone_map = np.zeros((gh, gw), dtype='uint8')
        zone_map[(phase >= 0) & (phase < 1)] = ZONE_EXPORT
        zone_map[(phase >= 1) & (phase < 2)] = ZONE_STORAGE
        zone_map[(phase >= 2) & (phase < 3)] = ZONE_IMPORT

        # Hollow core (the singularity) and outer fringe → neutral
        zone_map[r_norm <  core_frac]  = ZONE_NONE
        zone_map[r_norm >= outer_frac] = ZONE_NONE

        # Hard 2-cell perimeter so robots can always navigate the edge
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

    # ─────────────────────────────────────────────────────────────────────
    # Spawn-position maps (charger / robot)
    # ─────────────────────────────────────────────────────────────────────
    #
    # These maps define WHERE chargers/robots are allowed to spawn.
    # Any painted pixel = valid spawn cell; unpainted = forbidden.
    #
    # Colour conventions (load_spawn_map accepts any saturated pixel):
    #   Charger map  →  YELLOW   (BGR: 0, 220, 220)   key colour family: yellow
    #   Robot   map  →  CYAN     (BGR: 220, 220, 0)    key colour family: cyan / teal
    #   Neutral / forbidden  →  black / grey / white
    #
    # Both maps are optional.  If a file is absent the sim falls back to
    # the default perimeter coordinate list computed from the grid boundary.

    # HSV saturation/value floors (same as zone map)
    _SPAWN_S_MIN = 60
    _SPAWN_V_MIN = 40

    @classmethod
    def load_spawn_map(cls, path, gw, gh):
        """Load a spawn-map PNG and return a list of [x, y] valid-spawn coordinates.

        Any pixel that is *not* near-black/grey/white (i.e. has enough colour)
        is treated as a valid spawn cell, regardless of the exact hue.  This is
        simpler than zone maps because there is only one "on" state.

        Returns a list of [x, y] pairs (may be empty), or None if the file
        could not be read.  An empty list means the map was read successfully
        but had no painted cells — the caller should then fall back to default
        perimeter coords.
        """
        img = cv2.imread(path)
        if img is None:
            print("[MapImporter] Could not read spawn map: {}".format(path))
            return None

        img = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        valid_mask = (S >= cls._SPAWN_S_MIN) & (V >= cls._SPAWN_V_MIN)
        ys, xs = np.where(valid_mask)
        coords = [[int(x), int(y)] for x, y in zip(xs, ys)]

        print("[MapImporter] Spawn map '{}' -> {} valid cells".format(path, len(coords)))
        return coords

    @classmethod
    def generate_charger_map_example(cls, path, gw=80, gh=70, cell_px=8):
        """Write an annotated example charger-spawn-map PNG to *path*.

        The default pattern places chargers on the inner perimeter (1 cell in
        from the edge), which mirrors the default perimeter-minus-one logic.
        Paint yellow over any cells where chargers should be allowed to spawn.
        """
        canvas_w, canvas_h = gw * cell_px, gh * cell_px
        legend_h = 60
        img = np.full((canvas_h + legend_h, canvas_w, 3), 18, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Default inner-perimeter band (1 cell inset from all four edges)
        # Mirrors Warehouse_Init.init_warehousePerimeter(..., inset=1)
        yellow_bgr = (0, 220, 220)    # BGR: yellow
        for x in range(1, gw - 1):
            for y in [1, gh - 2]:
                px0, px1 = x * cell_px, (x + 1) * cell_px
                py0, py1 = y * cell_px, (y + 1) * cell_px
                img[py0:py1, px0:px1] = yellow_bgr
        for y in range(1, gh - 1):
            for x in [1, gw - 2]:
                px0, px1 = x * cell_px, (x + 1) * cell_px
                py0, py1 = y * cell_px, (y + 1) * cell_px
                img[py0:py1, px0:px1] = yellow_bgr

        # Grid guide lines
        lc = (38, 38, 38)
        for i in range(0, gw + 1, 10):
            cv2.line(img, (i * cell_px, 0), (i * cell_px, canvas_h - 1), lc, 1)
        for i in range(0, gh + 1, 10):
            cv2.line(img, (0, i * cell_px), (canvas_w - 1, i * cell_px), lc, 1)

        # Title
        cv2.putText(img, "CHARGER SPAWN MAP  — paint YELLOW where chargers may spawn",
                    (6, 14), font, 0.28, (110, 110, 110), 1)

        # Legend strip
        img[canvas_h:canvas_h + 1, :] = 50
        cv2.rectangle(img, (8, canvas_h + 12), (8 + 60, canvas_h + 32), yellow_bgr, -1)
        cv2.rectangle(img, (8, canvas_h + 12), (8 + 60, canvas_h + 32), (160, 160, 160), 1)
        cv2.putText(img, "YELLOW = valid charger spawn cell  (any yellow shade works)",
                    (80, canvas_h + 26), font, 0.30, (120, 200, 200), 1)
        cv2.putText(img, "Black / grey / white = forbidden (no charger spawns here)",
                    (80, canvas_h + 46), font, 0.28, (110, 110, 110), 1)
        instr = "Save as  resources/charger_map.png  |  press L in sim to hot-reload"
        (iw, _), _ = cv2.getTextSize(instr, font, 0.25, 1)
        cv2.putText(img, instr, ((canvas_w - iw) // 2, canvas_h + legend_h - 4),
                    font, 0.25, (55, 55, 55), 1)

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print("[MapImporter] Charger-map example written: {}".format(path))

    @classmethod
    def generate_robot_map_example(cls, path, gw=80, gh=70, cell_px=8):
        """Write an annotated example robot-spawn-map PNG to *path*.

        The default pattern places robots on the outer perimeter (0 cells inset),
        which mirrors the default perimeter coordinate logic.
        Paint cyan over any cells where robots should be allowed to spawn.
        """
        canvas_w, canvas_h = gw * cell_px, gh * cell_px
        legend_h = 60
        img = np.full((canvas_h + legend_h, canvas_w, 3), 18, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Default outer-perimeter band (0 cells inset — the very edge)
        cyan_bgr = (220, 220, 0)    # BGR: cyan
        for x in range(gw):
            for y in [0, gh - 1]:
                px0, px1 = x * cell_px, (x + 1) * cell_px
                py0, py1 = y * cell_px, (y + 1) * cell_px
                img[py0:py1, px0:px1] = cyan_bgr
        for y in range(gh):
            for x in [0, gw - 1]:
                px0, px1 = x * cell_px, (x + 1) * cell_px
                py0, py1 = y * cell_px, (y + 1) * cell_px
                img[py0:py1, px0:px1] = cyan_bgr

        # Grid guide lines
        lc = (38, 38, 38)
        for i in range(0, gw + 1, 10):
            cv2.line(img, (i * cell_px, 0), (i * cell_px, canvas_h - 1), lc, 1)
        for i in range(0, gh + 1, 10):
            cv2.line(img, (0, i * cell_px), (canvas_w - 1, i * cell_px), lc, 1)

        # Title
        cv2.putText(img, "ROBOT SPAWN MAP  — paint CYAN where robots may spawn",
                    (6, 14), font, 0.28, (110, 110, 110), 1)

        # Legend strip
        img[canvas_h:canvas_h + 1, :] = 50
        cv2.rectangle(img, (8, canvas_h + 12), (8 + 60, canvas_h + 32), cyan_bgr, -1)
        cv2.rectangle(img, (8, canvas_h + 12), (8 + 60, canvas_h + 32), (160, 160, 160), 1)
        cv2.putText(img, "CYAN = valid robot spawn cell  (any cyan/teal shade works)",
                    (80, canvas_h + 26), font, 0.30, (200, 200, 80), 1)
        cv2.putText(img, "Black / grey / white = forbidden (no robot spawns here)",
                    (80, canvas_h + 46), font, 0.28, (110, 110, 110), 1)
        instr = "Save as  resources/robot_map.png  |  press L in sim to hot-reload"
        (iw, _), _ = cv2.getTextSize(instr, font, 0.25, 1)
        cv2.putText(img, instr, ((canvas_w - iw) // 2, canvas_h + legend_h - 4),
                    font, 0.25, (55, 55, 55), 1)

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print("[MapImporter] Robot-map example written: {}".format(path))

    @classmethod
    def generate_charger_map(cls, path, gw=80, gh=70, cell_px=8):
        """Write a clean charger-spawn-map PNG (inner perimeter, yellow, no annotations).

        Mirrors the default warehousePerimeterCoordinatesMinusOne — one cell inset
        from all four edges.  Save to resources/charger_map.png and it will be
        auto-loaded on startup (or press L in-sim to hot-reload).
        """
        yellow_bgr = (0, 220, 220)
        img = np.zeros((gh * cell_px, gw * cell_px, 3), dtype=np.uint8)
        # Inner perimeter: x in [1..gw-2] at y=1 and y=gh-2;
        #                  y in [1..gh-2] at x=1 and x=gw-2
        for x in range(1, gw - 1):
            for y in [1, gh - 2]:
                img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = yellow_bgr
        for y in range(1, gh - 1):
            for x in [1, gw - 2]:
                img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = yellow_bgr
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print("[MapImporter] Charger map written: {}".format(path))

    @classmethod
    def generate_robot_map(cls, path, gw=80, gh=70, cell_px=8):
        """Write a clean robot-spawn-map PNG (outer perimeter, cyan, no annotations).

        Mirrors the default warehousePerimeterCoordinates — the very edge row/column
        on all four sides.  Save to resources/robot_map.png and it will be
        auto-loaded on startup (or press L in-sim to hot-reload).
        """
        cyan_bgr = (220, 220, 0)
        img = np.zeros((gh * cell_px, gw * cell_px, 3), dtype=np.uint8)
        # Outer perimeter: x in [0..gw-1] at y=0 and y=gh-1;
        #                  y in [0..gh-1] at x=0 and x=gw-1
        for x in range(gw):
            for y in [0, gh - 1]:
                img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = cyan_bgr
        for y in range(gh):
            for x in [0, gw - 1]:
                img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = cyan_bgr
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print("[MapImporter] Robot map written: {}".format(path))
