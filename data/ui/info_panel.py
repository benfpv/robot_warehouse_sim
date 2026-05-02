"""Info panel renderer for the MainGame dashboard.

draw_info_panel() is a pure function that renders the four-column INFO PANEL
strip (ROBOTS / CHARGERS / SIM STATS / PACKAGES) directly onto the composite
frame.  All state is passed as arguments — no reference to MainGame.
"""
import cv2

from data.ui.formatters import fmt_n, fmt_age
from data.ui.styles import (
    POWER_POLICY_STYLE, POWER_POLICY_HINT,
    FLOW_POLICY_STYLE, FLOW_POLICY_HINT,
    PKG_TARGET_STYLE,
)


def draw_info_panel(composite, mw, mh, sw, sh, ph, cw,
                    stats_lines, now_t, fleet, wh,
                    power_policy_buttons, flow_policy_buttons,
                    pkg_target_buttons, robot_count_buttons):
    """Render the four-column INFO PANEL onto *composite* in-place.

    Parameters
    ----------
    composite : np.ndarray
        The full composite frame (H×W×3 uint8).
    mw, mh : int
        Main warehouse view width and height in pixels.
    sw, sh : int
        Sub-view tile width and height in pixels.
    ph : int
        Info panel height in pixels.
    cw : int
        Full composite width.
    stats_lines : list
        Pre-built (label, value, colour) triples for the SIM STATS column.
    now_t : float
        Current simulation time in seconds (for age display).
    fleet : FleetSnapshot
        Immutable fleet aggregate snapshot from build_fleet_snapshot().
    wh : Warehouse
        Live warehouse instance (read-only access for robots/chargers/packages).
    power_policy_buttons : iterable of (mode, x0, y0, x1, y1)
        Button geometry from MainGame._power_policy_button_layout().
    flow_policy_buttons : iterable of (mode, x0, y0, x1, y1)
        Button geometry from MainGame._flow_policy_button_layout().
    pkg_target_buttons : iterable of (mode, x0, y0, x1, y1)
        Button geometry from MainGame._pkg_target_button_layout().
    robot_count_buttons : iterable of (action, x0, y0, x1, y1)
        Button geometry from MainGame._robot_count_button_layout().
    """
    # 4 columns of cw/4: ROBOTS | CHARGERS | SIM STATS | PACKAGES
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
    sec_col = (95, 95, 95)
    _n = fmt_n

    def put(text, x, y, col, max_x=None):
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
    _hdr_y      = mh + 10
    _data_start = mh + 22
    _split      = 11
    _div_y      = _data_start + _split * lh - lh // 2

    # Column dividers
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

    # ────────────── ROBOTS ──────────────
    _nr = len(wh.robots)
    x = put("ROBOTS ({})  ".format(_nr), pad_x, _hdr_y, white)
    put("status | battery", x, _hdr_y, sec_col)

    # +/- robot count buttons: [-] <target> [+]
    _rc_layout = list(robot_count_buttons)
    for action, bx0, by0, bx1, by1 in _rc_layout:
        _bg = (28, 28, 28)
        _edge = (55, 55, 55)
        composite[by0:by1, bx0:bx1] = _bg
        composite[by0:by0 + 1, bx0:bx1] = _edge
        composite[by1 - 1:by1, bx0:bx1] = _edge
        composite[by0:by1, bx0:bx0 + 1] = _edge
        composite[by0:by1, bx1 - 1:bx1] = _edge
        _fs_btn = 0.28
        (_tw, _th), _ = cv2.getTextSize(action, font, _fs_btn, 1)
        _tx = bx0 + max((bx1 - bx0 - _tw) // 2, 1)
        _ty = by0 + (by1 - by0 + _th) // 2
        cv2.putText(composite, action, (_tx, _ty), font, _fs_btn, (180, 180, 180), 1)
    _rc_target = wh._robot_target_count
    _rc_lbl = str(_rc_target)
    _minus_right = _rc_layout[0][3]
    _plus_left   = _rc_layout[1][1]
    _rc_mid = (_minus_right + _plus_left) // 2
    _rc_by0 = _rc_layout[0][2]
    _rc_by1 = _rc_layout[0][4]
    (_rc_tw, _rc_th), _ = cv2.getTextSize(_rc_lbl, font, 0.28, 1)
    _rc_col = (140, 200, 140) if _rc_target == _nr else (100, 180, 230)
    cv2.putText(composite, _rc_lbl, (_rc_mid - _rc_tw // 2, _rc_by0 + (_rc_by1 - _rc_by0 + _rc_th) // 2),
                font, 0.28, _rc_col, 1)

    _robots_by_status = sorted(wh.robots, key=lambda r: robot_priority.get(r.status, 7))
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
        _rzone = r.area[:4] if r.area else "?"
        _rtgt = r.areaTarget[:4] if r.areaTarget and r.areaTarget != r.area else ""
        if _rtgt:
            x = put("{}>{}  ".format(_rzone, _rtgt), x, y, zone_col.get(r.area, dim))
        else:
            x = put("{}  ".format(_rzone), x, y, zone_col.get(r.area, dim))
        age = now_t - r.createdAt
        put(fmt_age(int(age)), x, y, dim, max_x=col_w)

    composite[_div_y:_div_y + 1, 0:col_w] = 40

    _robots_by_batt = sorted(wh.robots, key=lambda r: r.batteryPercent)
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
    _nc = len(wh.chargers)
    put("CHARGERS ({})".format(_nc), col_w + pad_x, _hdr_y, white)
    _pp_mode = getattr(wh, '_power_policy_mode', 'balanced')
    _pp_style = POWER_POLICY_STYLE.get(_pp_mode, POWER_POLICY_STYLE['balanced'])
    for mode, bx0, by0, bx1, by1 in power_policy_buttons:
        style = POWER_POLICY_STYLE[mode]
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
    put("mode: {} ({})".format(_pp_mode.upper(), POWER_POLICY_HINT.get(_pp_mode, _pp_mode)),
        col_w + pad_x, _data_start - 1, _pp_style['text'], max_x=col_w * 2)
    _chg_data_start = _data_start + lh
    _chg_div_y = _div_y + lh

    _crmap_obj = {}
    for _cr in wh.robots:
        for _ct in _cr.actionQueue:
            if _ct[1] in ("move to charging station", "charging"):
                _crmap_obj[tuple(_ct[0])] = _cr

    _chargers_sorted = sorted(wh.chargers, key=lambda c: charger_priority.get(c.status, 3))
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

    composite[_chg_div_y:_chg_div_y + 1, col_w:col_w * 2] = 40

    _avg_batt  = fleet.avg_battery
    _pressure  = getattr(wh, '_charger_pressure', 0.0)
    _pressure_raw = getattr(wh, '_charger_pressure_raw', _pressure)
    _threshold = getattr(wh, '_charge_threshold', 30)
    _threshold_target = getattr(wh, '_charge_threshold_target', _threshold)
    _n_idle_r   = fleet.n_idle
    _n_chrging  = fleet.n_charging
    _n_enroute  = fleet.n_enroute
    _n_working  = fleet.n_working
    _avg_drain  = fleet.avg_drain

    _fleet_lines = [
        ("avg batt",   "{:.0f}%".format(_avg_batt),     batt_col(int(_avg_batt))),
        ("avg drain",  "{:.3f}/t".format(_avg_drain),    dim),
        ("policy",     "Power/{}".format(getattr(wh, '_power_policy_mode', 'balanced')[:4]), _pp_style['text']),
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
    _fp_mode = getattr(wh, '_flow_policy_mode', 'balanced')
    _fp_style = FLOW_POLICY_STYLE.get(_fp_mode, FLOW_POLICY_STYLE['balanced'])
    for mode, bx0, by0, bx1, by1 in flow_policy_buttons:
        style = FLOW_POLICY_STYLE[mode]
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
    put("mode: {} ({})".format(_fp_mode.upper(), FLOW_POLICY_HINT.get(_fp_mode, _fp_mode)),
        _sc2 + pad_x, _data_start - 1, _fp_style['text'], max_x=_sc3)
    _sim_data_start = _data_start + lh
    _swhite = (200, 200, 200)
    _sdim   = (95, 95, 95)
    _s_val_x = _sc2 + 52
    for si, (lbl, val, vcol) in enumerate(stats_lines):
        y = _sim_data_start + si * lh
        if y >= mh + ph - 2:
            break
        put(lbl + ":", _sc2 + pad_x, y, _sdim)
        put(val, _s_val_x, y, vcol or _swhite, max_x=_sc3)

    # ────────────── PACKAGES ──────────────
    _np = len(wh.packages)
    _nm = wh.packagesMaxQuantity
    put("PKGS ({}/{})".format(_np, _nm), col_w * 3 + pad_x, _hdr_y, white)
    _active_mode = getattr(wh, '_pkg_target_mode', 'random')
    _pm_style = PKG_TARGET_STYLE.get(_active_mode, PKG_TARGET_STYLE['random'])
    for mode, bx0, by0, bx1, by1 in pkg_target_buttons:
        style = PKG_TARGET_STYLE[mode]
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
        dl_str = "LATE" if td < 0 else fmt_age(int(td))
        age = now_t - p.createdAt
        age_str = fmt_age(int(age))
        tgt = p.areaTarget if p.areaTarget not in ("none", "") else ""
        st = pkg_short.get(p.status, p.status[:3])
        x = x_start
        x = put("P{} ".format(_n(p.packageNumber)), x, y, tuple(p.colour))
        x = put("{} ".format(dl_str), x, y, dl_col(td))
        x = put("{} ".format(age_str), x, y, (200, 160, 80))
        x = put("{} ".format(p.area[:4]), x, y, zone_col.get(p.area, gray))
        if tgt:
            x = put(">{} ".format(tgt[:4]), x, y, zone_col.get(tgt, gray))
        if p.carrier != -1:
            x = put("R{} ".format(p.carrier), x, y, (80, 160, 220))
        x = put("{} ".format(st), x, y, dim)
        _wt = getattr(p.itemValues, 'weight', None) if p.itemValues else None
        if _wt:
            x = put("{}kg ".format(_wt), x, y, (140, 140, 100))
        _iname = getattr(p.itemValues, 'name', '') if p.itemValues else ''
        if _iname:
            x = put("{} ".format(_iname), x, y, (120, 140, 120), max_x=max_end)
        _city = getattr(p.addressTo, 'city', '') if hasattr(p, 'addressTo') and p.addressTo else ''
        if _city and x < max_end - 10:
            put(_city, x, y, (110, 110, 130), max_x=max_end)

    _pkgs_deadline = sorted(wh.packages, key=lambda p: p.timeToDeadline)
    _pkg_rows_start = _data_start + lh
    _max_pkg_rows = _split * 2
    for pi in range(_max_pkg_rows):
        if pi >= len(_pkgs_deadline): break
        y = _pkg_rows_start + pi * lh
        if y >= mh + ph - 2: break
        _draw_pkg(col_w * 3 + pad_x, y, _pkgs_deadline[pi], cw)
