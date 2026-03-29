"""Physarum-inspired organic road builder  v4 — frame-distributed.

Heavy A* work is spread across simulation frames via a step-able job object.
Each call to ``job.step()`` routes 2-3 pairs, so the sim never blocks for
more than ~3-5 ms on road building.

Density controls tightened so roads cover essential corridors only — packages
will eventually not be placed on road cells.
"""
import numpy as np
import cv2
import heapq

# ── Tuning ──────────────────────────────────────────────────────────────
_MIN_TOTAL_VISITS    = 300      # gate: suppress until substantial traffic exists
_BLUR_SIGMA          = 1.5      # hotspot detection blur
_NMS_RADIUS          = 6        # non-max suppression spacing (cells) — wider = fewer hotspots
_MIN_HOTSPOT_PCT     = 40       # hotspot floor percentile — higher = fewer weak nodes
_MAX_HOTSPOTS        = 10       # cap — fewer = cleaner network
_KNN_K               = 1        # shortcut edges — minimal redundancy beyond MST
_COST_BLUR_SIGMA     = 2.5      # smooth cost grid — wider = less weaving
_TRAFFIC_WEIGHT      = 4.0      # how strongly paths prefer traffic corridors
_ZONE_COST_PENALTY   = 1.8      # cost multiplier inside zone interiors
_CHARGER_COST        = 1e6      # effectively impassable
_EMA_BLEND           = 0.35     # weight of recent EMA vs lifetime traffic
_REINFORCE_STRENGTH  = 3.0      # tube-thickening multiplier
_FLOW_MOMENTUM       = 0.70     # how much of old flow accumulator to keep
_BRANCH_PCT          = 50       # flow percentile → branch  (tier 1) — only strong routes
_COLLECTOR_PCT       = 70       # flow percentile → collector (tier 2)
_ARTERY_PCT          = 90       # flow percentile → arterial  (tier 3)
_PATHS_PER_STEP      = 3        # A* paths computed per step() call
_TURN_PENALTY        = 1.8      # A* surcharge for changing direction (strong → straighter)
_RDP_EPSILON         = 2.5      # Ramer-Douglas-Peucker simplification tolerance (aggressive smoothing)
_MIN_PATH_LEN        = 5        # discard paths shorter than this (trivial connections)
_COLD_FILTER_PCT     = 60       # percentile below which traffic is zeroed out

# Plaza detection
_PLAZA_FLOW_PCT      = 75       # flow percentile threshold (much higher than road tiers)
_MIN_PLAZA_CELLS     = 16       # minimum connected hot cells to form a plaza
_PLAZA_FILL_RATIO    = 0.45     # min fill ratio (hot / bounding-box area)
_PLAZA_MAX_ASPECT    = 2.5      # skip elongated components (thin lines)
_CHARGER_CLUSTER_R   = 2        # dilation radius for charger clustering
_CHARGER_PLAZA_PAD   = 1        # padding around charger cluster bounding box
_CHARGER_PLAZA_MIN_T = 0.25     # min fraction of plaza cells with traffic

# 8-direction movement: (dx, dy, base_step_cost)
_DIRS_8 = [
    ( 1,  0, 1.0), (-1,  0, 1.0), ( 0,  1, 1.0), ( 0, -1, 1.0),
    ( 1,  1, 1.414), ( 1, -1, 1.414), (-1,  1, 1.414), (-1, -1, 1.414),
]


# ═══════════════════════════════════════════════════════════════════════
#  Frame-distributed job
# ═══════════════════════════════════════════════════════════════════════

class RoadBuildJob:
    """Spreads A* computation across frames.  Call step() each tick."""

    __slots__ = (
        '_H', '_W', '_hotspots', '_cost_grid', '_all_pairs', '_mst_set',
        '_prev_flow', '_fresh_flow', '_pair_idx', '_phase', '_done',
        '_lane_map', '_edges', '_flow_accum', '_charger_grid', '_plaza_map',
    )

    def __init__(self, traffic_total, zone_map, traffic_ema, charger_grid, prev_flow):
        H, W = traffic_total.shape
        self._H, self._W = H, W
        self._prev_flow = prev_flow
        self._charger_grid = charger_grid
        self._done = False
        self._lane_map = np.zeros((H, W), dtype=np.uint8)
        self._plaza_map = np.zeros((H, W), dtype=np.uint8)
        self._edges = []
        self._flow_accum = prev_flow if prev_flow is not None else np.zeros((H, W), dtype=np.float64)

        # ── cheap setup (< 2 ms) ───────────────────────────────────────
        if traffic_total.sum() < _MIN_TOTAL_VISITS:
            self._hotspots = []
            self._done = True
            return

        blended = _blend_traffic(traffic_total, traffic_ema)
        self._hotspots = _detect_hotspots(blended, H, W)
        if len(self._hotspots) < 2:
            self._done = True
            return

        self._cost_grid = _build_cost_grid(blended, zone_map, charger_grid)

        coords = [(h[0], h[1]) for h in self._hotspots]
        mst_pairs = _prims_mst(coords)
        knn_pairs = _knn_pairs(self._hotspots, k=_KNN_K)
        all_pair_set = set()
        for i, j in mst_pairs:
            all_pair_set.add((min(i, j), max(i, j)))
        self._mst_set = set(all_pair_set)
        for i, j in knn_pairs:
            all_pair_set.add((min(i, j), max(i, j)))
        self._all_pairs = sorted(all_pair_set)

        self._fresh_flow = np.zeros((H, W), dtype=np.float64)
        self._pair_idx = 0
        self._phase = 'route'   # 'route' → 'reinforce' → 'finalise'

    # ─────────────────────────────────────────────────────────────────
    @property
    def done(self):
        return self._done

    def step(self, max_paths=_PATHS_PER_STEP):
        """Process up to *max_paths* A* queries.  Returns True when complete."""
        if self._done:
            return True

        if self._phase == 'route':
            self._step_route(max_paths)
        elif self._phase == 'reinforce':
            self._step_reinforce(max_paths)
        elif self._phase == 'finalise':
            self._finalise()

        return self._done

    def result(self):
        """Return (lane_map, hotspots, edges, flow_accum, plaza_map)."""
        return self._lane_map, self._hotspots, self._edges, self._flow_accum, self._plaza_map

    # ── internal phases ─────────────────────────────────────────────
    def _step_route(self, n):
        hs = self._hotspots
        pairs = self._all_pairs
        end = min(self._pair_idx + n, len(pairs))
        for k in range(self._pair_idx, end):
            i, j = pairs[k]
            path = _astar_8dir(self._cost_grid, self._H, self._W,
                               (hs[i][0], hs[i][1]), (hs[j][0], hs[j][1]))
            if path:
                path = _simplify_path(path)
                if len(path) < _MIN_PATH_LEN:
                    continue  # skip trivial short connections
                w = (hs[i][2] + hs[j][2]) * 0.5
                for px, py in path:
                    self._fresh_flow[py, px] += w
        self._pair_idx = end
        if self._pair_idx >= len(pairs):
            # Prepare reinforced cost grid and reset cursor
            fmax = self._fresh_flow.max()
            if fmax > 0:
                fn = self._fresh_flow / fmax
                self._cost_grid = self._cost_grid / (1.0 + np.sqrt(fn) * _REINFORCE_STRENGTH)
            self._pair_idx = 0
            self._phase = 'reinforce'

    def _step_reinforce(self, n):
        hs = self._hotspots
        pairs = self._all_pairs
        end = min(self._pair_idx + n, len(pairs))
        for k in range(self._pair_idx, end):
            i, j = pairs[k]
            path = _astar_8dir(self._cost_grid, self._H, self._W,
                               (hs[i][0], hs[i][1]), (hs[j][0], hs[j][1]))
            if path:
                path = _simplify_path(path)
                if len(path) < _MIN_PATH_LEN:
                    continue  # skip trivial short connections
                w = (hs[i][2] + hs[j][2]) * 0.5
                for px, py in path:
                    self._fresh_flow[py, px] += w * 1.5
        self._pair_idx = end
        if self._pair_idx >= len(pairs):
            self._phase = 'finalise'
            self._finalise()

    def _finalise(self):
        H, W = self._H, self._W
        # Blend into persistent accumulator
        if self._prev_flow is not None and self._prev_flow.shape == (H, W):
            self._flow_accum = _FLOW_MOMENTUM * self._prev_flow + (1.0 - _FLOW_MOMENTUM) * self._fresh_flow
        else:
            self._flow_accum = self._fresh_flow

        # Edge list
        self._edges = [(i, j, 'mst' if (i, j) in self._mst_set else 'shortcut')
                       for i, j in self._all_pairs]

        # Threshold
        active = self._flow_accum[self._flow_accum > 0]
        if len(active) > 0:
            p_b = float(np.percentile(active, _BRANCH_PCT))
            p_c = float(np.percentile(active, _COLLECTOR_PCT))
            p_a = float(np.percentile(active, _ARTERY_PCT))
            lm = self._lane_map
            lm[self._flow_accum >= p_b] = 1
            lm[self._flow_accum >= p_c] = 2
            lm[self._flow_accum >= p_a] = 3

            _prune_dead_ends(lm)

            # ── Widen all roads to minimum 2 cells thick ───────────
            _widen_roads(lm, self._charger_grid)

            # Re-prune after widening (dilation can create new stubs)
            _prune_dead_ends(lm)

            # Stamp plazas (traffic areas + charger clusters)
            _stamp_plazas(lm, self._flow_accum, self._charger_grid, self._plaza_map)

            for hx, hy, _ in self._hotspots:
                if 0 < hx < W - 1 and 0 < hy < H - 1:
                    lm[hy, hx] = max(lm[hy, hx], 3)

            lm[0, :] = 0;  lm[H - 1, :] = 0
            lm[:, 0] = 0;  lm[:, W - 1] = 0

        self._done = True


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════

def _blend_traffic(traffic_total, traffic_ema):
    """Blend lifetime + recent EMA, then zero out the coldest 50 %."""
    total_f = traffic_total.astype(np.float64)
    t_max = max(total_f.max(), 1.0)
    blended = total_f / t_max                               # normalised lifetime

    if traffic_ema is not None:
        ema_f = traffic_ema.astype(np.float64)
        e_max = max(ema_f.max(), 1.0)
        blended = (1.0 - _EMA_BLEND) * blended + _EMA_BLEND * (ema_f / e_max)

    # Discard the coldest traffic — only warm/hot cells contribute to roads
    nonzero = blended[blended > 0]
    if len(nonzero) > 0:
        median = float(np.percentile(nonzero, _COLD_FILTER_PCT))
        blended[blended < median] = 0.0

    return blended


# ── Hotspot detection ───────────────────────────────────────────────────

def _detect_hotspots(blended, H, W):
    """Find local-maxima hotspots with non-maximum suppression."""
    blurred = cv2.GaussianBlur(blended, (0, 0),
                                sigmaX=_BLUR_SIGMA, sigmaY=_BLUR_SIGMA)
    nonzero = blurred[blurred > 0]
    if len(nonzero) < 5:
        return []

    ksize = _NMS_RADIUS * 2 + 1
    dilated = cv2.dilate(blurred, np.ones((ksize, ksize), dtype=np.uint8))
    floor_val = float(np.percentile(nonzero, _MIN_HOTSPOT_PCT))
    local_max_mask = (blurred == dilated) & (blurred >= floor_val)

    ys, xs = np.where(local_max_mask)
    if len(xs) < 2:
        return []

    intensities = blurred[ys, xs]
    order = np.argsort(-intensities)

    kept = []
    suppressed = set()
    for idx in order:
        ix, iy = int(xs[idx]), int(ys[idx])
        if (ix, iy) in suppressed:
            continue
        kept.append((max(1, min(W - 2, ix)),
                      max(1, min(H - 2, iy)),
                      float(intensities[idx])))
        for jdx in order:
            jx, jy = int(xs[jdx]), int(ys[jdx])
            if (jx, jy) not in suppressed and (jx, jy) != (ix, iy):
                if abs(jx - ix) <= _NMS_RADIUS and abs(jy - iy) <= _NMS_RADIUS:
                    suppressed.add((jx, jy))
        if len(kept) >= _MAX_HOTSPOTS:
            break

    return kept


# ── Cost grid (zone-aware, charger-aware) ───────────────────────────────

def _build_cost_grid(blended, zone_map, charger_grid):
    """Build traversal cost: low where traffic is high, penalise zones/chargers."""
    # Smooth so cells near corridors also benefit
    smoothed = cv2.GaussianBlur(blended, (0, 0),
                                 sigmaX=_COST_BLUR_SIGMA, sigmaY=_COST_BLUR_SIGMA)
    # Base cost: inverse traffic
    cost_grid = 1.0 / (1.0 + _TRAFFIC_WEIGHT * smoothed)

    # Zone penalty: roads prefer neutral corridors (zone 0) over zone interiors
    if zone_map is not None:
        zone_interior = zone_map > 0  # import / storage / export
        cost_grid[zone_interior] *= _ZONE_COST_PENALTY

    # Charger cells: effectively impassable
    if charger_grid is not None:
        cg = np.asarray(charger_grid)
        if cg.shape == cost_grid.shape:
            cost_grid[cg > 0] = _CHARGER_COST

    return cost_grid


# ── Prim's MST (connectivity backbone) ─────────────────────────────────

def _prims_mst(nodes):
    """Prim's algorithm over Manhattan distances.  Returns list of (i, j)."""
    n = len(nodes)
    if n < 2:
        return []
    in_tree = [False] * n
    heap = []
    edges = []
    in_tree[0] = True
    for j in range(1, n):
        d = abs(nodes[0][0] - nodes[j][0]) + abs(nodes[0][1] - nodes[j][1])
        heapq.heappush(heap, (d, 0, j))
    while heap and len(edges) < n - 1:
        cost, frm, to = heapq.heappop(heap)
        if in_tree[to]:
            continue
        in_tree[to] = True
        edges.append((min(frm, to), max(frm, to)))
        for j in range(n):
            if not in_tree[j]:
                d = abs(nodes[to][0] - nodes[j][0]) + abs(nodes[to][1] - nodes[j][1])
                heapq.heappush(heap, (d, to, j))
    return edges


# ── K-nearest-neighbour pair selection ──────────────────────────────────

def _knn_pairs(hotspots, k=3):
    """Return unique (i, j) pairs where each node connects to K nearest."""
    n = len(hotspots)
    pairs = set()
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = abs(hotspots[i][0] - hotspots[j][0]) + abs(hotspots[i][1] - hotspots[j][1])
            dists.append((d, j))
        dists.sort()
        for _, j in dists[:k]:
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


# ── 8-direction A* through cost grid ───────────────────────────────────

def _astar_8dir(cost_grid, H, W, start, goal):
    """8-direction A* on a cost grid.  Returns list of (x, y) waypoints."""
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if sx == gx and sy == gy:
        return []
    # Minimum possible cell cost for admissible heuristic scaling
    min_cost = max(cost_grid.min(), 0.01)

    g_score = np.full((H, W), np.inf, dtype=np.float64)
    g_score[sy, sx] = 0.0
    closed = np.zeros((H, W), dtype=np.bool_)
    parent = {}
    tie = 0
    open_heap = [(0.0, tie, sx, sy)]
    tie += 1

    while open_heap:
        f, _, cx, cy = heapq.heappop(open_heap)
        if closed[cy, cx]:
            continue
        if cx == gx and cy == gy:
            path = [(gx, gy)]
            cur = (gx, gy)
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path
        closed[cy, cx] = True

        # Incoming direction for turn penalty
        if (cx, cy) in parent:
            px, py = parent[(cx, cy)]
            prev_dx, prev_dy = cx - px, cy - py
        else:
            prev_dx, prev_dy = None, None

        for dx, dy, base_cost in _DIRS_8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if closed[ny, nx]:
                continue
            cell_cost = cost_grid[ny, nx]
            move_cost = base_cost * cell_cost
            # Penalise direction changes → straighter paths
            # Use flat penalty so it's equally effective on low-cost (traffic) cells
            if prev_dx is not None and (dx != prev_dx or dy != prev_dy):
                move_cost += _TURN_PENALTY
            new_g = g_score[cy, cx] + move_cost
            if new_g < g_score[ny, nx]:
                g_score[ny, nx] = new_g
                parent[(nx, ny)] = (cx, cy)
                ddx = abs(nx - gx); ddy = abs(ny - gy)
                h = (max(ddx, ddy) + 0.414 * min(ddx, ddy)) * min_cost
                heapq.heappush(open_heap, (new_g + h, tie, nx, ny))
                tie += 1

    return []  # no path


# ── Path straightening (RDP + Bresenham rasterise) ─────────────────────

def _bresenham(x0, y0, x1, y1):
    """Yield (x, y) cells along a Bresenham line."""
    dx = abs(x1 - x0);  dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy;  x0 += sx
        if e2 <= dx:
            err += dx;  y0 += sy


def _simplify_path(path):
    """RDP simplify → Bresenham rasterise.  Returns list of (x, y) cells."""
    if len(path) < 3:
        return path
    pts = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts, _RDP_EPSILON, closed=False)
    waypoints = simplified.reshape(-1, 2).astype(np.int32)
    cells = []
    seen = set()
    for k in range(len(waypoints) - 1):
        for cx, cy in _bresenham(int(waypoints[k][0]), int(waypoints[k][1]),
                                  int(waypoints[k + 1][0]), int(waypoints[k + 1][1])):
            if (cx, cy) not in seen:
                seen.add((cx, cy))
                cells.append((cx, cy))
    # Ensure last waypoint included
    lx, ly = int(waypoints[-1][0]), int(waypoints[-1][1])
    if (lx, ly) not in seen:
        cells.append((lx, ly))
    return cells


# ── Plaza detection (traffic areas + charger clusters) ─────────────────

def _stamp_plazas(lane_map, flow_accum, charger_grid, plaza_map):
    """Detect compact high-traffic regions and charger clusters → filled road areas."""
    H, W = lane_map.shape

    # ── 1. Traffic-based plazas ─────────────────────────────────────
    # Only the densest traffic clusters qualify — well above normal road threshold.
    active = flow_accum[flow_accum > 0]
    if len(active) > 0:
        thresh = float(np.percentile(active, _PLAZA_FLOW_PCT))
        hot = (flow_accum >= thresh).astype(np.uint8)
        # No dilation — only naturally dense clusters qualify
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hot)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < _MIN_PLAZA_CELLS:
                continue
            x0 = stats[lab, cv2.CC_STAT_LEFT]
            y0 = stats[lab, cv2.CC_STAT_TOP]
            bw = stats[lab, cv2.CC_STAT_WIDTH]
            bh = stats[lab, cv2.CC_STAT_HEIGHT]
            bbox_area = max(bw * bh, 1)
            fill_ratio = area / bbox_area
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if fill_ratio < _PLAZA_FILL_RATIO or aspect > _PLAZA_MAX_ASPECT:
                continue
            # Component is compact → mark as plaza (separate from roads)
            pts_yx = np.column_stack(np.where(labels == lab))
            if len(pts_yx) < 3:
                continue
            hull = cv2.convexHull(pts_yx[:, ::-1].astype(np.int32))
            hull_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            if charger_grid is not None:
                hull_mask[charger_grid > 0] = 0
            plaza_map[hull_mask > 0] = 1

    # ── 2. Charger-cluster plazas ───────────────────────────────────
    if charger_grid is None:
        return
    cg = np.asarray(charger_grid, dtype=np.uint8)
    if cg.max() == 0:
        return
    # Dilate charger mask to connect nearby chargers
    ck = 2 * _CHARGER_CLUSTER_R + 1
    charger_dilated = cv2.dilate(cg, np.ones((ck, ck), dtype=np.uint8))
    num_c, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(charger_dilated)
    for lab in range(1, num_c):
        # Count actual chargers in this component
        component_mask = (c_labels == lab)
        n_chargers = int(cg[component_mask].sum())
        if n_chargers < 2:
            continue
        # Bounding box + padding
        x0 = max(c_stats[lab, cv2.CC_STAT_LEFT] - _CHARGER_PLAZA_PAD, 1)
        y0 = max(c_stats[lab, cv2.CC_STAT_TOP] - _CHARGER_PLAZA_PAD, 1)
        x1 = min(x0 + c_stats[lab, cv2.CC_STAT_WIDTH] + 2 * _CHARGER_PLAZA_PAD, W - 1)
        y1 = min(y0 + c_stats[lab, cv2.CC_STAT_HEIGHT] + 2 * _CHARGER_PLAZA_PAD, H - 1)
        # Only stamp if there's meaningful traffic in the region
        region_flow = flow_accum[y0:y1, x0:x1]
        if region_flow.size == 0:
            continue
        traffic_frac = np.count_nonzero(region_flow) / region_flow.size
        if traffic_frac < _CHARGER_PLAZA_MIN_T:
            continue
        # Fill bounding box as plaza only, skip charger cells
        plaza = np.zeros((H, W), dtype=np.uint8)
        plaza[y0:y1, x0:x1] = 1
        plaza[cg > 0] = 0
        plaza_map[plaza > 0] = 1

# ── Road widening (minimum 2-cell width) ───────────────────────────────

def _widen_roads(lane_map, charger_grid):
    """Expand thin (1-cell wide) road segments to minimum 2 cells wide.

    Only targets genuinely thin sections — roads already 2+ cells wide
    are untouched.  Each thin pixel gets exactly one new neighbour added
    in the perpendicular direction, so the result is exactly 2-wide.
    Charger cells and grid edges are never overwritten.
    """
    H, W = lane_map.shape
    road = lane_map > 0

    cg = None
    if charger_grid is not None:
        cg = np.asarray(charger_grid)
        if cg.shape != (H, W):
            cg = None

    additions = []  # (y, x, tier)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not road[y, x]:
                continue
            tier = int(lane_map[y, x])

            # Thin = both cardinal neighbours on an axis are non-road
            thin_h = not road[y, x - 1] and not road[y, x + 1]
            thin_v = not road[y - 1, x] and not road[y + 1, x]

            if not thin_h and not thin_v:
                continue  # already 2+ wide in both axes

            # Pick first valid empty neighbour on a thin axis
            candidates = []
            if thin_h:
                candidates.extend([(y, x + 1), (y, x - 1)])
            if thin_v:
                candidates.extend([(y + 1, x), (y - 1, x)])

            for ny, nx in candidates:
                if ny <= 0 or ny >= H - 1 or nx <= 0 or nx >= W - 1:
                    continue
                if road[ny, nx]:
                    continue  # already road
                if cg is not None and cg[ny, nx]:
                    continue  # charger
                additions.append((ny, nx, tier))
                break

    for ay, ax, atier in additions:
        if lane_map[ay, ax] < atier:
            lane_map[ay, ax] = atier


# ── Dead-end pruning ───────────────────────────────────────────────────

def _prune_dead_ends(lane_map):
    """Remove leaf pixels that have only 0-1 road neighbours (dead stubs).

    Runs multiple passes until stable — widening can create cascading stubs.
    """
    H, W = lane_map.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    for _ in range(4):  # up to 4 passes for cascading stubs
        road = (lane_map > 0).astype(np.uint8)
        nbr_count = cv2.filter2D(road, cv2.CV_16U, kernel)
        # Prune: road cell with ≤1 neighbour and NOT arterial (value 3)
        prune_mask = (lane_map > 0) & (lane_map < 3) & (nbr_count <= 1)
        if not prune_mask.any():
            break
        lane_map[prune_mask] = 0
