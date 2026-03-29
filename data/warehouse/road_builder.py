"""Build a road network (laneMap) from objective traffic heatmap data.

Algorithm — percentile-based hotspot detection → Prim's MST → L-path rasterisation:

  1. Gaussian-blur traffic_total (σ=2) to merge adjacent cells into blobs
  2. Threshold at a *percentile* of non-zero blurred values (not mean+σ) —
     this stays robust as traffic grows because it always selects the
     top fraction of visited cells regardless of absolute magnitudes
  3. Connected components → centroid per component = demand node
  4. If < 2 hotspots, progressively lower the percentile until enough emerge
  5. Prim's MST connects demand nodes with minimum total Manhattan distance
  6. Each MST edge rasterised as an L-path, choosing the bend direction
     that maximises overlap with already-drawn road (natural branching)

No perimeter ring — roads exist only where traffic justifies them.
"""
import numpy as np
import cv2
import heapq

# ── Tuning constants ────────────────────────────────────────────────────
_MIN_TOTAL_VISITS = 200     # suppress road building until this many total steps
_BLUR_SIGMA       = 2.0     # Gaussian σ for merging nearby traffic cells
_PERCENTILE_START = 70      # initial percentile threshold (top ~30% of visited cells)
_PERCENTILE_FLOOR = 30      # lowest percentile we'll try if hotspots are scarce
_PERCENTILE_STEP  = 10      # how much to lower each retry
_MIN_COMPONENT    = 2       # minimum cells in a blob to count as a hotspot
_TARGET_MIN_HOTSPOTS = 3    # keep lowering percentile until we have at least this many


def build_road_network(traffic_total, zone_map):
    """Return a road laneMap (uint8 H×W, 1=road 0=not) from traffic data.

    Parameters
    ----------
    traffic_total : np.ndarray (uint32, H×W)
        Lifetime cumulative visit counts per cell (never decays).
    zone_map : np.ndarray (uint8, H×W)
        Zone assignments (0=neutral, 1=import, 2=storage, 3=export).

    Returns
    -------
    lane_map : np.ndarray (uint8, H×W)
        1 = road cell, 0 = non-road cell.
    hotspots : list of (int, int)
        Centroids used for MST.  Empty if traffic is insufficient.
    edges : list of (int, int)
        MST edge index-pairs into hotspots.  Empty if < 2 hotspots.
    """
    H, W = traffic_total.shape
    lane_map = np.zeros((H, W), dtype=np.uint8)

    # ── Gate: need minimum traffic before building anything ─────────────
    if traffic_total.sum() < _MIN_TOTAL_VISITS:
        return lane_map, [], []

    # ── 1. Blur to merge nearby cells ───────────────────────────────────
    src = traffic_total.astype(np.float64)
    blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=_BLUR_SIGMA, sigmaY=_BLUR_SIGMA)
    nonzero_vals = blurred[blurred > 0]
    if len(nonzero_vals) < 5:
        return lane_map, [], []

    # ── 2. Percentile-based threshold with adaptive lowering ────────────
    hotspots = []
    pct = _PERCENTILE_START
    while pct >= _PERCENTILE_FLOOR:
        thresh = float(np.percentile(nonzero_vals, pct))
        binary = (blurred > max(thresh, 1e-9)).astype(np.uint8)
        n_comp, labels = cv2.connectedComponents(binary, connectivity=8)
        hotspots = []
        for c in range(1, n_comp):
            ys, xs = np.where(labels == c)
            if len(xs) < _MIN_COMPONENT:
                continue
            cx = int(round(xs.mean()))
            cy = int(round(ys.mean()))
            cx = max(1, min(W - 2, cx))
            cy = max(1, min(H - 2, cy))
            hotspots.append((cx, cy))
        if len(hotspots) >= _TARGET_MIN_HOTSPOTS:
            break
        pct -= _PERCENTILE_STEP

    # ── Deduplicate hotspots that collapsed to the same cell ────────────
    seen = set()
    unique = []
    for h in hotspots:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    hotspots = unique

    if len(hotspots) < 2:
        return lane_map, hotspots, []

    # ── 3. Prim's MST ──────────────────────────────────────────────────
    edges = _prims_mst(hotspots)

    # ── 4. Rasterise MST edges as Manhattan L-paths ─────────────────────
    # Try both L-bend orientations; pick the one reusing more existing
    # road cells so edges from the same hub share their initial stretch.
    for i, j in edges:
        ax, ay = hotspots[i]
        bx, by = hotspots[j]
        # Option A: horizontal first (row ay), then vertical (col bx)
        overlap_a = 0
        for x in range(min(ax, bx), max(ax, bx) + 1):
            overlap_a += lane_map[ay, x]
        for y in range(min(ay, by), max(ay, by) + 1):
            overlap_a += lane_map[y, bx]
        # Option B: vertical first (col ax), then horizontal (row by)
        overlap_b = 0
        for y in range(min(ay, by), max(ay, by) + 1):
            overlap_b += lane_map[y, ax]
        for x in range(min(ax, bx), max(ax, bx) + 1):
            overlap_b += lane_map[by, x]
        if overlap_a >= overlap_b:
            for x in range(min(ax, bx), max(ax, bx) + 1):
                lane_map[ay, x] = 1
            for y in range(min(ay, by), max(ay, by) + 1):
                lane_map[y, bx] = 1
        else:
            for y in range(min(ay, by), max(ay, by) + 1):
                lane_map[y, ax] = 1
            for x in range(min(ax, bx), max(ax, bx) + 1):
                lane_map[by, x] = 1

    # ── 5. Mark hotspot cells ───────────────────────────────────────────
    for cx, cy in hotspots:
        lane_map[cy, cx] = 1

    # ── 6. Clear outer border (no OOB hugging) ─────────────────────────
    lane_map[0, :] = 0
    lane_map[H - 1, :] = 0
    lane_map[:, 0] = 0
    lane_map[:, W - 1] = 0

    return lane_map, hotspots, edges


def _prims_mst(nodes):
    """Prim's algorithm over Manhattan distances. Returns list of (i, j) edges."""
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
        edges.append((frm, to))
        for j in range(n):
            if not in_tree[j]:
                d = abs(nodes[to][0] - nodes[j][0]) + abs(nodes[to][1] - nodes[j][1])
                heapq.heappush(heap, (d, to, j))
    return edges
