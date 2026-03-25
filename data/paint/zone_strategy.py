"""ZoneStrategy — dynamically rebalances import/storage/export zones.

Monitors per-zone utilization via an exponential moving average, identifies
bottleneck (over-utilized) and donor (under-utilized or neutral) zones, then
incrementally grows/shrinks zone boundaries at their frontiers.

Mutations go through ``wh.pending_zone_changes`` so the existing
``reconcile_zone_changes()`` pipeline handles in-flight package safety.
"""
import time
import numpy as np

# Zone constants (must match warehouse.py / paint_handler.py)
ZONE_NONE    = 0
ZONE_IMPORT  = 1
ZONE_STORAGE = 2
ZONE_EXPORT  = 3

_ZONE_IDS   = (ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT)
_ZONE_NAMES = {ZONE_NONE: 'neutral', ZONE_IMPORT: 'import',
               ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}
# Short abbreviations used in the UI stats line where space is tight
_ZONE_ABBR  = {ZONE_NONE: 'clr', ZONE_IMPORT: 'imp',
               ZONE_STORAGE: 'sto', ZONE_EXPORT: 'exp'}

# 4-connected neighbor offsets
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class ZoneStrategy:
    """Pluggable strategy for the WarehouseOptimizer orchestrator.

    Public interface consumed by WarehouseOptimizer:
        name            — human-readable name
        label           — short label for the UI toggle button
        enabled         — bool, toggled by UI
        EVAL_INTERVAL   — ticks between optimization passes
        step(wh, budget)— called by orchestrator each tick
        reset(wh)       — called on hot-reload
    """

    name  = 'Zone'
    label = 'AUTO'   # shown on the right-strip toggle button in the zone-map panel

    # ── Tuning constants ──────────────────────────────────────────────
    EVAL_INTERVAL   = 20      # ticks between passes (~0.5 sec at 40 tps)
    WARMUP_TICKS    = 200     # ~5 sec startup grace period
    EMA_ALPHA       = 0.20    # smoothing factor for utilization EMA
    HIGH_THRESH     = 0.80    # utilization above this = bottleneck
    LOW_THRESH      = 0.40    # utilization below this = donor candidate
    MIN_ZONE_CELLS  = 50      # never shrink a zone below this count
    COOLDOWN_MULTI  = 2       # cooldown = COOLDOWN_MULTI × EVAL_INTERVAL ticks

    # Target slot proportions (used as tiebreaker bias, not hard constraint)
    _TARGET_RATIOS  = {ZONE_IMPORT: 0.20, ZONE_STORAGE: 0.45, ZONE_EXPORT: 0.35}

    def __init__(self, warehouse_res):
        self.enabled = True
        self._gw, self._gh = warehouse_res
        self._tick      = 0       # ticks since last eval
        self._total     = 0       # total sim ticks elapsed (warmup; never reset)
        self._util_ema  = {z: 0.0 for z in _ZONE_IDS}
        self._cooldown  = {z: 0   for z in _ZONE_IDS}   # remaining cooldown ticks
        self._zonable   = None    # bool mask (H, W), computed lazily
        self._last_action_str  = ''   # description of last mutation (for UI)
        self._last_action_time = 0.0  # wall-clock time of last mutation

    # ── Orchestrator interface ─────────────────────────────────────────

    def step(self, wh, budget):
        """Called every tick by the orchestrator when enabled."""
        self._total += 1   # always advance so warmup measures sim age, not enable age
        if not self.enabled:
            return
        self._tick  += 1
        # Decrement cooldowns
        for z in _ZONE_IDS:
            if self._cooldown[z] > 0:
                self._cooldown[z] -= 1
        if self._total < self.WARMUP_TICKS:
            return
        if self._tick < self.EVAL_INTERVAL:
            return
        self._tick = 0
        self._ensure_zonable(wh)
        self._compute_utilization(wh)
        action = self._decide_action(wh)
        if action is None:
            return
        grow_zone, shrink_zone, intensity = action
        intensity = min(intensity, budget)
        n_grow = n_shrink = 0
        if grow_zone is not None:
            cells = self._find_frontier_grow(grow_zone, wh)[:intensity]
            self._apply(wh, cells, grow_zone)
            n_grow = len(cells)
        if shrink_zone is not None and shrink_zone != ZONE_NONE:
            cells = self._find_frontier_shrink(shrink_zone, wh)[:intensity]
            self._apply(wh, cells, ZONE_NONE)
            n_shrink = len(cells)
        if n_grow or n_shrink:
            parts = []
            if n_grow:
                parts.append('+{} {}'.format(n_grow, _ZONE_ABBR[grow_zone]))
            if n_shrink:
                parts.append('-{} {}'.format(n_shrink, _ZONE_ABBR[shrink_zone]))
            # Only cool down the donor (shrunk zone) — not the bottleneck.
            # The bottleneck must remain free to keep growing every eval until
            # utilization drops below HIGH_THRESH.
            if shrink_zone is not None and shrink_zone != ZONE_NONE:
                self._cooldown[shrink_zone] = self.COOLDOWN_MULTI * self.EVAL_INTERVAL
            self._last_action_str  = ', '.join(parts)
            self._last_action_time = time.time()

    def reset(self, wh):
        """Reset internal state (called on hot-reload).

        ``_total`` is intentionally preserved so the startup warmup does not
        re-trigger after a map hot-reload mid-simulation.
        """
        self._tick     = 0
        self._util_ema = {z: 0.0 for z in _ZONE_IDS}
        self._cooldown = {z: 0   for z in _ZONE_IDS}
        self._zonable  = None     # recomputed lazily from new spawn maps
        self._last_action_str  = ''
        self._last_action_time = 0.0

    # ── Utilization ────────────────────────────────────────────────────

    def _compute_utilization(self, wh):
        """Update per-zone utilization EMA from warehouse counts."""
        counts = {
            ZONE_IMPORT:  wh.packagesInImportCount  + wh.packagesPlannedInImportCount,
            ZONE_STORAGE: wh.packagesInStorageCount + wh.packagesPlannedInStorageCount,
            ZONE_EXPORT:  wh.packagesInExportCount  + wh.packagesPlannedInExportCount,
        }
        slots = {
            ZONE_IMPORT:  max(wh.numberOfImportSlots,  1),
            ZONE_STORAGE: max(wh.numberOfStorageSlots, 1),
            ZONE_EXPORT:  max(wh.numberOfExportSlots,  1),
        }
        a = self.EMA_ALPHA
        for z in _ZONE_IDS:
            util = min(counts[z] / slots[z], 1.0)
            self._util_ema[z] = a * util + (1.0 - a) * self._util_ema[z]

    # ── Decision ───────────────────────────────────────────────────────

    def _decide_action(self, wh):
        """Determine which zone to grow and which to shrink.

        Returns (grow_zone_id, shrink_zone_id, intensity) or None.
        """
        slot_counts = {
            ZONE_IMPORT:  wh.numberOfImportSlots,
            ZONE_STORAGE: wh.numberOfStorageSlots,
            ZONE_EXPORT:  wh.numberOfExportSlots,
        }

        # Find bottleneck: highest utilization above threshold
        bottleneck = None
        best_util  = self.HIGH_THRESH
        for z in _ZONE_IDS:
            if self._util_ema[z] > best_util:
                best_util  = self._util_ema[z]
                bottleneck = z
        if bottleneck is None:
            return None

        # Adaptive intensity based on severity
        delta     = best_util - self.HIGH_THRESH
        intensity = max(1, min(int(delta * 100), 20))

        # Find donor: lowest utilization below threshold (not on cooldown)
        donor      = None
        worst_util = self.LOW_THRESH
        for z in _ZONE_IDS:
            if z == bottleneck:
                continue
            if self._cooldown[z] > 0:
                continue
            if slot_counts[z] <= self.MIN_ZONE_CELLS:
                continue
            if self._util_ema[z] < worst_util:
                worst_util = self._util_ema[z]
                donor = z

        # If no zone donor is available, grow from neutral cells (no shrink)
        # donor = None means only grow (from neutral), don't shrink any zone
        return (bottleneck, donor, intensity)

    # ── Frontier selection ─────────────────────────────────────────────

    def _find_frontier_grow(self, zone_id, wh):
        """Return candidate cells to convert TO *zone_id*, sorted best-first.

        Candidates: cells with ≥2 zone-neighbours (compact), inside zonable mask,
        not occupied.  If ≥2 yields no candidates, falls back to ≥1 neighbour so
        small or fragmented zones are never permanently stuck.
        Prefer neutral cells over stealing from another zone.
        Sorted by distance to zone centroid (closest first).
        """
        zm = wh.zoneMap
        mask = self._zonable
        gw, gh = self._gw, self._gh

        is_zone = (zm == zone_id)  # bool (H, W)
        zy, zx = np.where(is_zone)
        if len(zx) == 0:
            return []
        cx, cy = float(zx.mean()), float(zy.mean())

        # Count 4-connected neighbors in the target zone via padded slicing
        padded = np.zeros((gh + 2, gw + 2), dtype=np.int8)
        padded[1:-1, 1:-1] = is_zone
        n_count = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                   + padded[1:-1, :-2] + padded[1:-1, 2:])  # (H, W)

        base = (~is_zone
                & mask
                & ~wh.packagesInWarehouse.astype(bool))

        # Try compact (>=2) first; fall back to >=1 if nothing found
        for min_n in (2, 1):
            eligible = base & (n_count >= min_n)
            ey, ex = np.where(eligible)
            if len(ex) > 0:
                break
        if len(ex) == 0:
            return []

        is_neutral = (zm[ey, ex] == ZONE_NONE).astype(np.int8)  # 0 = neutral, 1 = other
        dist = (ex.astype(np.float32) - cx) ** 2 + (ey.astype(np.float32) - cy) ** 2
        # Sort: neutral first (0 before 1), then closest
        order = np.lexsort((dist, 1 - is_neutral))
        return list(zip(ex[order].tolist(), ey[order].tolist()))

    def _find_frontier_shrink(self, zone_id, wh):
        """Return cells to convert FROM *zone_id* to neutral, sorted best-first.

        Candidates: cells in *zone_id* at the frontier (≥1 neighbor not in zone),
        not occupied by a package. Sorted farthest-from-centroid first.

        Uses numpy padded slicing for the neighbor count instead of nested loops.
        """
        zm = wh.zoneMap
        gw, gh = self._gw, self._gh

        is_zone = (zm == zone_id)
        zy, zx = np.where(is_zone)
        if len(zx) == 0:
            return []
        cx, cy = float(zx.mean()), float(zy.mean())

        # Count 4-connected neighbors that ARE this zone (pad=0 → edges are "not zone")
        padded = np.zeros((gh + 2, gw + 2), dtype=np.int8)
        padded[1:-1, 1:-1] = is_zone
        n_same = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                  + padded[1:-1, :-2] + padded[1:-1, 2:])  # (H, W)

        # Frontier: in-zone, fewer than 4 same-zone neighbors, no package
        eligible = (is_zone
                    & ~wh.packagesInWarehouse.astype(bool)
                    & (n_same < 4))
        ey, ex = np.where(eligible)
        if len(ex) == 0:
            return []

        dist = (ex.astype(np.float32) - cx) ** 2 + (ey.astype(np.float32) - cy) ** 2
        order = np.argsort(-dist)   # farthest first
        return list(zip(ex[order].tolist(), ey[order].tolist()))

    # ── Apply ──────────────────────────────────────────────────────────

    @staticmethod
    def _apply(wh, cells, zone_id):
        """Mutate zoneMap and register pending changes for reconciliation."""
        for x, y in cells:
            wh.zoneMap[y, x] = zone_id
            wh.pending_zone_changes.add((x, y))

    # ── Zonable mask ───────────────────────────────────────────────────

    def _ensure_zonable(self, wh):
        """Lazily build or reuse the zonable mask."""
        if self._zonable is not None:
            return
        gw, gh = self._gw, self._gh
        mask = np.ones((gh, gw), dtype=bool)
        # Exclude charger spawn positions
        for coord in wh.chargerSpawnMap:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < gw and 0 <= y < gh:
                mask[y, x] = False
        # Exclude robot spawn positions
        for coord in wh.robotSpawnMap:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < gw and 0 <= y < gh:
                mask[y, x] = False
        # Exclude cells currently occupied by chargers or robots
        mask[wh.chargersInWarehouse > 0] = False
        mask[wh.robotsInWarehouse > 0]   = False
        self._zonable = mask
