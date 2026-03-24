"""ZoneStrategy — dynamically rebalances import/storage/export zones.

Monitors per-zone utilization via an exponential moving average, identifies
bottleneck (over-utilized) and donor (under-utilized or neutral) zones, then
incrementally grows/shrinks zone boundaries at their frontiers.

Mutations go through ``wh.pending_zone_changes`` so the existing
``reconcile_zone_changes()`` pipeline handles in-flight package safety.
"""
import numpy as np

# Zone constants (must match warehouse.py / paint_handler.py)
ZONE_NONE    = 0
ZONE_IMPORT  = 1
ZONE_STORAGE = 2
ZONE_EXPORT  = 3

_ZONE_IDS   = (ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT)
_ZONE_NAMES = {ZONE_NONE: 'neutral', ZONE_IMPORT: 'import',
               ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}

# 4-connected neighbor offsets
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class ZoneStrategy:
    """Pluggable strategy for the WarehouseOptimizer orchestrator.

    Public interface consumed by WarehouseOptimizer:
        name            — human-readable name
        label           — 1-char label for the UI toggle button
        enabled         — bool, toggled by UI
        EVAL_INTERVAL   — ticks between optimization passes
        step(wh, budget)— called by orchestrator each tick
        reset(wh)       — called on hot-reload
    """

    name  = 'Zone'
    label = 'Z'

    # ── Tuning constants ──────────────────────────────────────────────
    EVAL_INTERVAL   = 40      # ticks between passes (~1 sec at 40 tps)
    WARMUP_TICKS    = 200     # ~5 sec startup grace period
    EMA_ALPHA       = 0.15    # smoothing factor for utilization EMA
    HIGH_THRESH     = 0.80    # utilization above this = bottleneck
    LOW_THRESH      = 0.40    # utilization below this = donor candidate
    MIN_ZONE_CELLS  = 50      # never shrink a zone below this count
    COOLDOWN_MULTI  = 3       # cooldown = COOLDOWN_MULTI × EVAL_INTERVAL ticks

    # Target slot proportions (used as tiebreaker bias, not hard constraint)
    _TARGET_RATIOS  = {ZONE_IMPORT: 0.20, ZONE_STORAGE: 0.45, ZONE_EXPORT: 0.35}

    def __init__(self, warehouse_res):
        self.enabled = False
        self._gw, self._gh = warehouse_res
        self._tick      = 0       # ticks since last eval
        self._total     = 0       # total ticks since enable (for warmup)
        self._util_ema  = {z: 0.0 for z in _ZONE_IDS}
        self._cooldown  = {z: 0   for z in _ZONE_IDS}   # remaining cooldown ticks
        self._zonable   = None    # bool mask (H, W), computed lazily

    # ── Orchestrator interface ─────────────────────────────────────────

    def step(self, wh, budget):
        """Called every tick by the orchestrator when enabled."""
        if not self.enabled:
            return
        self._total += 1
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
                parts.append('+{} {}'.format(n_grow, _ZONE_NAMES[grow_zone]))
            if n_shrink:
                parts.append('-{} {}'.format(n_shrink, _ZONE_NAMES[shrink_zone]))
            # Set cooldown on touched zones
            if grow_zone is not None:
                self._cooldown[grow_zone] = self.COOLDOWN_MULTI * self.EVAL_INTERVAL
            if shrink_zone is not None and shrink_zone != ZONE_NONE:
                self._cooldown[shrink_zone] = self.COOLDOWN_MULTI * self.EVAL_INTERVAL
            print('[ZoneStrategy] {}'.format(', '.join(parts)))

    def reset(self, wh):
        """Reset internal state (called on hot-reload)."""
        self._tick     = 0
        self._total    = 0
        self._util_ema = {z: 0.0 for z in _ZONE_IDS}
        self._cooldown = {z: 0   for z in _ZONE_IDS}
        self._zonable  = None     # recomputed lazily from new spawn maps

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
        intensity = max(1, min(int(delta * 50), 8))

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

        Candidates: cells not already *zone_id*, with ≥2 of 4 neighbors already
        in *zone_id* (compactness), inside zonable mask, not occupied.
        Prefer neutral (zone 0) cells over cells from another zone.
        Sorted by distance to zone centroid (closest first).
        """
        zm = wh.zoneMap
        mask = self._zonable
        gw, gh = self._gw, self._gh

        # Zone centroid
        zy, zx = np.where(zm == zone_id)
        if len(zx) == 0:
            return []
        cx, cy = float(zx.mean()), float(zy.mean())

        candidates = []
        for y in range(gh):
            for x in range(gw):
                if zm[y, x] == zone_id:
                    continue
                if not mask[y, x]:
                    continue
                if wh.packagesInWarehouse[y, x]:
                    continue
                # Count neighbors in target zone
                n_in = 0
                for dx, dy in _NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gw and 0 <= ny < gh and zm[ny, nx] == zone_id:
                        n_in += 1
                if n_in < 2:
                    continue
                is_neutral = (zm[y, x] == ZONE_NONE)
                dist = (x - cx) ** 2 + (y - cy) ** 2
                # Sort key: prefer neutral (0) over occupied-zone (1), then by distance
                candidates.append((0 if is_neutral else 1, dist, x, y))

        candidates.sort()
        return [(c[2], c[3]) for c in candidates]

    def _find_frontier_shrink(self, zone_id, wh):
        """Return cells to convert FROM *zone_id* to neutral, sorted best-first.

        Candidates: cells in *zone_id* at the frontier (≥1 neighbor not in zone),
        not occupied by a package. Sorted farthest-from-centroid first.
        """
        zm = wh.zoneMap
        gw, gh = self._gw, self._gh

        zy, zx = np.where(zm == zone_id)
        if len(zx) == 0:
            return []
        cx, cy = float(zx.mean()), float(zy.mean())

        candidates = []
        for y in range(gh):
            for x in range(gw):
                if zm[y, x] != zone_id:
                    continue
                if wh.packagesInWarehouse[y, x]:
                    continue
                # Must be at frontier (at least one neighbor is NOT this zone)
                at_edge = False
                for dx, dy in _NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < gw and 0 <= ny < gh) or zm[ny, nx] != zone_id:
                        at_edge = True
                        break
                if not at_edge:
                    continue
                dist = (x - cx) ** 2 + (y - cy) ** 2
                candidates.append((-dist, x, y))   # negative → farthest first

        candidates.sort()
        return [(c[1], c[2]) for c in candidates]

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
        # Exclude cells currently occupied by chargers
        mask[wh.chargersInWarehouse > 0] = False
        self._zonable = mask
