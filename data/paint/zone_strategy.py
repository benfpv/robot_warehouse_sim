"""ZoneStrategy — dynamically rebalances import/storage/export zones.

Monitors per-zone utilization via an exponential moving average, identifies
bottleneck (over-utilized) and donor (under-utilized) zones, then
incrementally converts donor-zone boundary cells into the bottleneck zone.

Zone swaps only: cells change from one zone type to another — no zone is
ever eliminated (MIN_ZONE_CELLS guard) and no new zone cells are created
from neutral space.  The total zoned footprint is invariant.

Mutations go through ``wh.pending_zone_changes`` so the existing
``reconcile_zone_changes()`` pipeline handles in-flight package safety.
"""
import time
import numpy as np

from data.constants import ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT

_ZONE_IDS   = (ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT)
# Short abbreviations used in the UI stats line where space is tight
_ZONE_ABBR  = {ZONE_NONE: 'clr', ZONE_IMPORT: 'imp',
               ZONE_STORAGE: 'sto', ZONE_EXPORT: 'exp'}


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

    def step(self, wh, budget, heatmap_slow=None):
        """Called every tick by the orchestrator when enabled.

        Args:
            heatmap_slow: float32 (H×W) slow-decay traffic heatmap passed through
                          from the orchestrator.  Reserved for future road-logic use.
        """
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
        # Zone-swap: find donor frontier cells adjacent to the bottleneck zone
        # and convert them directly (donor → bottleneck).  No neutral intermediary.
        if shrink_zone is None or shrink_zone == ZONE_NONE:
            return   # no valid donor — skip
        cells = self._find_swap_frontier(shrink_zone, grow_zone, wh)[:intensity]
        if not cells:
            return
        self._apply(wh, cells, grow_zone)
        n_swapped = len(cells)
        self._last_action_str  = '{} {} -> {}'.format(
            n_swapped, _ZONE_ABBR[shrink_zone], _ZONE_ABBR[grow_zone])
        self._last_action_time = time.time()
        # Cool down the donor so it isn't drained continuously
        self._cooldown[shrink_zone] = self.COOLDOWN_MULTI * self.EVAL_INTERVAL

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
        """Determine which zone to grow (bottleneck) and which to shrink (donor).

        Only zone-to-zone swaps are permitted — no zone may be eliminated and
        no new zones are created from neutral space.  Returns
        ``(grow_zone_id, shrink_zone_id, intensity)`` or ``None``.
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

        # Find donor: lowest utilization below threshold, not on cooldown,
        # and still above MIN_ZONE_CELLS so it cannot be eliminated.
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

        if donor is None:
            return None   # no valid donor — do nothing

        return (bottleneck, donor, intensity)

    # ── Frontier selection ─────────────────────────────────────────────

    def _find_swap_frontier(self, donor_zone, target_zone, wh):
        """Return donor-zone cells on the shared boundary with *target_zone*.

        Only cells that are:
          - in *donor_zone*
          - adjacent (4-connected) to at least one *target_zone* cell
          - not occupied by a package
          - inside the zonable mask
        are eligible.  Sorted farthest-from-donor-centroid first (peel from
        outside in to keep the donor compact).
        """
        zm = wh.zoneMap
        mask = self._zonable
        gw, gh = self._gw, self._gh

        is_donor  = (zm == donor_zone)
        is_target = (zm == target_zone)

        # Centroid of donor for distance sorting
        dy, dx = np.where(is_donor)
        if len(dx) == 0:
            return []
        cx, cy = float(dx.mean()), float(dy.mean())

        # Count 4-connected neighbours that belong to target zone
        padded = np.zeros((gh + 2, gw + 2), dtype=np.int8)
        padded[1:-1, 1:-1] = is_target
        n_target = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                    + padded[1:-1, :-2] + padded[1:-1, 2:])  # (H, W)

        eligible = (is_donor
                    & mask
                    & (n_target >= 1)
                    & ~wh.packagesInWarehouse.astype(bool))
        ey, ex = np.where(eligible)
        if len(ex) == 0:
            return []

        dist = (ex.astype(np.float32) - cx) ** 2 + (ey.astype(np.float32) - cy) ** 2
        order = np.argsort(-dist)   # farthest from donor centroid first
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
