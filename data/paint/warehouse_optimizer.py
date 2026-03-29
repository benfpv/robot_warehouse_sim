"""Orchestrator for pluggable warehouse-optimization strategies.

Holds a registry of strategies (e.g. ZoneStrategy, future ChargerStrategy,
RobotStrategy).  Each strategy can be independently enabled/disabled via the UI.
The orchestrator dispatches ``step(wh)`` once per sim tick (called from main.py
*before* ``warehouse.update_warehouse()``), distributing a per-second cell-change
budget across active strategies.
"""


class WarehouseOptimizer:
    """Central dispatcher for optimization strategies.

    Attributes:
        strategies: list of strategy instances (each has *name*, *label*,
                    *enabled*, *step(wh, budget)*, *reset(wh)*).
        MAX_CELLS_PER_SEC: global cap on zone-map mutations per second across
                           all strategies combined.
    """

    MAX_CELLS_PER_SEC = 40

    def __init__(self, tick_rate=40):
        """
        Args:
            tick_rate: expected sim ticks per second (used to convert the
                       per-second budget into a per-evaluation budget).
        """
        self.strategies = []
        self._tick_rate = tick_rate

    # ── Registry ───────────────────────────────────────────────────────

    def register(self, strategy):
        """Add a strategy to the dispatch list."""
        self.strategies.append(strategy)

    # ── Per-tick entry point ───────────────────────────────────────────

    def step(self, wh, heatmap_slow=None):
        """Called once per sim tick.  Forwards to all registered strategies.

        Disabled strategies still receive the call so they can advance internal
        counters (e.g. warmup ticker) — they return immediately without mutating
        warehouse state.  Budget is calculated against the active-strategy count
        so enabling a strategy gives it the correct share of the global cap.

        Args:
            heatmap_slow: float32 (H×W) slow-decay traffic heatmap (~5 min half-life).
                          Passed through to strategies for future road-logic use.
        """
        active   = [s for s in self.strategies if s.enabled]
        n_active = max(len(active), 1)
        for s in self.strategies:
            per_eval = max(1, (self.MAX_CELLS_PER_SEC * s.EVAL_INTERVAL)
                              // max(self._tick_rate, 1)
                              // n_active)
            s.step(wh, per_eval, heatmap_slow=heatmap_slow)

    # ── Reset (hot-reload) ─────────────────────────────────────────────

    def reset(self, wh):
        """Reset every strategy (called on L-key map hot-reload)."""
        for s in self.strategies:
            s.reset(wh)
