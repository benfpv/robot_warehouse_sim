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

    MAX_CELLS_PER_SEC = 16

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

    def step(self, wh):
        """Called once per sim tick.  Forwards to each enabled strategy."""
        active = [s for s in self.strategies if s.enabled]
        if not active:
            return
        # Budget: spread the per-second allowance evenly across active
        # strategies, scaled by each strategy's own EVAL_INTERVAL.
        for s in active:
            per_eval = max(1, (self.MAX_CELLS_PER_SEC * s.EVAL_INTERVAL)
                              // max(self._tick_rate, 1)
                              // max(len(active), 1))
            s.step(wh, per_eval)

    # ── Reset (hot-reload) ─────────────────────────────────────────────

    def reset(self, wh):
        """Reset every strategy (called on L-key map hot-reload)."""
        for s in self.strategies:
            s.reset(wh)
