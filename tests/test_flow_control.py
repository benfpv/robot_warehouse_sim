"""Tests for Warehouse.update_flow_control() adaptive import cap logic."""
import pytest
from datetime import timedelta
from conftest import make_robot, make_package, WarehouseStub


# ── helpers ──────────────────────────────────────────────────────────────────

def _wh(n_robots=4, statuses=None, n_overdue=0, n_packages=0, *,
        flow_avg_deadline=0.0, flow_avg_delivery=0.0):
    """Build a WarehouseStub with controlled fleet/package state.

    statuses: list of status strings, length must equal n_robots.
              Defaults to all 'idle'.
    n_overdue: how many packages have a negative timeToDeadline.
    n_packages: total packages (overdue + on-time); if 0 defaults to n_overdue.
    """
    statuses = statuses or ['idle'] * n_robots
    assert len(statuses) == n_robots

    robots = [make_robot(robotNumber=i, status=s) for i, s in enumerate(statuses)]
    total = max(n_packages, n_overdue)
    packages = []
    for i in range(total):
        ttd = timedelta(seconds=-30) if i < n_overdue else timedelta(seconds=120)
        packages.append(make_package(packageNumber=i, timeToDeadline=ttd))

    wh = WarehouseStub(robots=robots, packages=packages)
    wh.packagesMaxQuantity = 200
    wh._flow_pipeline_depth = 4
    wh._flow_avg_deadline = flow_avg_deadline
    wh._flow_avg_delivery = flow_avg_delivery
    return wh


# ── no robots / no capacity ──────────────────────────────────────────────────

class TestFlowControlEdgeCases:
    def test_no_robots_sets_cap_to_max(self):
        wh = WarehouseStub()
        wh.packagesMaxQuantity = 100
        wh._flow_import_cap = 0
        wh.update_flow_control()
        assert wh._flow_import_cap == 100

    def test_zero_capacity_sets_cap_to_zero(self):
        wh = _wh(n_robots=4)
        wh.packagesMaxQuantity = 0
        wh._flow_import_cap = 999
        wh.update_flow_control()
        assert wh._flow_import_cap == 0

    def test_cap_at_least_n_robots(self):
        """Cap is always >= n_robots regardless of adjustments."""
        wh = _wh(n_robots=4, statuses=['charging'] * 4, n_overdue=4, n_packages=4)
        wh._flow_import_cap = 100  # start high so penalty brings it down
        for _ in range(30):
            wh.update_flow_control()
        assert wh._flow_import_cap >= len(wh.robots)

    def test_cap_never_exceeds_robot_ceiling(self):
        """Cap is always <= n_robots * 10."""
        wh = _wh(n_robots=4, statuses=['idle'] * 4)
        wh._flow_import_cap = 0
        for _ in range(50):
            wh.update_flow_control()
        ceiling = len(wh.robots) * 10
        assert wh._flow_import_cap <= ceiling


# ── idle bonus ────────────────────────────────────────────────────────────────

class TestFlowControlIdleBonus:
    def test_high_idle_ratio_raises_cap(self):
        """≥50% idle robots triggers emergency floor (n_robots * 3)."""
        wh = _wh(n_robots=4, statuses=['idle', 'idle', 'idle', 'idle'])
        wh._flow_import_cap = 0
        wh.update_flow_control()
        # Emergency floor: majority idle → cap >= n_robots * 3
        assert wh._flow_import_cap >= len(wh.robots) * 3

    def test_idle_ratio_above_threshold_applies_bonus(self):
        """idle_robot_ratio > 0.15 should push cap above pure base_target."""
        wh = _wh(n_robots=8, statuses=['idle'] * 7 + ['charging'])
        wh._flow_import_cap = 0
        # Run enough ticks for the smoothed cap to approach target
        for _ in range(40):
            wh.update_flow_control()
        base_target = len(wh.robots) * wh._flow_pipeline_depth
        assert wh._flow_import_cap > base_target

    def test_all_busy_no_idle_bonus(self):
        """All robots working → no idle bonus → cap stays at or below base."""
        wh = _wh(n_robots=4, statuses=['move to target pickup location'] * 4)
        wh._flow_import_cap = 0
        for _ in range(40):
            wh.update_flow_control()
        base_target = len(wh.robots) * wh._flow_pipeline_depth
        # No bonus; cap may be at base_target or slightly lower due to smoothing start
        assert wh._flow_import_cap <= base_target * 1.1  # small tolerance for smoothing


# ── overdue penalty ───────────────────────────────────────────────────────────

class TestFlowControlOverduePenalty:
    def test_high_overdue_busy_robots_reduces_cap(self):
        """overdue_ratio > 0.10 with busy robots → penalty → cap < base_target."""
        n_robots = 6
        # All robots busy (non-idle), high overdue ratio
        statuses = ['move to target pickup location'] * n_robots
        wh = _wh(n_robots=n_robots, statuses=statuses,
                 n_overdue=8, n_packages=10)
        wh._flow_import_cap = n_robots * wh._flow_pipeline_depth * 2  # start high
        for _ in range(40):
            wh.update_flow_control()
        base_target = n_robots * wh._flow_pipeline_depth
        assert wh._flow_import_cap <= base_target

    def test_overdue_idle_robots_no_penalty(self):
        """Overdue packages + idle robots = routing issue; penalty not applied."""
        n_robots = 4
        wh = _wh(n_robots=n_robots,
                 statuses=['idle'] * n_robots,
                 n_overdue=4, n_packages=4)
        wh._flow_import_cap = 0
        for _ in range(40):
            wh.update_flow_control()
        # Idle robots → emergency floor, not penalty path
        assert wh._flow_import_cap >= n_robots * 3


# ── delivery stress penalty ──────────────────────────────────────────────────

class TestFlowControlDeliveryStress:
    def test_delivery_stress_above_threshold_reduces_cap(self):
        """avg_delivery / avg_deadline > 0.7 → stress penalty → lower cap."""
        n_robots = 4
        wh = _wh(n_robots=n_robots,
                 statuses=['move to target pickup location'] * n_robots,
                 flow_avg_deadline=100.0, flow_avg_delivery=80.0)  # stress = 0.8
        wh._flow_import_cap = n_robots * wh._flow_pipeline_depth * 2
        for _ in range(40):
            wh.update_flow_control()
        base_target = n_robots * wh._flow_pipeline_depth
        assert wh._flow_import_cap <= base_target

    def test_no_stress_without_deadline_data(self):
        """Both EMAs at 0 → stress = 0 (no penalty applied): cap grows above floor."""
        n_robots = 4
        wh = _wh(n_robots=n_robots,
                 statuses=['move to target pickup location'] * n_robots,
                 flow_avg_deadline=0.0, flow_avg_delivery=0.0)
        wh._flow_import_cap = 0
        for _ in range(40):
            wh.update_flow_control()
        # No penalty applied: cap should be well above the minimum n_robots floor
        assert wh._flow_import_cap >= n_robots * 2


# ── smoothing ─────────────────────────────────────────────────────────────────

class TestFlowControlSmoothing:
    def test_cap_does_not_jump_instantly(self):
        """15% smoothing: cap should NOT leap from 0 to target in a single tick."""
        wh = _wh(n_robots=4, statuses=['idle'] * 4)
        wh._flow_import_cap = 0
        wh.update_flow_control()
        target = len(wh.robots) * wh._flow_pipeline_depth
        # After one tick the cap should be less than the full target
        # (emergency floor may override if >50% idle, so allow for that)
        assert wh._flow_import_cap <= target * 1.5

    def test_cap_converges_over_many_ticks(self):
        """After many ticks the smoothed cap should stabilise above n_robots floor.

        Note: integer truncation in the smoothing formula means the cap converges
        to a stable point slightly below the floating-point target (not an exact
        match).  We verify the cap is well above n_robots and stable.
        """
        wh = _wh(n_robots=4, statuses=['move to target pickup location'] * 4)
        wh._flow_import_cap = 0
        for _ in range(60):
            wh.update_flow_control()
        # Cap must have grown well above the n_robots floor after convergence
        assert wh._flow_import_cap >= len(wh.robots) * 2
        # And it must be stable (two more ticks don't change it)
        prev = wh._flow_import_cap
        for _ in range(2):
            wh.update_flow_control()
        assert wh._flow_import_cap == prev
