"""Tests for static/helper methods: _robot_is_stopped, _estimate_charge_travel_cost,
_find_nearest_empty_cell, _normalize_deg, _rotate_toward_deg."""
import pytest
import numpy as np
from conftest import make_robot, WarehouseStub


# ═══════════════════════════════════════════════════════════════════════
# _robot_is_stopped  (static method)
# ═══════════════════════════════════════════════════════════════════════

class TestRobotIsStopped:
    def test_zero_velocity_zero_progress(self):
        r = make_robot(velocity=0)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r) is True

    def test_high_velocity(self):
        r = make_robot(velocity=0.5)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r) is False

    def test_high_progress(self):
        r = make_robot(velocity=0)
        r.movementProgress = 0.5
        assert WarehouseStub._robot_is_stopped(r) is False

    def test_at_epsilon_boundary_velocity(self):
        r = make_robot(velocity=0.02)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r) is True

    def test_at_epsilon_boundary_progress(self):
        r = make_robot(velocity=0)
        r.movementProgress = 0.05
        assert WarehouseStub._robot_is_stopped(r) is True

    def test_both_at_epsilon(self):
        r = make_robot(velocity=0.02)
        r.movementProgress = 0.05
        assert WarehouseStub._robot_is_stopped(r) is True

    def test_just_above_velocity_epsilon(self):
        r = make_robot(velocity=0.021)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r) is False

    def test_just_above_progress_epsilon(self):
        r = make_robot(velocity=0)
        r.movementProgress = 0.051
        assert WarehouseStub._robot_is_stopped(r) is False

    def test_negative_velocity(self):
        r = make_robot(velocity=-0.01)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r) is True

    def test_custom_eps(self):
        r = make_robot(velocity=0.1)
        r.movementProgress = 0.0
        assert WarehouseStub._robot_is_stopped(r, vel_eps=0.1) is True
        assert WarehouseStub._robot_is_stopped(r, vel_eps=0.05) is False


# ═══════════════════════════════════════════════════════════════════════
# _estimate_charge_travel_cost  (static method)
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateChargeTravelCost:
    def test_zero_distance(self):
        r = make_robot(xyLocation=[5, 5], batteryDepletingRate=0.1, carrying=-1)
        cost = WarehouseStub._estimate_charge_travel_cost(r, [5, 5])
        assert cost == 0.0

    def test_horizontal_distance(self):
        r = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.1, carrying=-1)
        cost = WarehouseStub._estimate_charge_travel_cost(r, [10, 0])
        # Chebyshev distance = 10, drain = 0.1 * 1.0, cost = 0.1 * 10 * 1.5
        assert cost == pytest.approx(1.5)

    def test_diagonal_distance(self):
        r = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.1, carrying=-1)
        cost = WarehouseStub._estimate_charge_travel_cost(r, [10, 10])
        # Chebyshev = 10
        assert cost == pytest.approx(1.5)

    def test_carrying_increases_cost(self):
        r_empty = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.1, carrying=-1)
        r_carry = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.1, carrying=5)
        cost_empty = WarehouseStub._estimate_charge_travel_cost(r_empty, [10, 0])
        cost_carry = WarehouseStub._estimate_charge_travel_cost(r_carry, [10, 0])
        # Carrying multiplier is 1.5
        assert cost_carry == pytest.approx(cost_empty * 1.5)

    def test_larger_depletion_rate(self):
        r = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.2, carrying=-1)
        cost = WarehouseStub._estimate_charge_travel_cost(r, [10, 0])
        assert cost == pytest.approx(3.0)

    def test_asymmetric_distance(self):
        r = make_robot(xyLocation=[0, 0], batteryDepletingRate=0.1, carrying=-1)
        cost = WarehouseStub._estimate_charge_travel_cost(r, [3, 7])
        # Chebyshev = max(3, 7) = 7
        assert cost == pytest.approx(0.1 * 7 * 1.5)

    def test_cost_always_non_negative(self):
        r = make_robot(xyLocation=[5, 5], batteryDepletingRate=0.1, carrying=-1)
        for loc in ([0, 0], [10, 10], [5, 5], [0, 10]):
            cost = WarehouseStub._estimate_charge_travel_cost(r, loc)
            assert cost >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# _find_nearest_empty_cell
# ═══════════════════════════════════════════════════════════════════════

class TestFindNearestEmptyCell:
    def test_origin_is_empty(self):
        wh = WarehouseStub(width=10, height=10)
        result = wh._find_nearest_empty_cell(5, 5)
        assert result == [5, 5]

    def test_origin_occupied_finds_neighbor(self):
        wh = WarehouseStub(width=10, height=10)
        wh.packagesInWarehouse[5][5] = 1
        result = wh._find_nearest_empty_cell(5, 5)
        # Must be adjacent (BFS cardinal neighbors)
        dx = abs(result[0] - 5)
        dy = abs(result[1] - 5)
        assert dx + dy == 1

    def test_charger_blocks(self):
        wh = WarehouseStub(width=10, height=10)
        wh.chargersInWarehouse[5][5] = 1
        result = wh._find_nearest_empty_cell(5, 5)
        assert result != [5, 5]

    def test_robot_blocks(self):
        wh = WarehouseStub(width=10, height=10)
        wh.robotsInWarehouse[5][5] = 1
        result = wh._find_nearest_empty_cell(5, 5)
        assert result != [5, 5]

    def test_multiple_occupied_finds_deeper(self):
        wh = WarehouseStub(width=10, height=10)
        # Occupy center and immediate cardinal neighbors
        for x, y in [(5, 5), (5, 4), (5, 6), (4, 5), (6, 5)]:
            wh.packagesInWarehouse[y][x] = 1
        result = wh._find_nearest_empty_cell(5, 5)
        # Must be 2 steps away
        dist = abs(result[0] - 5) + abs(result[1] - 5)
        assert dist == 2

    def test_all_occupied_returns_origin(self):
        """Fallback when every cell is occupied."""
        wh = WarehouseStub(width=3, height=3)
        wh.packagesInWarehouse[:] = 1
        result = wh._find_nearest_empty_cell(1, 1)
        assert result == [1, 1]

    def test_corner_origin(self):
        wh = WarehouseStub(width=5, height=5)
        wh.packagesInWarehouse[0][0] = 1
        result = wh._find_nearest_empty_cell(0, 0)
        assert 0 <= result[0] < 5
        assert 0 <= result[1] < 5

    def test_result_not_blocked_by_any_entity(self):
        wh = WarehouseStub(width=10, height=10)
        wh.packagesInWarehouse[5][5] = 1
        wh.robotsInWarehouse[5][4] = 1
        wh.chargersInWarehouse[4][5] = 1
        result = wh._find_nearest_empty_cell(5, 5)
        x, y = result
        assert wh.packagesInWarehouse[y][x] == 0
        assert wh.robotsInWarehouse[y][x] == 0
        assert wh.chargersInWarehouse[y][x] == 0


# ═══════════════════════════════════════════════════════════════════════
# _normalize_deg  (static method)
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizeDeg:
    def test_zero(self):
        assert WarehouseStub._normalize_deg(0) == 0.0

    def test_positive_within_range(self):
        assert WarehouseStub._normalize_deg(90) == 90.0

    def test_negative_within_range(self):
        assert WarehouseStub._normalize_deg(-90) == -90.0

    def test_at_180(self):
        # 180 wraps to -180
        assert WarehouseStub._normalize_deg(180) == pytest.approx(-180.0)

    def test_at_minus_180(self):
        assert WarehouseStub._normalize_deg(-180) == pytest.approx(-180.0)

    def test_360_wraps_to_zero(self):
        assert WarehouseStub._normalize_deg(360) == pytest.approx(0.0)

    def test_270_wraps_to_minus_90(self):
        assert WarehouseStub._normalize_deg(270) == pytest.approx(-90.0)

    def test_minus_270_wraps_to_90(self):
        assert WarehouseStub._normalize_deg(-270) == pytest.approx(90.0)

    def test_large_positive(self):
        assert WarehouseStub._normalize_deg(720) == pytest.approx(0.0)

    def test_large_negative(self):
        assert WarehouseStub._normalize_deg(-720) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# _rotate_toward_deg
# ═══════════════════════════════════════════════════════════════════════

class TestRotateTowardDeg:
    def test_target_within_step(self):
        wh = WarehouseStub()
        result = wh._rotate_toward_deg(0.0, 5.0, 10.0)
        assert result == 5.0

    def test_target_exactly_step(self):
        wh = WarehouseStub()
        result = wh._rotate_toward_deg(0.0, 10.0, 10.0)
        assert result == 10.0

    def test_target_beyond_step_positive(self):
        wh = WarehouseStub()
        result = wh._rotate_toward_deg(0.0, 30.0, 10.0)
        assert result == 10.0

    def test_target_beyond_step_negative(self):
        wh = WarehouseStub()
        result = wh._rotate_toward_deg(0.0, -30.0, 10.0)
        assert result == -10.0

    def test_already_at_target(self):
        wh = WarehouseStub()
        result = wh._rotate_toward_deg(45.0, 45.0, 10.0)
        assert result == 45.0
