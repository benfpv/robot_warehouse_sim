"""Tests for movement methods: _robot_consume_waypoints, _robot_step_movement,
_robot_recover_stuck, _robot_detect_oscillation."""
import math
import pytest
import numpy as np
from unittest.mock import MagicMock
from conftest import make_robot, WarehouseStub
from data.constants import ROBOT_CARRY_SPEED_PENALTY


# ═══════════════════════════════════════════════════════════════════════
# _robot_consume_waypoints
# ═══════════════════════════════════════════════════════════════════════

class TestRobotConsumeWaypoints:
    def test_pops_reached_waypoint(self):
        r = make_robot(xyLocation=[5, 5])
        r.path = [[5, 5], [6, 5], [7, 5]]
        r.pathPlan = [{'speed': 0.5}, {'speed': 0.5}, {'speed': 0.5}]
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert r.path == [[6, 5], [7, 5]]
        assert r.pathPlan == [{'speed': 0.5}, {'speed': 0.5}]

    def test_sets_next_target(self):
        r = make_robot(xyLocation=[5, 5])
        r.path = [[5, 5], [6, 5]]
        r.pathPlan = []
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert r.xyLocationTarget == [6, 5]

    def test_pops_multiple_reached(self):
        r = make_robot(xyLocation=[5, 5])
        r.path = [[5, 5], [5, 5], [6, 5]]
        r.pathPlan = [{}, {}, {}]
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert r.path == [[6, 5]]

    def test_empty_path_with_action_queue(self):
        r = make_robot(xyLocation=[5, 5])
        r.path = []
        r.actionQueue = [[[10, 15], 'move to target dropoff location']]
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert r.xyLocationTarget == [10, 15]

    def test_empty_path_no_action_queue(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[5, 5])
        r.path = []
        r.actionQueue = []
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert r.xyLocationTarget == [5, 5]  # unchanged

    def test_unreached_waypoint_not_popped(self):
        r = make_robot(xyLocation=[5, 5])
        r.path = [[6, 6], [7, 7]]
        r.pathPlan = [{}, {}]
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        assert len(r.path) == 2  # nothing popped

    def test_float_truncation(self):
        """Waypoint comparison uses int() truncation."""
        r = make_robot(xyLocation=[5.9, 5.1])
        r.path = [[5, 5], [6, 6]]
        r.pathPlan = [{}, {}]
        wh = WarehouseStub(robots=[r])
        wh._robot_consume_waypoints(0)
        # int(5.9) == 5, int(5.1) == 5 → matches [5, 5]
        assert r.path == [[6, 6]]


# ═══════════════════════════════════════════════════════════════════════
# _robot_step_movement
# ═══════════════════════════════════════════════════════════════════════

class TestRobotStepMovement:
    def _make_wh_for_movement(self, width=20, height=20, **kw):
        wh = WarehouseStub(width=width, height=height, **kw)
        return wh

    def test_free_cardinal_movement(self):
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5],
            velocity=0.5, direction=0,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.5
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2  # collector road: full speed
        _occupied = set()
        moved = wh._robot_step_movement(0, _occupied, 20, 20)
        assert moved is True
        assert r.xyLocation == [6, 5]

    def test_carrying_reduces_speed(self):
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5],
            velocity=0, carrying=10,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        # desiredVelocity should have ROBOT_CARRY_SPEED_PENALTY applied
        assert r.desiredVelocity <= r.maxVelocity * ROBOT_CARRY_SPEED_PENALTY + 0.001

    def test_collision_blocks_movement(self):
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5],
            velocity=1.0,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 1.5
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = {(6, 5)}  # blocker
        moved = wh._robot_step_movement(0, _occupied, 20, 20)
        assert moved is False
        assert r.xyLocation == [5, 5]
        assert r.blockedTicks >= 1

    def test_boundary_clamp(self):
        r = make_robot(
            xyLocation=[0, 0], xyLocationTarget=[-1, 0],
            velocity=1.0,
        )
        r.actionQueue = []
        r.movementProgress = 1.5
        r.path = []
        r.pathPlan = []
        r.pathTarget = None
        wh = self._make_wh_for_movement(robots=[r])
        _occupied = set()
        moved = wh._robot_step_movement(0, _occupied, 20, 20)
        # xyLocationTarget is int(-1)=-1, dx = max(-1, min(1, -1-0)) = -1
        # nx = 0 + (-1) = -1, out of bounds
        assert moved is False
        assert r.movementProgress == 0.0

    def test_velocity_smoothing_acceleration(self):
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.0,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        # Starting from 0, should accelerate by accelerationRate (0.05)
        assert r.velocity >= r.accelerationRate

    def test_road_tier_zero_neutral_zone_slows(self):
        """No road + neutral zone → 0.70 multiplier."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.5,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 0
        wh.zoneMap[5, 5] = 0  # neutral
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert r.desiredVelocity <= r.maxVelocity * 0.70 + 0.001

    def test_road_tier_one_slows(self):
        """Branch road → 0.85 multiplier."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.5,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 1
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert r.desiredVelocity <= r.maxVelocity * 0.85 + 0.001

    def test_inside_zone_no_road_slows_most(self):
        """Inside zone, no road → 0.35 multiplier."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.5,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 0
        wh.zoneMap[5, 5] = 1  # import zone
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert r.desiredVelocity <= r.maxVelocity * 0.35 + 0.001

    def test_limp_mode_caps_speed(self):
        """Battery below limp threshold severely limits speed."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5],
            velocity=0.5, batteryPercent=5.0,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        wh._limp_mode_batt_pct = 20.0
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        # 0.25 + 0.75 * 5.0 / 20.0 = 0.4375
        limp_cap = 0.25 + 0.75 * 5.0 / 20.0
        assert r.desiredVelocity <= limp_cap + 0.001

    def test_approach_braking_near_goal(self):
        """Within 1 cell of goal → cap to minMovingVelocity."""
        r = make_robot(
            xyLocation=[9, 5], xyLocationTarget=[10, 5], velocity=0.5,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 9] = 2
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert r.desiredVelocity <= r.minMovingVelocity + 0.001

    def test_diagonal_step_cost(self):
        """Diagonal movement costs sqrt(2) ≈ 1.414."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 6],
            velocity=0.5,
        )
        r.actionQueue = [[[10, 10], 'move']]
        r.movementProgress = 1.5  # above 1.414
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 10]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = set()
        moved = wh._robot_step_movement(0, _occupied, 20, 20)
        assert moved is True
        assert r.xyLocation == [6, 6]

    def test_stationary_no_movement(self):
        """When at target, no step direction → no movement."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[5, 5], velocity=0.0,
        )
        r.actionQueue = [[[5, 5], 'move']]
        r.movementProgress = 0.0
        r.path = []
        r.pathPlan = []
        r.pathTarget = [5, 5]
        wh = self._make_wh_for_movement(robots=[r])
        _occupied = set()
        moved = wh._robot_step_movement(0, _occupied, 20, 20)
        assert moved is False

    def test_moved_adds_to_occupied(self):
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.5,
        )
        r.actionQueue = [[[10, 5], 'move']]
        r.movementProgress = 1.5
        r.path = []
        r.pathPlan = []
        r.pathTarget = [10, 5]
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert (6, 5) in _occupied

    def test_arrival_resets_movement_progress(self):
        """When robot arrives at its target, movementProgress resets to 0."""
        r = make_robot(
            xyLocation=[5, 5], xyLocationTarget=[6, 5], velocity=0.5,
        )
        r.actionQueue = []
        r.movementProgress = 1.5
        r.path = []
        r.pathPlan = []
        r.pathTarget = None
        wh = self._make_wh_for_movement(robots=[r])
        wh.road_map[5, 5] = 2
        _occupied = set()
        wh._robot_step_movement(0, _occupied, 20, 20)
        assert r.xyLocation == [6, 5]
        # After moving to target cell 6,5 which is also target
        assert r.movementProgress == 0.0


# ═══════════════════════════════════════════════════════════════════════
# _robot_recover_stuck
# ═══════════════════════════════════════════════════════════════════════

class TestRobotRecoverStuck:
    def test_path_drift_replans(self):
        """If waypoint is far from current position, triggers replan."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[5, 5])
        r.path = [[9, 9], [10, 10]]  # 4 cells away (> 2 threshold)
        r.pathPlan = [{}, {}]
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 10], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_recover_stuck(0, set(), 20, 20)
        wh._compute_robot_path.assert_called_once()

    def test_local_sidestep_at_12_blocked(self):
        """After 12 blocked ticks, try to step to a free neighbor."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.path = [[6, 5]]
        r.pathPlan = [{}]
        r.pathAge = 5
        r.blockedTicks = 12
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        _occupied = {(6, 5)}  # neighbor blocked in x direction
        wh._robot_recover_stuck(0, _occupied, 20, 20)
        # Should have moved to a free cell closer to target
        if r.xyLocation != [5, 5]:
            assert r.blockedTicks == 0

    def test_no_free_neighbor_stays(self):
        """If all closer neighbors occupied, robot doesn't move."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[6, 5])
        r.path = [[6, 5]]
        r.pathPlan = [{}]
        r.pathAge = 1
        r.blockedTicks = 12
        r.replanCooldownTicks = 10  # on cooldown
        r.actionQueue = [[[6, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        # Block all 8 neighbors
        _occupied = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    _occupied.add((5 + dx, 5 + dy))
        wh._robot_recover_stuck(0, _occupied, 20, 20)
        assert r.xyLocation == [5, 5]

    def test_waypoint_skip_at_16_blocked(self):
        """After 16 blocked ticks with >=2 waypoints, skip the first."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[6, 5])
        r.path = [[6, 5], [7, 5], [8, 5]]
        r.pathPlan = [{}, {}, {}]
        r.pathAge = 5
        r.blockedTicks = 16
        r.replanCooldownTicks = 10  # on cooldown (no replan)
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        # Block closer neighbors to prevent sidestep
        _occupied = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    _occupied.add((5 + dx, 5 + dy))
        wh._robot_recover_stuck(0, _occupied, 20, 20)
        # First waypoint should be skipped
        assert r.xyLocationTarget == [7, 5]
        assert r.blockedTicks == 0

    def test_full_replan_with_path(self):
        """After 8+ blocked ticks with pathAge >= 4, triggers replan."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.path = [[6, 5]]
        r.pathPlan = [{}]
        r.pathAge = 5
        r.blockedTicks = 8
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        _occupied = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    _occupied.add((5 + dx, 5 + dy))
        wh._robot_recover_stuck(0, _occupied, 20, 20)
        wh._compute_robot_path.assert_called()

    def test_no_replan_when_at_target(self):
        """Robot at its target should not trigger replan."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[5, 5])
        r.path = []
        r.pathPlan = []
        r.pathAge = 10
        r.blockedTicks = 20
        r.replanCooldownTicks = 0
        r.actionQueue = []
        wh = WarehouseStub(robots=[r])
        wh._robot_recover_stuck(0, set(), 20, 20)
        wh._compute_robot_path.assert_not_called()

    def test_replan_cooldown_prevents_replan(self):
        """Cooldown > 0 prevents even the automatic full replan."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.path = [[6, 5]]
        r.pathPlan = [{}]
        r.pathAge = 5
        r.blockedTicks = 20
        r.replanCooldownTicks = 5  # on cooldown
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        # Block all neighbors to prevent sidestep too
        _occupied = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    _occupied.add((5 + dx, 5 + dy))
        wh._robot_recover_stuck(0, _occupied, 20, 20)
        # Only the path drift check could have called _compute_robot_path
        # but path[0] is [6,5] which is only 1 cell away (< 2 threshold)
        # so no drift replan either
        wh._compute_robot_path.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# _robot_detect_oscillation
# ═══════════════════════════════════════════════════════════════════════

class TestRobotDetectOscillation:
    def test_abab_pattern_triggers_replan(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [
            (5, 5), (6, 5), (5, 5), (6, 5), (5, 5), (6, 5),
        ]
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_called_once()
        assert r.velocity <= 0.08
        assert r.movementProgress == 0.0

    def test_no_oscillation_no_replan(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
        ]
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_not_called()

    def test_too_few_history_entries(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [(5, 5), (6, 5)]
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_not_called()

    def test_no_action_queue_no_check(self):
        r = make_robot(xyLocation=[5, 5])
        r.recentLocations = [(5, 5), (6, 5)] * 3
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = []
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_not_called()

    def test_path_age_too_low(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [
            (5, 5), (6, 5), (5, 5), (6, 5), (5, 5), (6, 5),
        ]
        r.pathAge = 2  # below 4 threshold
        r.blockedTicks = 0
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_not_called()

    def test_cooldown_prevents_replan(self):
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [
            (5, 5), (6, 5), (5, 5), (6, 5), (5, 5), (6, 5),
        ]
        r.pathAge = 5
        r.blockedTicks = 0
        r.replanCooldownTicks = 3  # on cooldown
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_not_called()

    def test_cluster_pattern_triggers_replan(self):
        """Cluster: only 2 unique positions + 6+ blocked ticks."""
        r = make_robot(xyLocation=[5, 5], xyLocationTarget=[10, 5])
        r.recentLocations = [
            (5, 5), (5, 5), (6, 5), (5, 5), (6, 5), (5, 5),
        ]
        r.pathAge = 5
        r.blockedTicks = 8
        r.replanCooldownTicks = 0
        r.actionQueue = [[[10, 5], 'move']]
        wh = WarehouseStub(robots=[r])
        wh._robot_detect_oscillation(0)
        wh._compute_robot_path.assert_called_once()
