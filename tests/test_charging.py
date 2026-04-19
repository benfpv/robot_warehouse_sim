"""Tests for charging-related methods: _reconcile_charger_reservations,
_update_charge_policy_metrics, _robot_dock_charger, _robot_charge_tick,
_dispatch_charging_if_needed, _emergency_battery_drop."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from conftest import make_robot, make_package, make_charger, WarehouseStub
from data.constants import STALL_TICKS, BATTERY_FULL_PCT


# ═══════════════════════════════════════════════════════════════════════
# _reconcile_charger_reservations
# ═══════════════════════════════════════════════════════════════════════

class TestReconcileChargerReservations:
    def test_orphan_charger_freed(self):
        """Charger in 'charging planned' with no robot heading there → freed."""
        c = make_charger(xyLocation=[5, 5], status='charging planned')
        wh = WarehouseStub(chargers=[c])
        wh._reconcile_charger_reservations()
        assert c.status == 'idle'

    def test_active_reservation_kept(self):
        """Charger claimed by a robot's actionQueue stays 'charging planned'."""
        c = make_charger(xyLocation=[5, 5], status='charging planned')
        r = make_robot(actionQueue=[[[5, 5], 'move to charging station']])
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._reconcile_charger_reservations()
        assert c.status == 'charging planned'

    def test_charging_task_also_claims(self):
        """A 'charging' task in the queue also keeps the charger claimed."""
        c = make_charger(xyLocation=[5, 5], status='charging planned')
        r = make_robot(actionQueue=[[[5, 5], 'charging']])
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._reconcile_charger_reservations()
        assert c.status == 'charging planned'

    def test_idle_charger_untouched(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        wh = WarehouseStub(chargers=[c])
        wh._reconcile_charger_reservations()
        assert c.status == 'idle'

    def test_charging_status_untouched(self):
        """A charger actively charging is never freed."""
        c = make_charger(xyLocation=[5, 5], status='charging')
        wh = WarehouseStub(chargers=[c])
        wh._reconcile_charger_reservations()
        assert c.status == 'charging'

    def test_multiple_chargers_mixed(self):
        c0 = make_charger(chargerNumber=0, xyLocation=[5, 5], status='charging planned')
        c1 = make_charger(chargerNumber=1, xyLocation=[10, 10], status='charging planned')
        r = make_robot(actionQueue=[[[5, 5], 'move to charging station']])
        wh = WarehouseStub(robots=[r], chargers=[c0, c1])
        wh._reconcile_charger_reservations()
        assert c0.status == 'charging planned'  # claimed
        assert c1.status == 'idle'  # orphan freed

    def test_multiple_robots_claim_different_chargers(self):
        c0 = make_charger(chargerNumber=0, xyLocation=[5, 5], status='charging planned')
        c1 = make_charger(chargerNumber=1, xyLocation=[10, 10], status='charging planned')
        r0 = make_robot(robotNumber=0, actionQueue=[[[5, 5], 'move to charging station']])
        r1 = make_robot(robotNumber=1, actionQueue=[[[10, 10], 'move to charging station']])
        wh = WarehouseStub(robots=[r0, r1], chargers=[c0, c1])
        wh._reconcile_charger_reservations()
        assert c0.status == 'charging planned'
        assert c1.status == 'charging planned'


# ═══════════════════════════════════════════════════════════════════════
# _update_charge_policy_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateChargePolicyMetrics:
    def test_all_chargers_idle(self):
        chargers = [make_charger(status='idle') for _ in range(4)]
        robots = [make_robot(batteryPercent=80.0) for _ in range(4)]
        wh = WarehouseStub(chargers=chargers, robots=robots)
        wh._update_charge_policy_metrics()
        assert wh._charger_pressure_raw == 0.0

    def test_all_chargers_busy(self):
        chargers = [make_charger(status='charging') for _ in range(4)]
        wh = WarehouseStub(chargers=chargers, robots=[make_robot()])
        wh._update_charge_policy_metrics()
        assert wh._charger_pressure_raw == 1.0

    def test_half_busy(self):
        chargers = [
            make_charger(chargerNumber=0, status='idle'),
            make_charger(chargerNumber=1, status='charging'),
        ]
        wh = WarehouseStub(chargers=chargers, robots=[make_robot()])
        wh._update_charge_policy_metrics()
        assert wh._charger_pressure_raw == 0.5

    def test_avg_fleet_battery(self):
        robots = [
            make_robot(robotNumber=0, batteryPercent=60.0),
            make_robot(robotNumber=1, batteryPercent=80.0),
        ]
        wh = WarehouseStub(chargers=[make_charger()], robots=robots)
        wh._update_charge_policy_metrics()
        assert wh._avg_fleet_battery == pytest.approx(70.0)

    def test_threshold_target_at_zero_pressure(self):
        wh = WarehouseStub(chargers=[make_charger(status='idle')],
                           robots=[make_robot()])
        wh._update_charge_policy_metrics()
        # target = base + span * 0 = 35
        assert wh._charge_threshold_target == pytest.approx(35.0)

    def test_threshold_target_at_full_pressure(self):
        wh = WarehouseStub(chargers=[make_charger(status='charging')],
                           robots=[make_robot()])
        wh._update_charge_policy_metrics()
        # target = base + span * 1 = 35 + 20 = 55
        assert wh._charge_threshold_target == pytest.approx(55.0)

    def test_threshold_smoothing_moves_toward_target(self):
        wh = WarehouseStub(chargers=[make_charger(status='charging')],
                           robots=[make_robot()])
        wh._charge_threshold = 35.0  # start at base
        wh._update_charge_policy_metrics()
        # Should move toward 55.0
        assert wh._charge_threshold > 35.0
        assert wh._charge_threshold < 55.0

    def test_threshold_clamped_to_range(self):
        wh = WarehouseStub(chargers=[make_charger(status='idle')],
                           robots=[make_robot()])
        wh._charge_threshold = 20.0  # below base
        wh._update_charge_policy_metrics()
        assert wh._charge_threshold >= wh._charge_threshold_base

    def test_no_robots_no_crash(self):
        wh = WarehouseStub(chargers=[make_charger()])
        wh._update_charge_policy_metrics()
        assert wh._avg_fleet_battery == 0.0  # sum/max(0,1)=0

    def test_pressure_ema_smoothing(self):
        wh = WarehouseStub(chargers=[make_charger(status='charging')],
                           robots=[make_robot()])
        wh._charger_pressure = 0.0
        wh._update_charge_policy_metrics()
        # pressure += alpha * (raw - pressure) = 0.025 * (1.0 - 0.0)
        assert wh._charger_pressure == pytest.approx(0.025)


# ═══════════════════════════════════════════════════════════════════════
# _robot_dock_charger
# ═══════════════════════════════════════════════════════════════════════

class TestRobotDockCharger:
    def test_successful_dock(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            xyLocation=[5, 5], velocity=0, status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        r.movementProgress = 0.0
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_dock_charger(0)
        assert c.status == 'charging'
        assert r.stallTicks == STALL_TICKS
        assert r.hasCharged is True
        wh.robot_replace_task.assert_called_once()

    def test_not_at_charger_location(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            xyLocation=[3, 3], velocity=0, status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        r.movementProgress = 0.0
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_dock_charger(0)
        assert c.status == 'idle'
        wh.robot_replace_task.assert_not_called()

    def test_robot_not_stopped(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            xyLocation=[5, 5], velocity=0.5, status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        r.movementProgress = 0.5
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_dock_charger(0)
        assert c.status == 'idle'
        assert r.desiredVelocity == 0.0

    def test_charger_already_charging_pops_task(self):
        """If another robot beat us, we pop our task and let re-assignment happen."""
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], velocity=0, status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        r.movementProgress = 0.0
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_dock_charger(0)
        wh.robot_pop_task.assert_called_once()
        assert c.status == 'charging'  # unchanged

    def test_charger_not_found(self):
        """If charger was removed, method returns safely."""
        r = make_robot(
            xyLocation=[5, 5], velocity=0,
            actionQueue=[[[99, 99], 'move to charging station']],
        )
        wh = WarehouseStub(robots=[r])
        wh._robot_dock_charger(0)  # should not raise

    def test_another_robot_on_charger_cell(self):
        """If another robot occupies the charger cell, don't dock."""
        c = make_charger(xyLocation=[5, 5], status='idle')
        r0 = make_robot(
            robotNumber=0, xyLocation=[5, 5], velocity=0,
            status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        r0.movementProgress = 0.0
        r1 = make_robot(robotNumber=1, xyLocation=[5, 5])
        wh = WarehouseStub(robots=[r0, r1], chargers=[c])
        wh._robot_dock_charger(0)
        assert c.status == 'idle'  # not docked


# ═══════════════════════════════════════════════════════════════════════
# _robot_charge_tick
# ═══════════════════════════════════════════════════════════════════════

class TestRobotChargeTick:
    def test_battery_increases(self):
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=50.0, batteryChargingRate=1.0,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_charge_tick(0)
        assert r.batteryPercent == 51.0

    def test_undock_when_full(self):
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=BATTERY_FULL_PCT + 0.1,
            batteryChargingRate=1.0,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_charge_tick(0)
        assert c.status == 'idle'
        assert r.stallTicks == STALL_TICKS
        wh.robot_pop_task.assert_called_once()

    def test_charges_up_to_full_pct(self):
        """Keeps charging at exactly BATTERY_FULL_PCT."""
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=float(BATTERY_FULL_PCT),
            batteryChargingRate=1.0,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_charge_tick(0)
        # At exactly BATTERY_FULL_PCT, condition is <= so it charges
        assert r.batteryPercent == float(BATTERY_FULL_PCT) + 1.0

    def test_not_at_charger_skips(self):
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[3, 3], batteryPercent=50.0,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_charge_tick(0)
        assert r.batteryPercent == 50.0  # unchanged

    def test_charger_removed_skips(self):
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=50.0,
            actionQueue=[[[99, 99], 'charging']],
        )
        wh = WarehouseStub(robots=[r])
        wh._robot_charge_tick(0)  # should not raise
        assert r.batteryPercent == 50.0

    def test_charging_rate_applies(self):
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=80.0, batteryChargingRate=1.5,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._robot_charge_tick(0)
        assert r.batteryPercent == 81.5

    def test_multiple_ticks_to_full(self):
        c = make_charger(xyLocation=[5, 5], status='charging')
        r = make_robot(
            xyLocation=[5, 5], batteryPercent=95.0, batteryChargingRate=1.0,
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        ticks = 0
        while ticks < 100:
            wh._robot_charge_tick(0)
            ticks += 1
            if c.status == 'idle':
                break
        # Should have undocked after exceeding BATTERY_FULL_PCT
        assert c.status == 'idle'
        assert ticks <= 5


# ═══════════════════════════════════════════════════════════════════════
# _dispatch_charging_if_needed
# ═══════════════════════════════════════════════════════════════════════

class TestDispatchChargingIfNeeded:
    def test_high_battery_no_dispatch(self):
        r = make_robot(batteryPercent=100.0, status='idle')
        wh = WarehouseStub(robots=[r], chargers=[make_charger()])
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()

    def test_low_battery_dispatches(self):
        c = make_charger(chargerNumber=0, xyLocation=[5, 5], status='idle')
        r = make_robot(
            robotNumber=0, batteryPercent=10.0, status='idle',
            xyLocation=[10, 10],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_called_once()
        assert c.status == 'charging planned'

    def test_already_charging_no_duplicate(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            batteryPercent=10.0, status='charging',
            actionQueue=[[[5, 5], 'charging']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()

    def test_already_queued_no_duplicate(self):
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            batteryPercent=10.0, status='move to charging station',
            actionQueue=[[[5, 5], 'move to charging station']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()

    def test_no_charger_available(self):
        r = make_robot(batteryPercent=10.0, status='idle')
        wh = WarehouseStub(robots=[r])
        wh.find_available_charger = MagicMock(return_value=(False, -1))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()

    def test_dropoff_status_blocks_dispatch(self):
        """Robot en route to dropoff ('move to target dropoff location') is blocked."""
        c = make_charger(xyLocation=[5, 5], status='idle')
        r = make_robot(
            batteryPercent=10.0, status='move to target dropoff location',
            actionQueue=[[[20, 20], 'move to target dropoff location']],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()

    def test_opportunistic_topoff_idle_low_pressure(self):
        """Idle robot at <=70% with low charger pressure → opportunistic charge."""
        c = make_charger(chargerNumber=0, xyLocation=[5, 5], status='idle')
        r = make_robot(
            robotNumber=0, batteryPercent=65.0, status='idle',
            actionQueue=[], xyLocation=[10, 10],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._charger_pressure = 0.1
        # Threshold is 35, robot at 65 > threshold, but opportunistic kicks in
        wh._charge_threshold = 35.0
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_called_once()

    def test_opportunistic_blocked_by_high_pressure(self):
        """Idle robot at <=70% but high charger pressure → no opportunistic charge."""
        c = make_charger(chargerNumber=0, xyLocation=[5, 5], status='idle')
        r = make_robot(
            robotNumber=0, batteryPercent=65.0, status='idle',
            actionQueue=[], xyLocation=[10, 10],
        )
        wh = WarehouseStub(robots=[r], chargers=[c])
        wh._charger_pressure = 0.5  # above 0.3 threshold
        wh._charge_threshold = 35.0
        wh.find_available_charger = MagicMock(return_value=(True, 0))
        wh._dispatch_charging_if_needed(0)
        wh.robot_insert_task.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# _emergency_battery_drop
# ═══════════════════════════════════════════════════════════════════════

class TestEmergencyBatteryDrop:
    def test_critical_battery_drops_package(self):
        p = make_package(packageNumber=10, xyLocation=[10, 10], status='carried',
                         carrier=0, areaTarget='storage')
        r = make_robot(
            robotNumber=0, batteryPercent=5.0, carrying=10,
            xyLocation=[10, 10],
            actionQueue=[[[20, 20], 'move to target dropoff location']],
        )
        wh = WarehouseStub(width=30, height=30, robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesMovingList = [10]
        wh.packagesMoveList = [10]
        wh._emergency_battery_drop(0)
        assert r.carrying == -1
        assert p.status == 'idle'
        assert p.carrier == -1
        assert p.areaTarget == 'none'
        assert 10 not in wh.packagesMovingList
        assert 10 not in wh.packagesMoveList

    def test_above_critical_no_drop(self):
        r = make_robot(batteryPercent=15.0, carrying=10)
        wh = WarehouseStub(robots=[r])
        wh._critical_batt_pct = 8.0
        wh._emergency_battery_drop(0)
        assert r.carrying == 10

    def test_not_carrying_no_drop(self):
        r = make_robot(batteryPercent=2.0, carrying=-1)
        wh = WarehouseStub(robots=[r])
        wh._emergency_battery_drop(0)
        # Should not raise

    def test_package_location_updated(self):
        p = make_package(packageNumber=5, xyLocation=[10, 10], status='carried',
                         carrier=0)
        r = make_robot(
            robotNumber=0, batteryPercent=3.0, carrying=5,
            xyLocation=[7, 7],
        )
        wh = WarehouseStub(width=20, height=20, robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh._emergency_battery_drop(0)
        # Package should be at or near robot's location
        assert isinstance(p.xyLocation, list)
        assert len(p.xyLocation) == 2

    def test_strips_delivery_tasks_from_queue(self):
        p = make_package(packageNumber=5, status='carried', carrier=0)
        r = make_robot(
            robotNumber=0, batteryPercent=3.0, carrying=5,
            xyLocation=[7, 7],
            actionQueue=[
                [[20, 20], 'move to target dropoff location'],
                [[20, 20], 'drop off target package'],
            ],
        )
        wh = WarehouseStub(width=20, height=20, robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh._emergency_battery_drop(0)
        # robot_pop_task should be called for each delivery task type
        assert wh.robot_pop_task.call_count >= 2
