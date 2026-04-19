"""Tests for Robot, Package, and Charger data classes."""
import pytest
from data.warehouse.robot import Robot, Robot_Log
from data.warehouse.package import Package
from data.warehouse.charger import Charger, Charger_Log
from conftest import make_robot, make_package, make_charger


# ═══════════════════════════════════════════════════════════════════════
# Robot
# ═══════════════════════════════════════════════════════════════════════

class TestRobotConstruction:
    def test_basic_construction(self):
        r = make_robot(robotNumber=7)
        assert r.robotNumber == 7

    def test_all_constructor_params_stored(self):
        r = make_robot(
            robotNumber=3, batteryPercent=80.0, batteryChargingRate=1.2,
            batteryDepletingRate=0.12, status='charging', carrying=5,
            xyLocation=[15, 25], areaTarget='storage',
        )
        assert r.robotNumber == 3
        assert r.batteryPercent == 80.0
        assert r.batteryChargingRate == 1.2
        assert r.batteryDepletingRate == 0.12
        assert r.status == 'charging'
        assert r.carrying == 5
        assert r.xyLocation == [15, 25]
        assert r.areaTarget == 'storage'

    def test_action_queue_is_list(self):
        r = make_robot()
        assert isinstance(r.actionQueue, list)
        assert len(r.actionQueue) == 0

    def test_xy_location_is_independent_copy(self):
        """Mutating the passed-in list must not affect the robot."""
        loc = [10, 10]
        r = make_robot(xyLocation=loc)
        loc[0] = 999
        assert r.xyLocation == [10, 10], "xyLocation must not alias caller list"

    def test_two_robots_no_shared_mutable_state(self):
        """Two robots created from same defaults must be fully independent."""
        r1 = make_robot()
        r2 = make_robot()
        r1.actionQueue.append('task')
        r1.xyLocation[0] = 99
        r1.path.append([1, 2])
        assert r2.actionQueue == []
        assert r2.xyLocation == [10, 10]
        assert r2.path == []


class TestRobotDefaults:
    """Verify secondary attributes initialised in __init__ body."""

    def test_motion_model_defaults(self):
        r = make_robot()
        assert r.desiredVelocity == 0.0
        assert r.maxVelocity == 0.50
        assert r.accelerationRate == 0.05
        assert r.decelerationRate == 0.05
        assert r.minMovingVelocity == 0.10
        assert r.movementProgress == 0.0
        assert r.blockedTicks == 0

    def test_path_planning_defaults(self):
        r = make_robot()
        assert r.path == []
        assert r.pathPlan == []
        assert r.pathTarget is None
        assert r.pathAge == 0
        assert r.recentLocations == []
        assert r.replanCooldownTicks == 0

    def test_lifecycle_defaults(self):
        r = make_robot()
        assert r.createdAt == 0.0
        assert r.stallTicks == 0
        assert r.hasCharged is False
        assert r.hasCarried is False
        assert r.decommissioning is False

    def test_birth_location_is_copy(self):
        loc = [10, 10]
        r = make_robot(xyLocation=loc)
        assert r.birthLocation == loc
        loc[0] = 99
        assert r.birthLocation[0] != 99, "birthLocation must be independent copy"

    def test_target_direction_matches_direction(self):
        r = make_robot(direction=45)
        assert r.targetDirection == 45

    def test_degrees_rounded(self):
        r = make_robot(direction=45.7)
        assert r.degrees == 46

    def test_battery_drain_multiplier_default(self):
        r = make_robot()
        assert r.batteryDrainMultiplier == 1.0


class TestRobotLog:
    def test_log_construction(self):
        from datetime import datetime
        now = datetime.now()
        log = Robot_Log("pickup", now)
        assert log.action == "pickup"
        assert log.datetimeNow == now


# ═══════════════════════════════════════════════════════════════════════
# Package
# ═══════════════════════════════════════════════════════════════════════

class TestPackageConstruction:
    def test_basic_construction(self):
        p = make_package(packageNumber=42)
        assert p.packageNumber == 42

    def test_all_constructor_params_stored(self):
        p = make_package(
            packageNumber=10, area='storage', status='carried',
            carrier=3, xyLocation=[5, 5], xyLocationTarget=[15, 15],
        )
        assert p.packageNumber == 10
        assert p.area == 'storage'
        assert p.status == 'carried'
        assert p.carrier == 3
        assert p.xyLocation == [5, 5]
        assert p.xyLocationTarget == [15, 15]

    def test_defaults(self):
        p = make_package()
        assert p.createdAt == 0.0
        assert p.deliveredAt is None
        assert p.plannedAt is None
        assert p.carrier == -1

    def test_xy_location_is_independent_copy(self):
        loc = [20, 20]
        p = make_package(xyLocation=loc)
        loc[0] = 999
        assert p.xyLocation == [20, 20], "package xyLocation must not alias caller list"

    def test_two_packages_no_shared_mutable_state(self):
        p1 = make_package()
        p2 = make_package()
        p1.xyLocation[0] = 99
        p1.packageLog.append('entry')
        assert p2.xyLocation[0] == 20
        assert p2.packageLog == []


# ═══════════════════════════════════════════════════════════════════════
# Charger
# ═══════════════════════════════════════════════════════════════════════

class TestChargerConstruction:
    def test_basic_construction(self):
        c = make_charger(chargerNumber=2)
        assert c.chargerNumber == 2

    def test_all_params_stored(self):
        c = make_charger(
            chargerNumber=5, colour=[0, 0, 255], area='import',
            xyLocation=[3, 7], status='charging',
        )
        assert c.chargerNumber == 5
        assert c.colour == [0, 0, 255]
        assert c.area == 'import'
        assert c.xyLocation == [3, 7]
        assert c.status == 'charging'

    def test_default_status_is_idle(self):
        c = make_charger()
        assert c.status == 'idle'

    def test_xy_location_is_independent_copy(self):
        loc = [5, 5]
        c = make_charger(xyLocation=loc)
        loc[0] = 999
        assert c.xyLocation == [5, 5], "charger xyLocation must not alias caller list"


class TestChargerLog:
    def test_log_construction(self):
        from datetime import datetime
        now = datetime.now()
        log = Charger_Log("docked", now)
        assert log.action == "docked"
        assert log.datetimeNow == now
