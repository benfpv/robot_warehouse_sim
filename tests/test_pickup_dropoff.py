"""Tests for _robot_do_pickup, _robot_do_dropoff, and package_dropoff."""
import pytest
from unittest.mock import MagicMock
from conftest import make_robot, make_package, make_charger, WarehouseStub
from data.constants import STALL_TICKS


# ═══════════════════════════════════════════════════════════════════════
# _robot_do_pickup
# ═══════════════════════════════════════════════════════════════════════

class TestRobotDoPickup:
    def test_successful_pickup(self):
        p = make_package(
            packageNumber=10, xyLocation=[15, 15], status='idle',
            xyLocationTarget=[25, 25], carrier=-1,
        )
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='pick up target package',
            batteryPercent=80.0,
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._robot_do_pickup(0)
        assert r.carrying == 10
        assert r.stallTicks == STALL_TICKS
        assert r.hasCarried is True
        assert p.status == 'carried'
        assert p.carrier == 0
        # Package location must track robot location after pickup
        assert p.xyLocation == r.xyLocation
        wh.robot_replace_task.assert_called_once()

    def test_no_package_at_location(self):
        r = make_robot(
            xyLocation=[15, 15], status='pick up target package',
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r])
        wh._robot_do_pickup(0)
        wh.robot_replace_task.assert_not_called()

    def test_battery_too_low_blocks_pickup(self):
        p = make_package(
            packageNumber=10, xyLocation=[15, 15], status='idle',
            xyLocationTarget=[25, 25],
        )
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='pick up target package',
            batteryPercent=10.0,
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._work_min_batt_pct = 20.0
        wh._robot_do_pickup(0)
        assert r.carrying == -1  # not picked up
        assert p.status == 'idle'
        wh.robot_pop_task.assert_called_once()  # task removed

    def test_package_already_carried_skips(self):
        p = make_package(
            packageNumber=10, xyLocation=[15, 15], status='carried',
            carrier=5,
        )
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='pick up target package',
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._robot_do_pickup(0)
        wh.robot_replace_task.assert_not_called()

    def test_robot_status_carried_skips(self):
        p = make_package(packageNumber=10, xyLocation=[15, 15], status='idle')
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='carried',
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._robot_do_pickup(0)
        wh.robot_replace_task.assert_not_called()

    def test_pickup_at_exact_battery_threshold(self):
        """At exactly _work_min_batt_pct, pickup is allowed (< not <=)."""
        p = make_package(
            packageNumber=10, xyLocation=[15, 15], status='idle',
            xyLocationTarget=[25, 25],
        )
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='pick up target package',
            batteryPercent=20.0,
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._work_min_batt_pct = 20.0
        wh._robot_do_pickup(0)
        # 20.0 < 20.0 is False, so pickup proceeds
        assert r.carrying == 10
        assert p.status == 'carried'

    def test_pickup_replaces_with_dropoff_task(self):
        p = make_package(
            packageNumber=10, xyLocation=[15, 15], status='idle',
            xyLocationTarget=[25, 25],
        )
        r = make_robot(
            robotNumber=0, xyLocation=[15, 15], status='pick up target package',
            batteryPercent=80.0,
            actionQueue=[[[15, 15], 'pick up target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._robot_do_pickup(0)
        # Verify replace_task called with dropoff location
        call = wh.robot_replace_task.call_args
        assert call[0][2] == [25, 25]  # package target location
        assert 'dropoff' in call[0][3]


# ═══════════════════════════════════════════════════════════════════════
# _robot_do_dropoff
# ═══════════════════════════════════════════════════════════════════════

class TestRobotDoDropoff:
    def test_successful_dropoff(self):
        p = make_package(
            packageNumber=10, xyLocation=[25, 25], status='carried',
            carrier=0, areaTarget='storage',
        )
        r = make_robot(
            robotNumber=0, xyLocation=[25, 25], carrying=10,
            status='drop off target package',
            actionQueue=[[[25, 25], 'drop off target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesMovingList = [10]
        wh.packagesMoveList = [10]
        wh.warehouseLoopCount = 400  # sim_time = 400/40 = 10.0
        wh._robot_do_dropoff(0)
        assert r.carrying == -1
        assert r.stallTicks == STALL_TICKS
        assert p.status == 'idle'
        assert p.areaTarget == 'none'
        assert p.carrier == -1
        # deliveredAt must be stamped with sim time
        assert p.deliveredAt == pytest.approx(10.0)
        # Package removed from move lists
        assert 10 not in wh.packagesMoveList
        assert 10 not in wh.packagesMovingList

    def test_package_not_found_skips(self):
        r = make_robot(
            robotNumber=0, carrying=999,
            status='drop off target package',
            actionQueue=[[[25, 25], 'drop off target package']],
        )
        wh = WarehouseStub(robots=[r])
        wh._rebuild_entity_indices()
        wh._robot_do_dropoff(0)
        wh.robot_pop_task.assert_not_called()

    def test_package_not_carried_skips(self):
        p = make_package(packageNumber=10, status='idle', carrier=-1)
        r = make_robot(
            robotNumber=0, carrying=10,
            status='drop off target package',
            actionQueue=[[[25, 25], 'drop off target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh._robot_do_dropoff(0)
        wh.robot_pop_task.assert_not_called()

    def test_robot_status_carried_skips(self):
        p = make_package(packageNumber=10, status='carried', carrier=0)
        r = make_robot(
            robotNumber=0, carrying=10, status='carried',
            actionQueue=[[[25, 25], 'drop off target package']],
        )
        wh = WarehouseStub(robots=[r], packages=[p])
        wh._rebuild_entity_indices()
        wh._robot_do_dropoff(0)
        wh.robot_pop_task.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# package_dropoff
# ═══════════════════════════════════════════════════════════════════════

class TestPackageDropoff:
    def test_removes_from_move_list(self):
        p = make_package(packageNumber=10, areaTarget='storage', status='carried')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesMoveList = [10, 20, 30]
        wh.packagesMovingList = [10, 20]
        wh.package_dropoff(10)
        assert 10 not in wh.packagesMoveList
        assert 10 not in wh.packagesMovingList

    def test_decrements_planned_import_count(self):
        p = make_package(packageNumber=10, areaTarget='import')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesPlannedInImportCount = 5
        wh.package_dropoff(10)
        assert wh.packagesPlannedInImportCount == 4

    def test_decrements_planned_storage_count(self):
        p = make_package(packageNumber=10, areaTarget='storage')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesPlannedInStorageCount = 3
        wh.package_dropoff(10)
        assert wh.packagesPlannedInStorageCount == 2

    def test_decrements_planned_export_count(self):
        p = make_package(packageNumber=10, areaTarget='export')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesPlannedInExportCount = 7
        wh.package_dropoff(10)
        assert wh.packagesPlannedInExportCount == 6

    def test_resets_package_state(self):
        p = make_package(
            packageNumber=10, areaTarget='export',
            status='carried', carrier=3,
        )
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.warehouseLoopCount = 200  # sim_time = 200/40 = 5.0
        wh.package_dropoff(10)
        assert p.status == 'idle'
        assert p.areaTarget == 'none'
        assert p.carrier == -1
        assert p.deliveredAt == pytest.approx(5.0)

    def test_not_in_lists_still_succeeds(self):
        """Tolerant: package not in move/moving lists doesn't crash."""
        p = make_package(packageNumber=10, areaTarget='storage')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesMoveList = []
        wh.packagesMovingList = []
        wh.package_dropoff(10)
        assert p.status == 'idle'

    def test_returns_updated_values(self):
        p = make_package(packageNumber=10, areaTarget='storage')
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        wh.packagesMoveList = [10]
        wh.packagesMovingList = [10]
        wh.packagesPlannedInStorageCount = 1
        result = wh.package_dropoff(10)
        move_list, moving_list, packages, imp, stor, exp = result
        assert 10 not in move_list
        assert 10 not in moving_list
        assert stor == 0
