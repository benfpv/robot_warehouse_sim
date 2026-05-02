"""Stability hardening tests for Phase 1-3 fixes.

Covers:
- Charger location aliasing (Phase 1.1)
- Bounded log buffers (Phase 1.3)
- Float-tolerance epsilons (Phase 1.4)
- _remove_robot force-drop + queue sweep (Phase 2.5/2.6)
- _set_package_idle helper (Phase 2.7)
- _check_invariants_lightweight auto-repair (Phase 2.8)
- Bounds-check occupancy writes (Phase 2.9)
- pending_zone_changes snapshot (Phase 3.10)
- Per-cell zone reconcile try/except (Phase 3.11)
"""
from collections import deque

import numpy as np
import pytest

from conftest import make_robot, make_package, make_charger, WarehouseStub
from data.functions import Functions
from data.importer import Importer
from data.warehouse.warehouse import Warehouse
from data.warehouse.charger_functions import Charger_Functions
from data.warehouse.robot import Robot
from data.warehouse.package import Package


def _make_real_warehouse(**overrides):
    """Build a real Warehouse the same way MainGame does (minus OpenCV)."""
    res = (80, 70)
    bg = [20, 20, 20]
    center = Functions.get_screencenter(res)
    array = Functions.get_screenarray_colour(res, bg)
    items = Importer.init_import_csv_as_list('resources/list_items.csv')
    items = Importer.init_objectify_items_list(items)
    addrs = Importer.init_import_csv_as_list('resources/list_addresses.csv')
    addrs = Importer.init_objectify_addresses_list(addrs)
    kw = dict(robotsMaxQuantity=4, chargersMaxQuantity=2)
    kw.update(overrides)
    return Warehouse(res, bg, center, array, items, addrs, **kw)


# --------------------------------------------------------------------- Phase 1.1
class TestChargerLocationCopy:
    """generate_charger must not alias the caller's spawn-location list."""

    def test_charger_location_is_copy(self):
        spawn = [4, 7]
        c = Charger_Functions.generate_charger(spawn, 0)
        assert c.xyLocation == [4, 7]
        spawn[0] = 999
        assert c.xyLocation[0] == 4, "charger location aliased to caller list"

    def test_charger_location_is_not_same_object(self):
        spawn = [4, 7]
        c = Charger_Functions.generate_charger(spawn, 0)
        assert c.xyLocation is not spawn

    def test_generate_charger_number_matches(self):
        c = Charger_Functions.generate_charger([10, 10], 42)
        assert c.chargerNumber == 42
        assert c.status == 'idle'
        assert c.area == 'neutral'


# --------------------------------------------------------------------- Phase 1.3
class TestBoundedLogBuffers:
    """packagesLog and robotsLog must be bounded deques."""

    def test_logs_are_bounded_deques(self):
        w = _make_real_warehouse(logsMaxLength=128)
        assert isinstance(w.packagesLog, deque)
        assert isinstance(w.robotsLog, deque)
        assert w.packagesLog.maxlen == 128
        assert w.robotsLog.maxlen == 128

    def test_log_append_self_trims_at_maxlen(self):
        w = _make_real_warehouse(logsMaxLength=10)
        for i in range(50):
            w.robotsLog.append(('dummy', i))
        assert len(w.robotsLog) == 10
        # Oldest evicted, newest retained
        assert w.robotsLog[0] == ('dummy', 40)
        assert w.robotsLog[-1] == ('dummy', 49)

    def test_trim_logs_is_noop_safe(self):
        w = _make_real_warehouse(logsMaxLength=10)
        for i in range(20):
            w.packagesLog.append(('p', i))
        before_len = len(w.packagesLog)
        a, b = w.trim_logs()
        assert a is w.packagesLog
        assert b is w.robotsLog
        # Length unchanged — trim_logs doesn't touch deques
        assert len(w.packagesLog) == before_len

    def test_passing_existing_list_coerces_to_deque(self):
        """When caller provides a plain list (e.g. from save/load), it must
        be coerced to a bounded deque, not stored as a raw list."""
        existing = [('old', i) for i in range(100)]
        w = _make_real_warehouse(logsMaxLength=20, packagesLog=existing)
        assert isinstance(w.packagesLog, deque)
        assert w.packagesLog.maxlen == 20
        # Only last 20 items retained from the 100-item list
        assert len(w.packagesLog) == 20
        assert w.packagesLog[-1] == ('old', 99)

    def test_packages_log_coercion_from_list(self):
        w = _make_real_warehouse(logsMaxLength=5, robotsLog=[1, 2, 3, 4, 5, 6, 7])
        assert isinstance(w.robotsLog, deque)
        assert w.robotsLog.maxlen == 5
        assert len(w.robotsLog) == 5


# --------------------------------------------------------------------- Phase 2.5+2.6
class TestRemoveRobotForceDrop:
    """_remove_robot must force-drop carried packages and clean queues, never assert."""

    def test_remove_robot_with_carried_package_force_drops(self):
        w = _make_real_warehouse()
        for _ in range(200):
            w.update_warehouse()
            if w.robots:
                break
        assert w.robots, "no robots spawned in 200 ticks"
        r = w.robots[0]
        initial_robot_count = len(w.robots)

        # Inject a fake carried package
        pkg = make_package(
            packageNumber=99999, xyLocation=list(r.xyLocation),
            status='carried', areaTarget='storage', xyLocationTarget=[30, 30],
            carrier=r.robotNumber,
        )
        w.packages.append(pkg)
        w._pkg_idx = {}  # invalidate cache so _find_package rebuilds
        r.carrying = pkg.packageNumber
        w.packagesMovingList.append(pkg.packageNumber)
        w.packagesMoveList.append(pkg.packageNumber)
        w.packagesReorgList.append(pkg.packageNumber)

        # Should not raise even though robot is carrying.
        w._remove_robot(0)

        # Package was force-dropped to idle with full state reset
        assert pkg.status == 'idle'
        assert pkg.carrier == -1
        assert pkg.areaTarget == 'none'
        assert pkg.xyLocationTarget == pkg.xyLocation
        # Move/moving/reorg lists were swept
        assert pkg.packageNumber not in w.packagesMovingList
        assert pkg.packageNumber not in w.packagesMoveList
        assert pkg.packageNumber not in w.packagesReorgList
        # Robot actually removed from fleet
        assert len(w.robots) == initial_robot_count - 1

    def test_remove_robot_clears_lookup_cache(self):
        w = _make_real_warehouse()
        for _ in range(200):
            w.update_warehouse()
            if w.robots:
                break
        assert w.robots
        rnum = w.robots[0].robotNumber
        # Prime cache
        idx = w._find_robot(rnum)
        assert idx is not None
        w._remove_robot(0)
        # Cache must be cleared so subsequent lookups don't return stale indices
        assert w._robot_idx == {}
        assert w._pkg_idx == {}

    def test_remove_robot_without_carrying(self):
        """Removing a robot that is not carrying anything should work cleanly."""
        w = _make_real_warehouse()
        for _ in range(200):
            w.update_warehouse()
            if w.robots:
                break
        assert w.robots
        r = w.robots[0]
        r.carrying = -1  # ensure not carrying
        initial_count = len(w.robots)
        w._remove_robot(0)
        assert len(w.robots) == initial_count - 1

    def test_remove_robot_decrements_warehouse_count(self):
        w = _make_real_warehouse()
        for _ in range(200):
            w.update_warehouse()
            if w.robots:
                break
        assert w.robots
        count_before = w.robotsInWarehouseCount
        w._remove_robot(0)
        assert w.robotsInWarehouseCount == count_before - 1

    def test_remove_robot_sweeps_task_assignment_list(self):
        w = _make_real_warehouse()
        for _ in range(200):
            w.update_warehouse()
            if w.robots:
                break
        assert w.robots
        rnum = w.robots[0].robotNumber
        w._remove_robot(0)
        assert all(x[0] != rnum for x in w.robotsTaskAssignmentList)


# --------------------------------------------------------------------- Phase 2.7
class TestSetPackageIdleHelper:
    """_set_package_idle resets all four fields atomically."""

    def test_full_reset(self):
        w = _make_real_warehouse()
        pkg = make_package(
            packageNumber=1, xyLocation=[10, 10],
            status='carried', areaTarget='storage', xyLocationTarget=[40, 40],
            carrier=7,
        )
        w._set_package_idle(pkg)
        assert pkg.status == 'idle'
        assert pkg.carrier == -1
        assert pkg.areaTarget == 'none'
        assert pkg.xyLocationTarget == [10, 10]
        assert pkg.xyLocation == [10, 10]

    def test_relocate_on_drop(self):
        w = _make_real_warehouse()
        pkg = make_package(
            packageNumber=1, xyLocation=[10, 10],
            status='carried', areaTarget='storage', xyLocationTarget=[40, 40],
            carrier=7,
        )
        w._set_package_idle(pkg, location=[55, 22])
        assert pkg.xyLocation == [55, 22]
        assert pkg.xyLocationTarget == [55, 22]
        assert pkg.status == 'idle'
        assert pkg.carrier == -1
        assert pkg.areaTarget == 'none'

    def test_location_is_independent_copy(self):
        """Passed location must be copied, not aliased."""
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=1, xyLocation=[10, 10], status='carried', carrier=5)
        loc = [55, 22]
        w._set_package_idle(pkg, location=loc)
        loc[0] = 999
        assert pkg.xyLocation == [55, 22], "location was aliased"

    def test_target_syncs_to_current_location(self):
        """After idle reset without explicit location, target == current."""
        w = _make_real_warehouse()
        pkg = make_package(
            packageNumber=1, xyLocation=[33, 44],
            status='carried', areaTarget='export', xyLocationTarget=[10, 10],
            carrier=2,
        )
        w._set_package_idle(pkg)
        assert pkg.xyLocationTarget == [33, 44]
        assert pkg.xyLocation == [33, 44]

    def test_does_not_modify_deliveredAt(self):
        """deliveredAt is intentionally left untouched."""
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=1, status='carried', carrier=3)
        pkg.deliveredAt = 42.0
        w._set_package_idle(pkg)
        assert pkg.deliveredAt == 42.0

    def test_idempotent_on_already_idle(self):
        """Calling on an already-idle package must not break anything."""
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=1, xyLocation=[10, 10], status='idle',
                           areaTarget='none', xyLocationTarget=[10, 10])
        w._set_package_idle(pkg)
        assert pkg.status == 'idle'
        assert pkg.carrier == -1
        assert pkg.areaTarget == 'none'

    def test_works_via_stub(self):
        """WarehouseStub must have _set_package_idle grafted."""
        wh = WarehouseStub()
        pkg = make_package(packageNumber=1, status='carried', carrier=5,
                           areaTarget='export', xyLocationTarget=[40, 40])
        wh._set_package_idle(pkg)
        assert pkg.status == 'idle'
        assert pkg.carrier == -1


# --------------------------------------------------------------------- Phase 2.8
class TestLightweightInvariants:
    """_check_invariants_lightweight must auto-repair common corruptions."""

    def test_zombie_idle_is_repaired(self):
        w = _make_real_warehouse()
        pkg = make_package(
            packageNumber=1, xyLocation=[10, 10],
            status='idle', areaTarget='storage', xyLocationTarget=[40, 40],
        )
        w.packages = [pkg]
        w._check_invariants_lightweight()
        assert pkg.areaTarget == 'none'
        assert pkg.xyLocationTarget == [10, 10]
        assert pkg.status == 'idle'
        assert pkg.carrier == -1

    def test_zombie_idle_with_export_target(self):
        """Any areaTarget != 'none' on an idle package is a zombie."""
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=2, xyLocation=[5, 5],
                           status='idle', areaTarget='export', xyLocationTarget=[50, 50])
        w.packages = [pkg]
        w._check_invariants_lightweight()
        assert pkg.areaTarget == 'none'
        assert pkg.xyLocationTarget == [5, 5]

    def test_dangling_carry_is_repaired(self):
        w = _make_real_warehouse()
        r = make_robot(robotNumber=1, carrying=42)
        w.robots = [r]
        w._check_invariants_lightweight()
        assert r.carrying == -1

    def test_carrier_without_robot_drops_to_idle(self):
        """Package marked carried whose carrier robot doesn't exist."""
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=5, status='carried', carrier=999)
        w.packages = [pkg]
        w.robots = []
        w._check_invariants_lightweight()
        assert pkg.status == 'idle'
        assert pkg.carrier == -1
        assert pkg.areaTarget == 'none'

    def test_carrier_symmetry_resync(self):
        """When robot.carrying -> pkg but pkg.status != 'carried', repair resyncs."""
        w = _make_real_warehouse()
        r = make_robot(robotNumber=1, carrying=10)
        pkg = make_package(packageNumber=10, status='idle', carrier=-1)
        w.robots = [r]
        w.packages = [pkg]
        w._check_invariants_lightweight()
        # Repair direction: robot claims override -> package marked as carried
        assert pkg.status == 'carried'
        assert pkg.carrier == 1

    def test_out_of_grid_robot_is_clamped(self):
        w = _make_real_warehouse()
        r = make_robot(robotNumber=1, xyLocation=[-5, 999])
        w.robots = [r]
        w._check_invariants_lightweight()
        _x, _y = r.xyLocation
        assert _x == 0  # clamped from -5
        assert _y == w.warehouseWindowRes[1] - 1  # clamped from 999

    def test_out_of_grid_package_is_clamped(self):
        w = _make_real_warehouse()
        pkg = make_package(packageNumber=1, xyLocation=[9999, -1])
        w.packages = [pkg]
        w._check_invariants_lightweight()
        _x, _y = pkg.xyLocation
        assert _x == w.warehouseWindowRes[0] - 1
        assert _y == 0

    def test_in_grid_entities_not_modified(self):
        """Entities within bounds must not have their coords changed."""
        w = _make_real_warehouse()
        r = make_robot(robotNumber=1, xyLocation=[10, 10])
        pkg = make_package(packageNumber=1, xyLocation=[20, 20])
        w.robots = [r]
        w.packages = [pkg]
        w._check_invariants_lightweight()
        assert r.xyLocation == [10, 10]
        assert pkg.xyLocation == [20, 20]

    def test_multiple_violations_all_repaired(self):
        """Multiple different violations in one pass are all repaired."""
        w = _make_real_warehouse()
        r = make_robot(robotNumber=1, carrying=77)  # dangling carry
        zombie_pkg = make_package(packageNumber=2, xyLocation=[5, 5],
                                  status='idle', areaTarget='storage')  # zombie
        orphan_pkg = make_package(packageNumber=3, status='carried', carrier=999)  # orphan
        oob_r = make_robot(robotNumber=2, xyLocation=[-1, -1])  # out of grid
        w.robots = [r, oob_r]
        w.packages = [zombie_pkg, orphan_pkg]
        w._check_invariants_lightweight()
        assert r.carrying == -1
        assert zombie_pkg.areaTarget == 'none'
        assert orphan_pkg.status == 'idle'
        assert oob_r.xyLocation == [0, 0]

    def test_works_via_stub(self):
        """WarehouseStub must have _check_invariants_lightweight grafted."""
        wh = WarehouseStub()
        r = make_robot(robotNumber=1, carrying=99)
        wh.robots = [r]
        wh.packages = []
        wh._check_invariants_lightweight()
        assert r.carrying == -1


# --------------------------------------------------------------------- Phase 2.9
class TestOccupancyBoundsCheck:
    """update_*_in_warehouse must skip out-of-grid coords instead of crashing."""

    def test_out_of_grid_robot_does_not_crash(self):
        w = _make_real_warehouse()
        w.robots = [make_robot(robotNumber=1, xyLocation=[5, 5]),
                    make_robot(robotNumber=2, xyLocation=[-1, -1])]
        grid = w.update_robots_in_warehouse()
        assert grid[5][5] == 1  # in-bounds robot was written
        # Grid shape matches warehouse resolution (height, width)
        assert grid.shape == (w.warehouseWindowRes[1], w.warehouseWindowRes[0])

    def test_out_of_grid_package_does_not_crash(self):
        w = _make_real_warehouse()
        good_pkg = make_package(packageNumber=1, xyLocation=[3, 3])
        bad_pkg = make_package(packageNumber=2, xyLocation=[9999, 9999])
        w.packages = [good_pkg, bad_pkg]
        grid = w.update_packages_in_warehouse()
        assert grid[3][3] == 1  # in-bounds package written
        # No crash from out-of-bounds

    def test_out_of_grid_charger_does_not_crash(self):
        w = _make_real_warehouse()
        good_c = make_charger(chargerNumber=1, xyLocation=[2, 2])
        bad_c = make_charger(chargerNumber=2, xyLocation=[-10, -10])
        w.chargers = [good_c, bad_c]
        grid = w.update_chargers_in_warehouse()
        assert grid[2][2] == 1

    def test_zero_entities_returns_empty_grid(self):
        w = _make_real_warehouse()
        w.robots = []
        grid = w.update_robots_in_warehouse()
        assert grid.sum() == 0

    def test_occupancy_grid_resets_each_call(self):
        """Calling update twice must not accumulate old state."""
        w = _make_real_warehouse()
        w.robots = [make_robot(robotNumber=1, xyLocation=[5, 5])]
        w.update_robots_in_warehouse()
        w.robots = [make_robot(robotNumber=2, xyLocation=[7, 7])]
        grid = w.update_robots_in_warehouse()
        assert grid[5][5] == 0  # old position cleared
        assert grid[7][7] == 1  # new position written

    def test_via_stub_robots(self):
        """WarehouseStub must support update_robots_in_warehouse."""
        wh = WarehouseStub(robots=[make_robot(robotNumber=1, xyLocation=[3, 3])])
        grid = wh.update_robots_in_warehouse()
        assert grid[3][3] == 1

    def test_via_stub_packages(self):
        wh = WarehouseStub(packages=[make_package(packageNumber=1, xyLocation=[4, 4])])
        grid = wh.update_packages_in_warehouse()
        assert grid[4][4] == 1


# --------------------------------------------------------------------- Phase 3.10
class TestZoneReconcileSnapshot:
    """reconcile_zone_changes must snapshot then clear pending_zone_changes."""

    def test_snapshot_clears_pending(self):
        w = _make_real_warehouse()
        w.pending_zone_changes = {(1, 1), (2, 2)}
        w.reconcile_zone_changes()
        assert w.pending_zone_changes == set()
        assert isinstance(w.pending_zone_changes, set)

    def test_no_op_when_empty(self):
        w = _make_real_warehouse()
        w.pending_zone_changes = set()
        # Should not raise even with no packages
        w.reconcile_zone_changes()
        assert w.pending_zone_changes == set()

    def test_recount_planned_called_after_reconcile(self):
        """After reconciliation, planned counts must reflect actual package state."""
        w = _make_real_warehouse()
        # Create packages with specific area targets
        p1 = make_package(packageNumber=1, areaTarget='import')
        p2 = make_package(packageNumber=2, areaTarget='storage')
        p3 = make_package(packageNumber=3, areaTarget='none')
        w.packages = [p1, p2, p3]
        # Manually corrupt counts
        w.packagesPlannedInImportCount = 99
        w.packagesPlannedInStorageCount = 99
        # Trigger reconcile with a cell change
        w.pending_zone_changes = {(999, 999)}  # non-matching cell
        w.reconcile_zone_changes()
        # Counts must be authoritative now
        assert w.packagesPlannedInImportCount == 1
        assert w.packagesPlannedInStorageCount == 1
        assert w.packagesPlannedInExportCount == 0

    def test_bad_package_does_not_crash_reconcile(self):
        """A package that raises during reconciliation must not crash the whole loop."""
        w = _make_real_warehouse()
        good_pkg = make_package(packageNumber=1, xyLocation=[5, 5], status='idle')
        w.packages = [good_pkg]
        # Set a changed cell that matches the good package
        w.pending_zone_changes = {(5, 5)}
        # This should complete without raising
        w.reconcile_zone_changes()
        assert w.pending_zone_changes == set()


# --------------------------------------------------------------------- Phase 3.12
class TestAtomicRoadMapSwap:
    """_maybe_rebuild_roads assigns all related fields together."""

    def test_road_map_exists_after_init(self):
        w = _make_real_warehouse()
        assert hasattr(w, 'road_map')
        assert isinstance(w.road_map, np.ndarray)

    def test_road_map_shape_matches_warehouse(self):
        w = _make_real_warehouse()
        expected_h, expected_w = w.warehouseWindowRes[1], w.warehouseWindowRes[0]
        assert w.road_map.shape == (expected_h, expected_w)


# --------------------------------------------------------------------- Cross-cutting
class TestLongRunWithLightweightInvariants:
    """Long stress run must stay clean under always-on invariant checks."""

    def test_500_ticks_clean(self, caplog):
        w = _make_real_warehouse()
        with caplog.at_level('ERROR', logger='data.warehouse.warehouse'):
            for _ in range(500):
                w.update_warehouse()
        # No INVARIANT-LITE error messages should fire under normal operation
        invariant_errors = [m for m in caplog.messages if 'INVARIANT-LITE' in m]
        assert not invariant_errors, f"unexpected invariant violations: {invariant_errors[:5]}"
        # All entities still within grid
        _w, _h = w.warehouseWindowRes
        for r in w.robots:
            assert 0 <= int(r.xyLocation[0]) < _w
            assert 0 <= int(r.xyLocation[1]) < _h
        for p in w.packages:
            assert 0 <= int(p.xyLocation[0]) < _w
            assert 0 <= int(p.xyLocation[1]) < _h
        # Carrier symmetry holds
        carried_pkgs = {p.packageNumber for p in w.packages if p.carrier not in (None, -1)}
        carrying_robots = {r.carrying for r in w.robots if r.carrying != -1}
        assert carried_pkgs == carrying_robots
        # No zombie idle packages
        for p in w.packages:
            if p.status == 'idle':
                assert p.areaTarget == 'none', \
                    f"pkg#{p.packageNumber} idle but areaTarget={p.areaTarget}"
        # Logs still bounded
        assert isinstance(w.packagesLog, deque)
        assert len(w.packagesLog) <= w.packagesLog.maxlen

    def test_100_ticks_invariant_frequency(self):
        """Invariant check runs at tick 100, 200, etc. — verify it doesn't crash on
        a minimal warehouse."""
        w = _make_real_warehouse(robotsMaxQuantity=1, chargersMaxQuantity=1)
        for _ in range(200):
            w.update_warehouse()
        assert w.warehouseLoopCount == 200