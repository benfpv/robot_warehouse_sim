"""Tests for Warehouse.reconcile_zone_changes() zone-paint surgery logic."""
import pytest
import numpy as np
from datetime import timedelta
from conftest import make_robot, make_package, WarehouseStub
from data.constants import ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT


# ── helpers ──────────────────────────────────────────────────────────────────

def _wh_with_zone(width=20, height=20):
    """Stub with a simple zone map: import strip on left, storage middle, export right."""
    wh = WarehouseStub(width=width, height=height)
    # Paint a minimal zone map so ZONE_NAMES lookups work
    wh.zoneMap[5][2]  = ZONE_IMPORT
    wh.zoneMap[5][10] = ZONE_STORAGE
    wh.zoneMap[5][17] = ZONE_EXPORT
    return wh


# ── no-op when empty ─────────────────────────────────────────────────────────

class TestReconcileNoOp:
    def test_empty_pending_set_is_noop(self):
        """reconcile_zone_changes with no painted cells must not touch packages."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5], status='idle',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.pending_zone_changes = set()
        wh.reconcile_zone_changes()
        assert pkg.status == 'idle'
        assert pkg.areaTarget == 'storage'

    def test_pending_set_cleared_after_call(self):
        """pending_zone_changes must be empty after the call."""
        wh = _wh_with_zone()
        wh.packages = []
        wh.pending_zone_changes = {(2, 5), (3, 6)}
        wh.reconcile_zone_changes()
        assert wh.pending_zone_changes == set()

    def test_unrelated_cell_leaves_packages_intact(self):
        """A painted cell that no package occupies or targets has no effect."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5], status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.pending_zone_changes = {(15, 15)}  # nowhere near the package
        wh.reconcile_zone_changes()
        assert pkg.status == 'move planned'
        assert 0 in wh.packagesMoveList


# ── Case A: idle package on repainted cell ────────────────────────────────────

class TestReconcileCaseA:
    def test_idle_pkg_on_changed_cell_added_to_reorg(self):
        """Idle package sitting on a repainted cell → added to packagesReorgList."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5], status='idle',
                           areaTarget='none', xyLocationTarget=[2, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert 0 in wh.packagesReorgList

    def test_idle_pkg_status_unchanged(self):
        """Case A must not change status from idle."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5], status='idle',
                           areaTarget='none', xyLocationTarget=[2, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert pkg.status == 'idle'

    def test_idle_pkg_not_added_to_reorg_twice(self):
        """Already in reorgList → not duplicated."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5], status='idle',
                           areaTarget='none', xyLocationTarget=[2, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesReorgList = [0]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert wh.packagesReorgList.count(0) == 1


# ── Case B: move-planned package on repainted cell ────────────────────────────

class TestReconcileCaseB:
    def test_move_planned_pkg_reverts_to_idle(self):
        """Case B: 'move planned' package on a repainted cell → idle."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5],
                           status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert pkg.status == 'idle'
        assert pkg.areaTarget == 'none'

    def test_move_planned_removed_from_move_list(self):
        pkg = make_package(packageNumber=0, xyLocation=[2, 5],
                           status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert 0 not in wh.packagesMoveList

    def test_inbound_robot_tasks_stripped(self):
        """Robot with a pickup task for the affected package gets its task stripped."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5],
                           status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        r = make_robot(robotNumber=0,
                       actionQueue=[[[2, 5], 'move to target pickup location'],
                                    [[10, 5], 'move to target dropoff location']])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.robots = [r]
        wh.robotsTaskAssignmentList = [[0, 2]]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        pickup_actions = {t[1] for t in r.actionQueue}
        assert 'move to target pickup location' not in pickup_actions

    def test_case_b_pkg_added_to_reorg(self):
        pkg = make_package(packageNumber=0, xyLocation=[2, 5],
                           status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.pending_zone_changes = {(2, 5)}
        wh.reconcile_zone_changes()
        assert 0 in wh.packagesReorgList


# ── Case D: target cell repainted, package elsewhere ─────────────────────────

class TestReconcileCaseD:
    def test_target_changed_cancels_plan(self):
        """Case D: target zone repainted → 'move planned' plan cancelled."""
        pkg = make_package(packageNumber=0, xyLocation=[2, 5],
                           status='move planned',
                           areaTarget='storage', xyLocationTarget=[10, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg]
        wh.packagesMoveList = [0]
        wh.pending_zone_changes = {(10, 5)}  # target cell changed, not origin
        wh.reconcile_zone_changes()
        assert pkg.status == 'idle'
        assert pkg.areaTarget == 'none'
        assert 0 not in wh.packagesMoveList


# ── planned count recount ─────────────────────────────────────────────────────

class TestReconcilePlannedRecount:
    def test_planned_counts_updated_after_reconcile(self):
        """After a Case B cancellation, planned counts must be refreshed."""
        pkg0 = make_package(packageNumber=0, xyLocation=[2, 5],
                            status='move planned',
                            areaTarget='storage', xyLocationTarget=[10, 5])
        pkg1 = make_package(packageNumber=1, xyLocation=[3, 5],
                            status='move planned',
                            areaTarget='export', xyLocationTarget=[17, 5])
        wh = _wh_with_zone()
        wh.packages = [pkg0, pkg1]
        wh.packagesMoveList = [0, 1]
        wh.packagesPlannedInStorageCount = 1
        wh.packagesPlannedInExportCount = 1
        wh.pending_zone_changes = {(2, 5)}  # only pkg0's cell
        wh.reconcile_zone_changes()
        # pkg0 plan cancelled → storage planned should now be 0
        assert wh.packagesPlannedInStorageCount == 0
        # pkg1 unaffected → export planned still 1
        assert wh.packagesPlannedInExportCount == 1


# ── robustness: bad cell must not abort processing ────────────────────────────

class TestReconcileRobustness:
    def test_one_bad_package_does_not_abort_others(self):
        """An exception processing one package should not prevent others."""
        class BadPackage:
            """Package whose xyLocation access raises — simulates corrupt state."""
            packageNumber = 99
            areaTarget = 'none'  # required by recount_planned after the loop
            @property
            def xyLocation(self):
                raise RuntimeError("simulated corrupt attribute")

        pkg_good = make_package(packageNumber=0, xyLocation=[2, 5],
                                status='idle', areaTarget='none',
                                xyLocationTarget=[2, 5])
        wh = _wh_with_zone()
        wh.packages = [BadPackage(), pkg_good]
        wh.pending_zone_changes = {(2, 5)}
        # Should not raise; the good package processes normally
        wh.reconcile_zone_changes()
        assert 0 in wh.packagesReorgList
