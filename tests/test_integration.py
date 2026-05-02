"""Integration smoke tests — construct a real Warehouse and tick it."""
import numpy as np
import pytest
from data.functions import Functions
from data.importer import Importer
from data.warehouse.warehouse import Warehouse


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


class TestMutableDefaults:
    """Two independent Warehouse instances must NOT share mutable state."""

    def test_packages_not_shared(self):
        w1 = _make_real_warehouse()
        w2 = _make_real_warehouse()
        assert w1.packages is not w2.packages

    def test_robots_not_shared(self):
        w1 = _make_real_warehouse()
        w2 = _make_real_warehouse()
        assert w1.robots is not w2.robots

    def test_chargers_not_shared(self):
        w1 = _make_real_warehouse()
        w2 = _make_real_warehouse()
        assert w1.chargers is not w2.chargers

    def test_actions_lists_not_shared(self):
        w1 = _make_real_warehouse()
        w2 = _make_real_warehouse()
        assert w1.packagesActionsList is not w2.packagesActionsList
        assert w1.robotsActionsList is not w2.robotsActionsList
        assert w1.chargersActionsList is not w2.chargersActionsList


class TestRealWarehouseTick:
    """Construct a real Warehouse and run update_warehouse() ticks."""

    def test_single_tick_completes(self):
        w = _make_real_warehouse()
        w.update_warehouse()
        assert w.warehouseLoopCount == 1

    def test_ten_ticks_no_crash(self):
        w = _make_real_warehouse()
        for _ in range(10):
            w.update_warehouse()
        assert w.warehouseLoopCount == 10

    def test_entities_spawn_after_ticks(self):
        w = _make_real_warehouse()
        for _ in range(80):
            w.update_warehouse()
        # After 80 ticks robots and chargers should start appearing
        assert len(w.robots) > 0 or len(w.chargers) > 0


class TestInvariantChecker:
    """Verify the debug invariant checker runs clean during normal operation."""

    def test_invariant_check_clean_run(self):
        w = _make_real_warehouse(_debug_invariants=True)
        for _ in range(100):
            w.update_warehouse()
        assert w.warehouseLoopCount == 100


class TestDecommission:
    """Decommissioning must not orphan carried packages."""

    def test_decommission_no_orphan_packages(self):
        w = _make_real_warehouse(robotsMaxQuantity=6)
        for _ in range(200):
            w.update_warehouse()
        initial_count = len(w.robots)
        # Shrink fleet by 1 to trigger decommissioning
        if initial_count > 1:
            target = initial_count - 1
            w.set_robot_count(target)
            for _ in range(400):
                w.update_warehouse()
            # At least one robot should be decommissioning or removed
            assert len(w.robots) <= initial_count
        # Every carried package must have a matching carrying robot
        for p in w.packages:
            if p.status == 'carried':
                assert any(r.carrying == p.packageNumber for r in w.robots), \
                    f"pkg#{p.packageNumber} status=carried but no robot carrying it"
        # No idle packages with a carrier set
        for p in w.packages:
            if p.status == 'idle':
                assert p.carrier in (None, -1), \
                    f"pkg#{p.packageNumber} status=idle but carrier={p.carrier}"


class TestCarrierSymmetry:
    """Carrier/carrying linkage must stay symmetric after sustained operation."""

    def test_carrier_symmetry_after_200_ticks(self):
        w = _make_real_warehouse(robotsMaxQuantity=8, chargersMaxQuantity=3)
        for _ in range(200):
            w.update_warehouse()
        carried_pkgs = {p.packageNumber for p in w.packages if p.carrier not in (None, -1)}
        carrying_robots = {r.carrying for r in w.robots if r.carrying != -1}
        assert carried_pkgs == carrying_robots
        # No zombie idle packages (idle + stale areaTarget)
        for p in w.packages:
            if p.status == 'idle':
                assert p.carrier in (None, -1), \
                    f"pkg#{p.packageNumber} status=idle but carrier={p.carrier}"
                assert p.areaTarget == 'none', \
                    f"pkg#{p.packageNumber} status=idle but areaTarget={p.areaTarget}"


@pytest.mark.slow
class TestStress:
    """Long-running stability tests (skipped by default, run with -m slow)."""

    def test_2000_tick_stability(self):
        from collections import deque
        w = _make_real_warehouse(robotsMaxQuantity=12, chargersMaxQuantity=4)
        for _ in range(2000):
            w.update_warehouse()
        assert w.warehouseLoopCount == 2000
        # No orphaned carriers
        carried_pkgs = {p.packageNumber for p in w.packages if p.carrier not in (None, -1)}
        carrying_robots = {r.carrying for r in w.robots if r.carrying != -1}
        assert carried_pkgs == carrying_robots
        # No idle packages with a carrier set or stale areaTarget
        for p in w.packages:
            if p.status == 'idle':
                assert p.carrier in (None, -1), \
                    f"pkg#{p.packageNumber} status=idle but carrier={p.carrier}"
                assert p.areaTarget == 'none', \
                    f"pkg#{p.packageNumber} status=idle but areaTarget={p.areaTarget}"
        # All entity coords within grid bounds
        _w, _h = w.warehouseWindowRes
        for r in w.robots:
            assert 0 <= int(r.xyLocation[0]) < _w, f"robot#{r.robotNumber} x={r.xyLocation[0]} out of grid"
            assert 0 <= int(r.xyLocation[1]) < _h, f"robot#{r.robotNumber} y={r.xyLocation[1]} out of grid"
        for p in w.packages:
            assert 0 <= int(p.xyLocation[0]) < _w, f"pkg#{p.packageNumber} x={p.xyLocation[0]} out of grid"
            assert 0 <= int(p.xyLocation[1]) < _h, f"pkg#{p.packageNumber} y={p.xyLocation[1]} out of grid"
        # Log buffers stayed bounded
        assert isinstance(w.packagesLog, deque)
        assert isinstance(w.robotsLog, deque)
        assert len(w.packagesLog) <= w.packagesLog.maxlen
        assert len(w.robotsLog) <= w.robotsLog.maxlen
