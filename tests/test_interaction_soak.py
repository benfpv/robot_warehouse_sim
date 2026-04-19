"""Deterministic interaction soak tests.

These tests simulate repeated user-like interaction patterns while ticking the
real warehouse to protect long-term UX behavior and prevent regression debt.
"""

import random

import pytest

from data.functions import Functions
from data.importer import Importer
from data.warehouse.warehouse import Warehouse


def _make_real_warehouse(**overrides):
    res = (80, 70)
    bg = [20, 20, 20]
    center = Functions.get_screencenter(res)
    array = Functions.get_screenarray_colour(res, bg)
    items = Importer.init_import_csv_as_list("resources/list_items.csv")
    items = Importer.init_objectify_items_list(items)
    addrs = Importer.init_import_csv_as_list("resources/list_addresses.csv")
    addrs = Importer.init_objectify_addresses_list(addrs)
    kw = dict(robotsMaxQuantity=6, chargersMaxQuantity=3)
    kw.update(overrides)
    return Warehouse(res, bg, center, array, items, addrs, **kw)


@pytest.mark.slow
class TestDeterministicInteractionSoak:
    """High-value long-run interaction sequence checks."""

    def test_policy_switching_and_zone_edits_remain_consistent(self):
        rng = random.Random(20260419)
        w = _make_real_warehouse()

        power_modes = ["eco", "balanced", "performance"]
        flow_modes = ["steady", "balanced", "throughput"]
        target_modes = ["random", "nearest", "zone_edge"]

        ticks = 600
        for t in range(ticks):
            # Simulate occasional user policy interactions.
            if t % 41 == 0:
                assert w.set_power_policy_mode(rng.choice(power_modes)) is True
            if t % 53 == 0:
                assert w.set_flow_policy_mode(rng.choice(flow_modes)) is True
            if t % 47 == 0:
                assert w.set_package_target_mode(rng.choice(target_modes)) is True

            # Simulate user-painted zone edits by mutating a handful of random cells.
            if t % 17 == 0:
                for _ in range(8):
                    x = rng.randrange(w.warehouseWindowRes[0])
                    y = rng.randrange(w.warehouseWindowRes[1])
                    zid = rng.choice([0, 1, 2, 3])
                    w.zoneMap[y][x] = zid
                    w.pending_zone_changes.add((x, y))

            w.update_warehouse()

            # Ongoing UX-critical sanity checks each tick.
            assert len(w.packagesMoveList) == len(set(w.packagesMoveList))
            assert len(w.packagesMovingList) == len(set(w.packagesMovingList))
            assert len(w.packagesReorgList) == len(set(w.packagesReorgList))

        # Post-run integrity assertions.
        assert w.warehouseLoopCount == ticks

        robot_ids = [r.robotNumber for r in w.robots]
        package_ids = [p.packageNumber for p in w.packages]
        assert len(robot_ids) == len(set(robot_ids))
        assert len(package_ids) == len(set(package_ids))

        # Carrier symmetry after sustained mixed interactions.
        carried_pkgs = {p.packageNumber for p in w.packages if p.status == "carried"}
        carrying_robots = {r.carrying for r in w.robots if r.carrying != -1}
        assert carried_pkgs == carrying_robots

        # Core occupancy bounds safety.
        width, height = w.warehouseWindowRes
        for r in w.robots:
            assert 0 <= int(r.xyLocation[0]) < width
            assert 0 <= int(r.xyLocation[1]) < height
        for p in w.packages:
            assert 0 <= int(p.xyLocation[0]) < width
            assert 0 <= int(p.xyLocation[1]) < height
        for c in w.chargers:
            assert 0 <= int(c.xyLocation[0]) < width
            assert 0 <= int(c.xyLocation[1]) < height
