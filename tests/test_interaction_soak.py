"""Deterministic interaction soak tests.

These tests simulate repeated user-like interaction patterns while ticking the
real warehouse to protect long-term UX behavior and prevent regression debt.
"""

import logging
import random
from contextlib import contextmanager

import pytest

from data.functions import Functions
from data.importer import Importer
from data.warehouse import warehouse as warehouse_module
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


@contextmanager
def _capture_warehouse_logs():
    """Capture records emitted by the warehouse module logger."""
    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler()
    warehouse_module._log.addHandler(handler)
    try:
        yield records
    finally:
        warehouse_module._log.removeHandler(handler)


def _assert_no_invariant_error_logs(records):
    """Fail when invariant-repair/error patterns appear during normal soak."""
    markers = (
        "INVARIANT",
        "INVARIANT-LITE",
        "reconcile_zone_changes: failed",
        "out-of-grid",
    )
    bad = [
        rec.getMessage()
        for rec in records
        if rec.levelno >= logging.ERROR
        and any(marker in rec.getMessage() for marker in markers)
    ]
    assert not bad, "Unexpected invariant/error logs during soak: {}".format(bad[:5])


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
        with _capture_warehouse_logs() as records:
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
                assert w._power_policy_mode in power_modes
                assert w._flow_policy_mode in flow_modes
                assert w._pkg_target_mode in target_modes

        _assert_no_invariant_error_logs(records)

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

    @pytest.mark.parametrize("seed", [20260420, 20260421, 20260422])
    def test_multi_seed_interaction_churn_has_clean_error_signals(self, seed):
        """Different deterministic interaction traces should stay robust and clean."""
        rng = random.Random(seed)
        w = _make_real_warehouse(robotsMaxQuantity=7, chargersMaxQuantity=3)

        power_modes = ["eco", "balanced", "performance"]
        flow_modes = ["steady", "balanced", "throughput"]
        target_modes = ["random", "nearest", "zone_edge"]

        with _capture_warehouse_logs() as records:
            for t in range(420):
                if t % 19 == 0:
                    assert w.set_power_policy_mode(rng.choice(power_modes)) is True
                if t % 23 == 0:
                    assert w.set_flow_policy_mode(rng.choice(flow_modes)) is True
                if t % 29 == 0:
                    assert w.set_package_target_mode(rng.choice(target_modes)) is True

                # Simulate UI robot +/- interactions while keeping valid bounds.
                if t % 37 == 0:
                    delta = rng.choice([-1, 1])
                    target = max(1, min(12, len(w.robots) + delta))
                    w.set_robot_count(target)

                # Simulate paint bursts in compact clusters (common user pattern).
                if t % 13 == 0:
                    cx = rng.randrange(w.warehouseWindowRes[0])
                    cy = rng.randrange(w.warehouseWindowRes[1])
                    for _ in range(10):
                        x = max(0, min(w.warehouseWindowRes[0] - 1, cx + rng.randint(-2, 2)))
                        y = max(0, min(w.warehouseWindowRes[1] - 1, cy + rng.randint(-2, 2)))
                        zid = rng.choice([0, 1, 2, 3])
                        w.zoneMap[y][x] = zid
                        w.pending_zone_changes.add((x, y))

                w.update_warehouse()

                # Lists should not accumulate duplicates under churn.
                assert len(w.packagesMoveList) == len(set(w.packagesMoveList))
                assert len(w.packagesMovingList) == len(set(w.packagesMovingList))
                assert len(w.packagesReorgList) == len(set(w.packagesReorgList))

        _assert_no_invariant_error_logs(records)

        # End-state sanity checks.
        carried_pkgs = {p.packageNumber for p in w.packages if p.status == "carried"}
        carrying_robots = {r.carrying for r in w.robots if r.carrying != -1}
        assert carried_pkgs == carrying_robots
