"""Tests for O(1) entity lookup: _rebuild_entity_indices, _find_robot,
_find_package, _find_charger_at."""
import pytest
from conftest import make_robot, make_package, make_charger, WarehouseStub


# ═══════════════════════════════════════════════════════════════════════
# _rebuild_entity_indices
# ═══════════════════════════════════════════════════════════════════════

class TestRebuildEntityIndices:
    def test_builds_robot_idx(self):
        r0 = make_robot(robotNumber=10)
        r1 = make_robot(robotNumber=20)
        wh = WarehouseStub(robots=[r0, r1])
        wh._rebuild_entity_indices()
        assert wh._robot_idx == {10: 0, 20: 1}

    def test_builds_pkg_idx(self):
        p0 = make_package(packageNumber=100)
        p1 = make_package(packageNumber=200)
        wh = WarehouseStub(packages=[p0, p1])
        wh._rebuild_entity_indices()
        assert wh._pkg_idx == {100: 0, 200: 1}

    def test_empty_lists(self):
        wh = WarehouseStub()
        wh._rebuild_entity_indices()
        assert wh._robot_idx == {}
        assert wh._pkg_idx == {}

    def test_rebuild_after_append(self):
        wh = WarehouseStub(robots=[make_robot(robotNumber=1)])
        wh._rebuild_entity_indices()
        assert wh._robot_idx == {1: 0}
        wh.robots.append(make_robot(robotNumber=2))
        wh._rebuild_entity_indices()
        assert wh._robot_idx == {1: 0, 2: 1}

    def test_rebuild_after_removal(self):
        wh = WarehouseStub(robots=[
            make_robot(robotNumber=1),
            make_robot(robotNumber=2),
            make_robot(robotNumber=3),
        ])
        wh._rebuild_entity_indices()
        wh.robots.pop(1)  # remove robot 2
        wh._rebuild_entity_indices()
        assert wh._robot_idx == {1: 0, 3: 1}


# ═══════════════════════════════════════════════════════════════════════
# _find_robot
# ═══════════════════════════════════════════════════════════════════════

class TestFindRobot:
    def test_cache_hit(self):
        r = make_robot(robotNumber=5)
        wh = WarehouseStub(robots=[r])
        wh._rebuild_entity_indices()
        assert wh._find_robot(5) == 0

    def test_cache_miss_rebuilds(self):
        """If cache is stale, _find_robot rebuilds and returns correct index."""
        r0 = make_robot(robotNumber=10)
        r1 = make_robot(robotNumber=20)
        wh = WarehouseStub(robots=[r0, r1])
        # Don't build cache → getattr fallback
        assert wh._find_robot(20) == 1

    def test_not_found_returns_none(self):
        wh = WarehouseStub(robots=[make_robot(robotNumber=1)])
        wh._rebuild_entity_indices()
        assert wh._find_robot(999) is None

    def test_empty_list_returns_none(self):
        wh = WarehouseStub()
        assert wh._find_robot(0) is None

    def test_stale_cache_after_removal(self):
        """After popping a robot, a stale cache is corrected."""
        r0 = make_robot(robotNumber=0)
        r1 = make_robot(robotNumber=1)
        r2 = make_robot(robotNumber=2)
        wh = WarehouseStub(robots=[r0, r1, r2])
        wh._rebuild_entity_indices()
        # Cache: {0:0, 1:1, 2:2}
        wh.robots.pop(1)  # remove robot 1; now [r0, r2]
        # Cache says robot 2 is at index 2, but list is now length 2
        assert wh._find_robot(2) == 1  # should auto-rebuild

    def test_stale_cache_wrong_robot_at_index(self):
        """Cache points to index that now holds a different robot."""
        r0 = make_robot(robotNumber=0)
        r1 = make_robot(robotNumber=1)
        wh = WarehouseStub(robots=[r0, r1])
        wh._rebuild_entity_indices()
        # Swap order
        wh.robots[0], wh.robots[1] = wh.robots[1], wh.robots[0]
        # Cache is stale: says robot 0 is at index 0, but index 0 now holds robot 1
        assert wh._find_robot(0) == 1  # should rebuild and find correct index

    def test_multiple_lookups(self):
        robots = [make_robot(robotNumber=i) for i in range(10)]
        wh = WarehouseStub(robots=robots)
        wh._rebuild_entity_indices()
        for i in range(10):
            assert wh._find_robot(i) == i


# ═══════════════════════════════════════════════════════════════════════
# _find_package
# ═══════════════════════════════════════════════════════════════════════

class TestFindPackage:
    def test_cache_hit(self):
        p = make_package(packageNumber=42)
        wh = WarehouseStub(packages=[p])
        wh._rebuild_entity_indices()
        assert wh._find_package(42) == 0

    def test_cache_miss_rebuilds(self):
        p = make_package(packageNumber=99)
        wh = WarehouseStub(packages=[p])
        assert wh._find_package(99) == 0

    def test_not_found_returns_none(self):
        wh = WarehouseStub(packages=[make_package(packageNumber=1)])
        wh._rebuild_entity_indices()
        assert wh._find_package(999) is None

    def test_empty_list(self):
        wh = WarehouseStub()
        assert wh._find_package(0) is None

    def test_stale_cache_after_removal(self):
        pkgs = [make_package(packageNumber=i) for i in range(5)]
        wh = WarehouseStub(packages=pkgs)
        wh._rebuild_entity_indices()
        wh.packages.pop(2)  # remove package 2
        assert wh._find_package(3) == 2  # should auto-rebuild
        assert wh._find_package(2) is None  # removed

    def test_stale_cache_index_out_of_range(self):
        """Cache index exceeds list length after removals."""
        pkgs = [make_package(packageNumber=i) for i in range(3)]
        wh = WarehouseStub(packages=pkgs)
        wh._rebuild_entity_indices()
        wh.packages.pop()  # remove last
        wh.packages.pop()  # remove second
        # Only package 0 remains; old cache had {0:0, 1:1, 2:2}
        assert wh._find_package(0) == 0
        assert wh._find_package(1) is None
        assert wh._find_package(2) is None


# ═══════════════════════════════════════════════════════════════════════
# _find_charger_at
# ═══════════════════════════════════════════════════════════════════════

class TestFindChargerAt:
    def test_found(self):
        c = make_charger(chargerNumber=0, xyLocation=[5, 5])
        wh = WarehouseStub(chargers=[c])
        charger, idx = wh._find_charger_at([5, 5])
        assert charger is c
        assert idx == 0

    def test_not_found(self):
        c = make_charger(xyLocation=[5, 5])
        wh = WarehouseStub(chargers=[c])
        charger, idx = wh._find_charger_at([1, 1])
        assert charger is None
        assert idx == -1

    def test_empty_chargers(self):
        wh = WarehouseStub()
        charger, idx = wh._find_charger_at([5, 5])
        assert charger is None
        assert idx == -1

    def test_multiple_chargers_returns_first_match(self):
        c0 = make_charger(chargerNumber=0, xyLocation=[5, 5])
        c1 = make_charger(chargerNumber=1, xyLocation=[10, 10])
        c2 = make_charger(chargerNumber=2, xyLocation=[5, 5])
        wh = WarehouseStub(chargers=[c0, c1, c2])
        charger, idx = wh._find_charger_at([5, 5])
        assert charger is c0
        assert idx == 0

    def test_finds_second_charger(self):
        c0 = make_charger(chargerNumber=0, xyLocation=[1, 1])
        c1 = make_charger(chargerNumber=1, xyLocation=[5, 5])
        wh = WarehouseStub(chargers=[c0, c1])
        charger, idx = wh._find_charger_at([5, 5])
        assert charger is c1
        assert idx == 1
