"""Unit tests for `data.ui.presenter` — pure ViewModel computations."""

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from data.ui.presenter import FleetSnapshot, build_fleet_snapshot


def _robot(status, batt=50.0, drain=0.01, mult=1.0):
    return SimpleNamespace(
        status=status,
        batteryPercent=batt,
        batteryDepletingRate=drain,
        batteryDrainMultiplier=mult,
    )


class TestBuildFleetSnapshotEmpty:
    def test_empty_returns_zeros(self):
        snap = build_fleet_snapshot([])
        assert snap.n == 0
        assert snap.n_idle == 0
        assert snap.n_charging == 0
        assert snap.n_enroute == 0
        assert snap.n_working == 0
        assert snap.avg_battery == 0.0
        assert snap.avg_drain == 0.0


class TestBuildFleetSnapshotCounts:
    def test_status_counts(self):
        robots = [
            _robot('idle'),
            _robot('idle'),
            _robot('charging'),
            _robot('move to charging station'),
            _robot('move to target pickup location'),  # working
            _robot('drop off target package'),         # working
        ]
        snap = build_fleet_snapshot(robots)
        assert snap.n == 6
        assert snap.n_idle == 2
        assert snap.n_charging == 1
        assert snap.n_enroute == 1
        assert snap.n_working == 2

    def test_working_includes_all_unknown_active_statuses(self):
        # Anything that is not idle / charging / en-route is "working"
        robots = [_robot('weird-future-status') for _ in range(4)]
        snap = build_fleet_snapshot(robots)
        assert snap.n_working == 4
        assert snap.n_idle == snap.n_charging == snap.n_enroute == 0

    def test_counts_partition_total(self):
        robots = [
            _robot('idle'), _robot('charging'), _robot('move to charging station'),
            _robot('pick up target package'), _robot('idle'),
        ]
        snap = build_fleet_snapshot(robots)
        assert snap.n_idle + snap.n_charging + snap.n_enroute + snap.n_working == snap.n


class TestBuildFleetSnapshotAggregates:
    def test_avg_battery(self):
        robots = [_robot('idle', batt=20), _robot('idle', batt=80)]
        snap = build_fleet_snapshot(robots)
        assert snap.avg_battery == pytest.approx(50.0)

    def test_avg_drain_multiplies_drain_and_multiplier(self):
        robots = [
            _robot('idle', drain=0.02, mult=2.0),  # 0.04
            _robot('idle', drain=0.01, mult=4.0),  # 0.04
        ]
        snap = build_fleet_snapshot(robots)
        assert snap.avg_drain == pytest.approx(0.04)


class TestFleetSnapshotImmutability:
    def test_is_frozen(self):
        snap = build_fleet_snapshot([_robot('idle')])
        with pytest.raises(FrozenInstanceError):
            snap.n = 99

    def test_has_expected_fields(self):
        names = {f.name for f in fields(FleetSnapshot)}
        assert names == {
            'n', 'n_idle', 'n_charging', 'n_enroute', 'n_working',
            'avg_battery', 'avg_drain',
        }


class TestBuildFleetSnapshotPurity:
    def test_does_not_mutate_input(self):
        robots = [_robot('idle', batt=42)]
        before = robots[0].__dict__.copy()
        build_fleet_snapshot(robots)
        assert robots[0].__dict__ == before

    def test_idempotent(self):
        robots = [_robot('idle'), _robot('charging', batt=30)]
        a = build_fleet_snapshot(robots)
        b = build_fleet_snapshot(robots)
        assert a == b
