"""Tests for data.constants — shared simulation constants."""
import pytest
from data.constants import (
    ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT, ZONE_NAMES,
    SIM_TICK_RATE, STALL_TICKS, ROBOT_CARRY_SPEED_PENALTY, BATTERY_FULL_PCT,
)


# ── Value tests ────────────────────────────────────────────────────────

class TestZoneConstants:
    def test_zone_none_is_zero(self):
        assert ZONE_NONE == 0

    def test_zone_import_is_one(self):
        assert ZONE_IMPORT == 1

    def test_zone_storage_is_two(self):
        assert ZONE_STORAGE == 2

    def test_zone_export_is_three(self):
        assert ZONE_EXPORT == 3

    def test_zone_ids_are_unique(self):
        ids = [ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT]
        assert len(ids) == len(set(ids))

    def test_zone_ids_are_int(self):
        for z in (ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT):
            assert isinstance(z, int)


class TestZoneNames:
    def test_zone_names_is_dict(self):
        assert isinstance(ZONE_NAMES, dict)

    def test_zone_names_has_all_zones(self):
        for zid in (ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT):
            assert zid in ZONE_NAMES

    def test_zone_names_values_are_strings(self):
        for v in ZONE_NAMES.values():
            assert isinstance(v, str)

    def test_zone_names_none_maps_to_neutral(self):
        assert ZONE_NAMES[ZONE_NONE] == 'neutral'

    def test_zone_names_import(self):
        assert ZONE_NAMES[ZONE_IMPORT] == 'import'

    def test_zone_names_storage(self):
        assert ZONE_NAMES[ZONE_STORAGE] == 'storage'

    def test_zone_names_export(self):
        assert ZONE_NAMES[ZONE_EXPORT] == 'export'

    def test_zone_names_no_extra_entries(self):
        assert len(ZONE_NAMES) == 4


class TestTimingConstants:
    def test_sim_tick_rate_is_40(self):
        assert SIM_TICK_RATE == 40

    def test_sim_tick_rate_is_int(self):
        assert isinstance(SIM_TICK_RATE, int)

    def test_stall_ticks_equals_sim_tick_rate(self):
        assert STALL_TICKS == SIM_TICK_RATE

    def test_stall_ticks_is_int(self):
        assert isinstance(STALL_TICKS, int)


class TestRobotTuning:
    def test_carry_speed_penalty_value(self):
        assert ROBOT_CARRY_SPEED_PENALTY == 0.92

    def test_carry_speed_penalty_is_float(self):
        assert isinstance(ROBOT_CARRY_SPEED_PENALTY, float)

    def test_carry_speed_penalty_less_than_one(self):
        assert 0.0 < ROBOT_CARRY_SPEED_PENALTY < 1.0

    def test_battery_full_pct_value(self):
        assert BATTERY_FULL_PCT == 98

    def test_battery_full_pct_is_int(self):
        assert isinstance(BATTERY_FULL_PCT, int)

    def test_battery_full_pct_below_100(self):
        assert BATTERY_FULL_PCT < 100

    def test_battery_full_pct_positive(self):
        assert BATTERY_FULL_PCT > 0


# ── No-duplication check ──────────────────────────────────────────────

class TestNoDuplication:
    """Verify that critical constants are imported from data.constants,
    not re-defined elsewhere."""

    def test_warehouse_imports_constants(self):
        """warehouse.py should import from data.constants, not define its own."""
        import data.warehouse.warehouse as wmod
        assert wmod.SIM_TICK_RATE is SIM_TICK_RATE
        assert wmod.STALL_TICKS is STALL_TICKS
        assert wmod.ROBOT_CARRY_SPEED_PENALTY is ROBOT_CARRY_SPEED_PENALTY
        assert wmod.BATTERY_FULL_PCT is BATTERY_FULL_PCT
        assert wmod.ZONE_NONE is ZONE_NONE
        assert wmod.ZONE_IMPORT is ZONE_IMPORT
        assert wmod.ZONE_STORAGE is ZONE_STORAGE
        assert wmod.ZONE_EXPORT is ZONE_EXPORT
