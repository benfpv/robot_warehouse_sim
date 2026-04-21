"""ViewModel layer between simulation state and UI panels.

The Presenter computes per-frame derived quantities (aggregates, sortings,
status counts) once, into immutable frozen dataclasses, so multiple panels can
read the same values without recomputing them. Pure functions — no I/O, no
rendering, no mutation of warehouse state.

Tested in tests/test_dashboard_presenter.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FleetSnapshot:
    """Robot-fleet aggregate counts + averages, computed once per frame."""
    n: int                 # total robots
    n_idle: int            # status == 'idle'
    n_charging: int        # status == 'charging'
    n_enroute: int         # status == 'move to charging station'
    n_working: int         # all other statuses (active task)
    avg_battery: float     # mean batteryPercent (0 if no robots)
    avg_drain: float       # mean batteryDepletingRate * batteryDrainMultiplier


def build_fleet_snapshot(robots) -> FleetSnapshot:
    """Compute a FleetSnapshot from a sequence of Robot instances.

    Single-pass O(n). Safe with empty fleets (returns zeros).
    """
    n = len(robots)
    if n == 0:
        return FleetSnapshot(0, 0, 0, 0, 0, 0.0, 0.0)
    n_idle = n_charging = n_enroute = 0
    sum_batt = 0.0
    sum_drain = 0.0
    for r in robots:
        s = r.status
        if s == 'idle':
            n_idle += 1
        elif s == 'charging':
            n_charging += 1
        elif s == 'move to charging station':
            n_enroute += 1
        sum_batt  += r.batteryPercent
        sum_drain += r.batteryDepletingRate * r.batteryDrainMultiplier
    n_working = n - n_idle - n_charging - n_enroute
    return FleetSnapshot(
        n=n,
        n_idle=n_idle,
        n_charging=n_charging,
        n_enroute=n_enroute,
        n_working=n_working,
        avg_battery=sum_batt / n,
        avg_drain=sum_drain / n,
    )
