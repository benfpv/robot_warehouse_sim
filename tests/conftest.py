"""Shared fixtures and factories for warehouse simulation tests."""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

from data.warehouse.robot import Robot
from data.warehouse.package import Package
from data.warehouse.charger import Charger
from data.constants import SIM_TICK_RATE


# ---------------------------------------------------------------------------
# Entity factories
# ---------------------------------------------------------------------------

def make_robot(**kw):
    """Create a Robot with sensible defaults.  Override any attribute via kwargs."""
    defaults = dict(
        robotNumber=0,
        robotLog=[],
        timerCheckBattery=0.0,
        batteryPercent=100.0,
        batteryChargingRate=1.0,
        batteryDepletingRate=0.1,
        actionQueue=[],
        colour=[255, 255, 255],
        area='import',
        xyLocation=[10, 10],
        areaTarget='import',
        xyLocationTarget=[10, 10],
        xyLocationDiff=[0, 0],
        direction=0,
        cardinal='',
        velocity=0,
        status='idle',
        carrying=-1,
        carrier=-1,
    )
    defaults.update(kw)
    return Robot(**defaults)


def make_package(**kw):
    """Create a Package with sensible defaults."""
    from datetime import datetime, timedelta
    defaults = dict(
        packageNumber=0,
        packageLog=[],
        itemValues=None,
        addressFrom=None,
        addressTo=None,
        deadline=datetime.now() + timedelta(hours=1),
        timeToDeadline=timedelta(hours=1),
        colour=[0, 200, 0],
        area='import',
        xyLocation=[20, 20],
        areaTarget='storage',
        xyLocationTarget=[30, 30],
        status='idle',
        carrier=-1,
    )
    defaults.update(kw)
    return Package(**defaults)


def make_charger(**kw):
    """Create a Charger with sensible defaults."""
    defaults = dict(
        chargerNumber=0,
        colour=[0, 255, 255],
        area='none',
        xyLocation=[5, 5],
        status='idle',
    )
    defaults.update(kw)
    return Charger(**defaults)


# ---------------------------------------------------------------------------
# Warehouse stub
# ---------------------------------------------------------------------------

class WarehouseStub:
    """Lightweight stand-in for Warehouse.

    Real method implementations are copied from the Warehouse class so logic
    is tested faithfully; heavyweight dependencies (pathfinding, display, I/O)
    are left as MagicMocks.
    """

    @property
    def _sim_time(self):
        return self.warehouseLoopCount / SIM_TICK_RATE

    def __init__(self, *, width=20, height=20, robots=None, packages=None,
                 chargers=None):
        self.warehouseWindowRes = (width, height)
        self.warehouseLoopCount = 0

        # Entity lists
        self.robots = list(robots) if robots is not None else []
        self.packages = list(packages) if packages is not None else []
        self.chargers = list(chargers) if chargers is not None else []

        # Grids
        self.packagesInWarehouse = np.zeros((height, width), dtype='uint8')
        self.robotsInWarehouse = np.zeros((height, width), dtype='uint8')
        self.chargersInWarehouse = np.zeros((height, width), dtype='uint8')
        self.packageTargetsInWarehouse = np.zeros((height, width), dtype='uint8')
        self.zoneMap = np.zeros((height, width), dtype='uint8')
        self.road_map = np.zeros((height, width), dtype='uint8')

        # Task management
        self.robotsTaskAssignmentList = []
        self.robotsLog = []
        self.packagesMoveList = []
        self.packagesMovingList = []
        self.packagesReorgList = []
        self.packagesPlannedInImportCount = 0
        self.packagesPlannedInStorageCount = 0
        self.packagesPlannedInExportCount = 0
        self.robotsInWarehouseCount = len(self.robots)
        self.robotsMaxQuantity = 4
        self.packagesMaxQuantity = 60
        self.packagesMaxMoveQuantity = 16
        self._flow_pipeline_depth = 3.0
        self._flow_import_cap = 0
        self._flow_avg_deadline = 0.0
        self._flow_avg_delivery = 0.0
        self._flow_ema_alpha = 0.02
        self.pending_zone_changes = set()

        # Battery/charging policy
        self._power_policy_mode = 'balanced'
        self._limp_mode_batt_pct = 20.0
        self._work_min_batt_pct = 20.0
        self._critical_batt_pct = 8.0
        self._charge_threshold_base = 35.0
        self._charge_threshold_span = 20.0
        self._power_policy_alpha_pressure = 0.025
        self._power_policy_alpha_threshold = 0.020
        self._charger_pressure_raw = 0.0
        self._charger_pressure = 0.0
        self._avg_fleet_battery = 100.0
        self._charge_threshold_target = 35.0
        self._charge_threshold = 35.0

        # Mocked side-effect helpers
        self.robot_pop_task = MagicMock(side_effect=self._mock_pop_task)
        self.robot_insert_task = MagicMock(side_effect=self._mock_insert_task)
        self.robot_replace_task = MagicMock(side_effect=self._mock_replace_task)
        self._compute_robot_path = MagicMock()
        self.find_available_charger = MagicMock(return_value=(False, -1))

    # -- Mock helpers for task management ----------------------------------

    def _mock_pop_task(self, robotIndex, actionToPop):
        robot = self.robots[robotIndex]
        for idx, task in enumerate(robot.actionQueue):
            if actionToPop in task[1]:
                robot.actionQueue.pop(idx)
                break
        if not robot.actionQueue:
            robot.status = 'idle'
        else:
            robot.status = robot.actionQueue[0][1]
        return robot, self.robotsTaskAssignmentList, self.robotsLog

    def _mock_insert_task(self, robotIndex, actionIndex, location, action):
        robot = self.robots[robotIndex]
        robot.actionQueue.insert(actionIndex, [location, action])
        robot.status = robot.actionQueue[0][1]
        return robot, self.robotsTaskAssignmentList, self.robotsLog

    def _mock_replace_task(self, robotIndex, actionToPop, newLocation,
                           newAction):
        robot = self.robots[robotIndex]
        pop_idx = 0
        for idx, task in enumerate(robot.actionQueue):
            if actionToPop in task[1]:
                pop_idx = idx
                robot.actionQueue.pop(idx)
                break
        robot.actionQueue.insert(pop_idx, [newLocation, newAction])
        robot.status = robot.actionQueue[0][1]
        return robot, self.robotsTaskAssignmentList, self.robotsLog


# Copy real Warehouse methods onto the stub so tests exercise actual logic.
#
# Stub contract / drift guard:
#   The names below MUST exist on the real Warehouse class. They are copied
#   verbatim so the stub exercises production logic rather than a re-implementation.
#   If a method is renamed or removed in warehouse.py without updating this list,
#   the drift guard at the bottom of this block raises at collection time instead
#   of silently leaving the stub with stale/missing behaviour.
from data.warehouse.warehouse import Warehouse as _RealWarehouse

_REAL_METHOD_NAMES = (
    '_rebuild_entity_indices',
    '_find_robot',
    '_find_package',
    '_find_charger_at',
    '_find_nearest_empty_cell',
    '_reconcile_charger_reservations',
    '_update_charge_policy_metrics',
    '_robot_consume_waypoints',
    '_robot_dock_charger',
    '_robot_charge_tick',
    '_robot_do_pickup',
    '_robot_do_dropoff',
    '_dispatch_charging_if_needed',
    '_emergency_battery_drop',
    '_robot_step_movement',
    '_robot_recover_stuck',
    '_robot_detect_oscillation',
    'package_dropoff',
    '_rotate_toward_deg',
    '_set_package_idle',
    '_check_invariants_lightweight',
    '_remove_robot',
    '_rescale_flow_config',
    'update_packages_in_warehouse',
    'update_robots_in_warehouse',
    'update_chargers_in_warehouse',
    'recount_planned',
    'update_flow_control',
    'reconcile_zone_changes',
)

# Static methods copied verbatim (preserve @staticmethod descriptor).
_REAL_STATIC_NAMES = (
    '_robot_is_stopped',
    '_estimate_charge_travel_cost',
    '_normalize_deg',
)

# Drift guard: fail loudly at collection time if any listed name no longer
# exists on the real Warehouse class (e.g. after a rename), instead of
# silently leaving the stub with stale behaviour.
_missing = [
    name for name in (_REAL_METHOD_NAMES + _REAL_STATIC_NAMES)
    if name not in _RealWarehouse.__dict__
]
if _missing:
    raise RuntimeError(
        "WarehouseStub drift detected: the following names are listed in "
        "tests/conftest.py but no longer exist on Warehouse: "
        + ", ".join(_missing)
        + ". Update the copy lists to match warehouse.py."
    )

for _name in _REAL_METHOD_NAMES + _REAL_STATIC_NAMES:
    setattr(WarehouseStub, _name, _RealWarehouse.__dict__[_name])


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wh():
    """A fresh WarehouseStub for each test."""
    return WarehouseStub()
