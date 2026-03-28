"""Core warehouse simulation engine — tick loop, flow control, scheduling, and pathfinding."""
from datetime import date, datetime, timedelta
import math
import random
import logging
from collections import deque
import numpy as np
import cv2 
import time

from data.functions import *
from data.functions_timeseries import *
from data.warehouse.warehouse_init import *
from data.warehouse.package import *
from data.warehouse.package_functions import *
from data.warehouse.charger import *
from data.warehouse.charger_functions import *
from data.warehouse.robot import *
from data.warehouse.robot_functions import *
from data.warehouse.lane_planner import LanePathPlanner
from data.warehouse.pathfinder import AStarPathfinder
from data.warehouse.warehouse_data import *

ZONE_NONE = 0
ZONE_IMPORT = 1
ZONE_STORAGE = 2
ZONE_EXPORT = 3

# ── Module-level logger ────────────────────────────────────────────────
_log = logging.getLogger('warehouse')
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    _fh = logging.FileHandler('warehouse.log', mode='w', encoding='utf-8')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-5s  %(message)s', datefmt='%H:%M:%S'))
    _log.addHandler(_fh)
    _log.propagate = False

class Warehouse:
    """Core warehouse simulation engine.

    Manages an 80×70 grid with import/storage/export zones, a fleet of
    autonomous robots, charging stations, and a package lifecycle from
    import → storage → export.  Each tick (40 Hz) runs:

        update_warehouse()
          ├─ update_zone_counts       recompute slot capacity
          ├─ update_flow_control      adaptive import throttle
          ├─ reconcile_zone_changes   handle user-painted zone edits
          ├─ invalidate_stale_targets  cancel moves to wrong-zone cells
          ├─ recount_planned          authoritative planned-count refresh
          ├─ update_packages          spawn, deadline, move-list, targets, colours, export
          ├─ update_chargers          spawn chargers up to max
          ├─ update_robots            spawn, assign, pathfind, move, battery, idle-park
          ├─ update_carried_packages  sync carried-package positions to robot
          ├─ update_packages_areas    recount per-zone package counts
          ├─ update occupancy arrays  packagesInWarehouse, chargersInWarehouse, robotsInWarehouse
          ├─ update_space_available   flag zone-full booleans
          └─ record_warehouse_data    append to rolling timeseries

    Key subsystems:
        Flow control  — robot-scaled adaptive import cap (pipeline_depth × n_robots)
        Power policy  — adaptive charge threshold with pressure/eco damping
        Pathfinding   — dense A* with soft robot-occupancy costs and speed profiling
        Decommission  — graceful robot retirement with perimeter exit
    """
    _POWER_POLICY_MODES = ('eco', 'balanced', 'performance')
    _FLOW_POLICY_MODES = ('steady', 'balanced', 'throughput')
    _PKG_TARGET_MODES = ('random', 'nearest', 'zone_edge')

    def __init__(self,
                    # Window Resolution, Window Center
                    windowRes = [], windowBackgroundColour = [], windowCenter = [], windowArray = [], itemsList = [], addressesList = [],
                    # Warehouse
                    warehouseLoopCount = 0, datetimeNow = datetime.now(), timeStart = time.time(), timeElapsed = 0, warehouseWindowRes = [], warehouseBackgroundColour = [], warehouseWindowCenter = [], warehouseWindowArray = [], warehousePerimeterCoordinates = [],
                    colourOfImportAreas = (10,30,10), colourOfStorageAreas = (30,10,10), colourOfExportAreas = (10,10,30),
                    logsMaxLength = 10000, dataMaxLength = 200, warehouse_data = [],
                    # Space Availability
                    packagesInImportCount = 0, packagesInStorageCount = 0, packagesInExportCount = 0, packagesInWarehouseCount = 0,
                    packagesPlannedInImportCount = 0, packagesPlannedInStorageCount = 0, packagesPlannedInExportCount = 0,
                    importSpaceAvailable = True, storageSpaceAvailable = True, exportSpaceAvailable = True,
                    packageExportCount = 0, packageExportRollingCount = 0,
                    # Packages
                    packageDimensionsLimit = 1, packagesActionsList = ["idle", "carried"],
                    packages = [], packagesLog = [], packagesMaxQuantity = 2000,
                    packagesMoveList = [], packagesMaxMoveQuantity = 120, packagesRollingCount = 0,
                    packagesInWarehouse = [], packageTargetsInWarehouse = [],
                    # Robots
                    packagesMovingList = [], robotsActionsList = ["none", "idle", "charging", "move to charging station", "move to target pickup location", "move to target dropoff location", "pickup target package", "dropoff target package", "move to exit", "move to idle"],
                    robots = [], robotsLog = [], robotsInWarehouse = [], robotsMaxQuantity = 60, robotsRollingCount = 0, robotsInWarehouseCount = 0,
                    robotsTaskAssignmentList = [], robotsTaskAssignmentStyle = 0, robotsTaskAssignmentMaxQuantity = 1,
                    numRobotsIdle = 0, numRobotsMoving = 0, numRobotsCharging = 0,
                    # Charging station(s)
                    chargersActionsList = ["none", "idle", "charging planned", "charging"], 
                    chargers = [], chargersMaxQuantity = 10, chargersInWarehouse = [], chargersRollingCount = 0,
                    # Spawn maps (list of [x,y] coords; None = use default perimeter)
                    chargerSpawnMap = None, robotSpawnMap = None
                ) -> None:
        print("--- Warehouse Init ---")
        # Window Resolution, Window Center
        self.windowRes = windowRes
        self.windowBackgroundColour = windowBackgroundColour
        self.windowCenter = windowCenter
        self.windowArray = windowArray
        self.itemsList = itemsList
        self.addressesList = addressesList
        # Warehouse
        self.warehouseLoopCount = warehouseLoopCount
        self.datetimeNow = datetimeNow
        self.timeStart = timeStart
        self.timeElapsed = timeElapsed
        self.warehouseBackgroundColour = warehouseBackgroundColour
        self.warehouseWindowRes = warehouseWindowRes
        self.warehouseWindowCenter = warehouseWindowCenter
        self.warehouseWindowArray = warehouseWindowArray
        self.warehousePerimeterCoordinates = warehousePerimeterCoordinates
        self.logsMaxLength = logsMaxLength
        self.dataMaxLength = dataMaxLength
        self.warehouse_data = warehouse_data
        # Zone colours
        self.colourOfImportAreas = colourOfImportAreas
        self.colourOfStorageAreas = colourOfStorageAreas
        self.colourOfExportAreas = colourOfExportAreas
        # Areas Count and Availability
        self.packagesInImportCount = packagesInImportCount
        self.packagesInStorageCount = packagesInStorageCount
        self.packagesInExportCount = packagesInExportCount
        self.packagesInWarehouseCount = packagesInWarehouseCount
        self.packagesPlannedInImportCount = packagesPlannedInImportCount
        self.packagesPlannedInStorageCount = packagesPlannedInStorageCount
        self.packagesPlannedInExportCount = packagesPlannedInExportCount
        self.importSpaceAvailable = importSpaceAvailable
        self.storageSpaceAvailable = storageSpaceAvailable
        self.exportSpaceAvailable = exportSpaceAvailable
        # Package Export
        self.packageExportCount = packageExportCount
        self.packageExportRollingCount = packageExportRollingCount
        # Packages
        self.packageDimensionsLimit = packageDimensionsLimit
        self.packagesActionsList = packagesActionsList
        self.packages = packages #[package#, addressFrom, addressTo, itemvalues, deadline, colour, area, location, status]
        self.packagesLog = packagesLog #[package#, packagesListinput, action(import,export), datetime of action]
        self.packagesMaxQuantity = packagesMaxQuantity
        self.packagesMoveList = packagesMoveList #[package#]
        self.packagesMaxMoveQuantity = packagesMaxMoveQuantity
        self.packagesRollingCount = packagesRollingCount
        self.packagesInWarehouse = packagesInWarehouse
        self.packageTargetsInWarehouse = packageTargetsInWarehouse
        # Robots
        self.packagesMovingList = packagesMovingList
        self.packagesReorgList = []          # pkg numbers needing priority re-scheduling after a zone change
        self.pending_zone_changes = set()    # (gx,gy) cells painted this tick, consumed by reconcile
        self.robotsActionsList = robotsActionsList
        self.robots = robots #[robot#, battery%, actionQueue, robotLog, xyLoc, xyLocTarget, status]
        self.robotsLog = robotsLog
        self.robotsInWarehouse = robotsInWarehouse
        self.robotsMaxQuantity = robotsMaxQuantity
        self._robot_target_count = robotsMaxQuantity
        self.robotsRollingCount = robotsRollingCount
        self.robotsInWarehouseCount = robotsInWarehouseCount
        self.robotsTaskAssignmentList = robotsTaskAssignmentList #[robot#, #oftasks]
        self.robotsTaskAssignmentStyle = robotsTaskAssignmentStyle
        self.robotsTaskAssignmentMaxQuantity = robotsTaskAssignmentMaxQuantity
        # Charging Stations
        self.chargersActionsList = chargersActionsList
        self.chargers = chargers #[charger#, xyLocation, status]
        self.chargersMaxQuantity = chargersMaxQuantity
        self.chargersInWarehouse = chargersInWarehouse
        self.chargersRollingCount = chargersRollingCount
        # Spawn maps (None = use default perimeter coords, set after perimeter init)
        self._chargerSpawnMap_override = chargerSpawnMap
        self._robotSpawnMap_override   = robotSpawnMap
        
        # Init Warehouse Screen
        self.windowCenter = Functions.get_screencenter(self.windowRes)
        self.warehouseWindowArray = self.windowArray
        self.warehousePerimeterCoordinates, self.warehousePerimeterCoordinatesMinusOne = Warehouse_Init.init_warehousePerimeter(self.windowRes)
        self.warehouseWindowRes, self.warehouseWindowCenter, self.packagesInWarehouse, self.robotsInWarehouse, self.chargersInWarehouse, self.idleAreasInWarehouse, self.packageTargetsInWarehouse = Warehouse_Init.init_warehouseWindow(self.windowRes, self.windowCenter)
        self.zoneMap = Warehouse_Init.init_zoneMap(self.warehouseWindowRes)
        self.laneMap = None
        self.lanePlanner = None
        self.numberOfImportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_IMPORT))
        self.numberOfStorageSlots = int(np.count_nonzero(self.zoneMap == ZONE_STORAGE))
        self.numberOfExportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_EXPORT))
        self.packagesMaxQuantity  = self.numberOfImportSlots + self.numberOfStorageSlots + self.numberOfExportSlots
        # Resolve spawn maps: use override if provided, else default to computed perimeter coords
        self.chargerSpawnMap = (self._chargerSpawnMap_override
                                if self._chargerSpawnMap_override is not None
                                else self.warehousePerimeterCoordinatesMinusOne)
        self.robotSpawnMap   = (self._robotSpawnMap_override
                                if self._robotSpawnMap_override is not None
                                else self.warehousePerimeterCoordinates)
        self.warehouse_data = Warehouse_Data(self.dataMaxLength)
        # Adaptive Power Policy thresholds
        self._power_policy_mode = 'balanced'
        self._limp_mode_batt_pct = 20.0
        self._work_min_batt_pct = 20.0
        self._critical_batt_pct = 8.0
        self._charge_threshold_base = 35.0
        self._charge_threshold_span = 20.0  # max threshold = base + span

        # Adaptive Power Policy state (smoothed every tick)
        self._power_policy_name = 'Power Policy'
        self._power_policy_alpha_pressure = 0.025
        self._power_policy_alpha_threshold = 0.020
        self._power_policy_alpha_eco = 0.030
        self._power_policy_eco_strength = 0.16
        self._power_policy_eco = 1.0
        self._charger_pressure_raw = 0.0
        self._charger_pressure = 0.0    # smoothed ratio of busy chargers (0.0-1.0)
        self._avg_fleet_battery = 100.0
        self._charge_threshold_target = self._charge_threshold_base
        self._charge_threshold = self._charge_threshold_base   # smoothed dynamic charge threshold

        # ── Flow-control metrics ──
        self._flow_export_times = []     # recent delivery durations (seconds)
        self._flow_avg_delivery = 0.0    # EMA of delivery time (seconds)
        self._flow_avg_deadline = 0.0    # EMA of time-to-deadline at spawn (seconds)
        self._flow_throughput   = 0.0    # exports per second (EMA)
        self._flow_last_export_t = time.time()
        self._flow_policy_mode = 'balanced'
        self._flow_ema_alpha    = 0.02   # smoothing factor for EMAs
        # ── Conservative robot-scaled import configuration ──
        # Scale target occupancy and import cap to robot fleet size so a
        # small fleet isn't overwhelmed with packages it can't deliver.
        self._flow_target_occ = 0.0
        self._flow_import_cap = 0
        self._flow_pipeline_depth = 4  # packages-per-robot baseline (policy-dependent)
        self._rescale_flow_config()

        # Package target selection mode: 'random', 'nearest', or 'zone_edge'
        self._pkg_target_mode = 'nearest'

    def set_power_policy_mode(self, mode):
        """Override adaptive power policy profile from UI.

        Modes:
            eco        - conservative charging and stronger limp behavior
            balanced   - default profile
            performance- lower buffers for higher utilization
        """
        mode = str(mode).strip().lower()
        profiles = {
            'eco': {
                'limp': 25.0,
                'work': 25.0,
                'critical': 10.0,
                'base': 40.0,
                'span': 15.0,
                'pressure_alpha': 0.020,
                'threshold_alpha': 0.015,
                'eco_alpha': 0.025,
                'eco_strength': 0.22,
            },
            'balanced': {
                'limp': 20.0,
                'work': 20.0,
                'critical': 8.0,
                'base': 35.0,
                'span': 20.0,
                'pressure_alpha': 0.025,
                'threshold_alpha': 0.020,
                'eco_alpha': 0.030,
                'eco_strength': 0.16,
            },
            'performance': {
                'limp': 15.0,
                'work': 15.0,
                'critical': 6.0,
                'base': 32.0,
                'span': 13.0,
                'pressure_alpha': 0.030,
                'threshold_alpha': 0.025,
                'eco_alpha': 0.035,
                'eco_strength': 0.12,
            },
        }
        cfg = profiles.get(mode)
        if not cfg:
            return False
        self._power_policy_mode = mode
        self._limp_mode_batt_pct = cfg['limp']
        self._work_min_batt_pct = cfg['work']
        self._critical_batt_pct = cfg['critical']
        self._charge_threshold_base = cfg['base']
        self._charge_threshold_span = cfg['span']
        self._power_policy_alpha_pressure = cfg['pressure_alpha']
        self._power_policy_alpha_threshold = cfg['threshold_alpha']
        self._power_policy_alpha_eco = cfg['eco_alpha']
        self._power_policy_eco_strength = cfg['eco_strength']
        self._charge_threshold_target = self._charge_threshold_base + self._charge_threshold_span * self._charger_pressure_raw
        self._charge_threshold = np.clip(self._charge_threshold, self._charge_threshold_base, self._charge_threshold_base + self._charge_threshold_span)
        self._power_policy_eco = np.clip(self._power_policy_eco, 1.0 - self._power_policy_eco_strength, 1.0)
        return True

    def set_flow_policy_mode(self, mode):
        """Override flow-control policy profile from UI.
        Target occupancy is capped by the robot-scaled baseline so a small
        fleet is never overwhelmed with more packages than it can handle."""
        mode = str(mode).strip().lower()
        _n_robots = max(len(self.robots), self.robotsMaxQuantity, 1)
        _max_cap  = max(self.packagesMaxQuantity, 1)
        _robot_ceiling = max(0.10, min(0.40, (_n_robots * 8) / _max_cap))
        profiles = {
            'steady':      {'target_occ': min(0.30, _robot_ceiling * 0.75), 'alpha': 0.03, 'pipeline': 3},
            'balanced':    {'target_occ': _robot_ceiling,                    'alpha': 0.02, 'pipeline': 4},
            'throughput':  {'target_occ': min(0.50, _robot_ceiling * 1.25), 'alpha': 0.015, 'pipeline': 6},
        }
        cfg = profiles.get(mode)
        if not cfg:
            return False
        self._flow_policy_mode = mode
        self._flow_target_occ = cfg['target_occ']
        self._flow_ema_alpha = cfg['alpha']
        self._flow_pipeline_depth = cfg['pipeline']
        return True

    def set_package_target_mode(self, mode):
        """Override package target placement mode from UI.

        Supported values: random, nearest, zone_edge.
        """
        mode = str(mode).strip().lower()
        if mode not in self._PKG_TARGET_MODES:
            return False
        self._pkg_target_mode = mode
        return True

    # ── Decommission system ──────────────────────────────────────────

    def _queue_idle_move(self, robot_index):
        """Queue a 'move to idle' task to the nearest ZONE_NONE cell.

        Called when a robot finishes all tasks and is sitting inside a zone.
        Moves the robot to a neutral corridor so it doesn't block zone slots.
        """
        r = self.robots[robot_index]
        rx, ry = r.xyLocation
        # Only move if currently inside a zone (import/storage/export)
        if self.zoneMap[ry][rx] == ZONE_NONE:
            return
        target = self._find_nearest_neutral_cell(rx, ry)
        if target is None:
            return
        self.robots[robot_index], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(
            robot_index, len(r.actionQueue), list(target), "move to idle")
        self._compute_robot_path(robot_index, list(target))

    def _queue_exit_task(self, robot_index):
        """Queue a 'move to exit' task toward the robot's birth location.

        Falls back to nearest perimeter cell if birth location is unavailable.
        """
        r = self.robots[robot_index]
        target = getattr(r, 'birthLocation', None)
        if target is None:
            # Fallback: nearest perimeter cell
            rx, ry = r.xyLocation
            best_dist = float('inf')
            best_cell = None
            for coord in self.warehousePerimeterCoordinates:
                cx, cy = coord[0], coord[1]
                d = abs(rx - cx) + abs(ry - cy)
                if d < best_dist:
                    best_dist = d
                    best_cell = [cx, cy]
            target = best_cell if best_cell is not None else [0, 0]
        else:
            target = list(target)
        self.robots[robot_index], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(
            robot_index, len(r.actionQueue), target, "move to exit")
        self._compute_robot_path(robot_index, target)
        _log.info('decommission: robot#%d queued exit to %s (birth)', r.robotNumber, target)

    def decommission_robot(self, robot_index):
        """Mark a robot for retirement. If idle, immediately queue exit task."""
        r = self.robots[robot_index]
        if r.decommissioning:
            return
        r.decommissioning = True
        _log.info('decommission: robot#%d flagged for retirement', r.robotNumber)
        if r.status == 'idle' and not r.actionQueue:
            self._queue_exit_task(robot_index)

    def _remove_robot(self, robot_index):
        """Remove a decommissioned robot that has reached the perimeter."""
        r = self.robots[robot_index]
        rnum = r.robotNumber
        assert r.carrying == -1, f'robot#{rnum} still carrying a package'
        # Remove from task assignment list
        self.robotsTaskAssignmentList = [
            x for x in self.robotsTaskAssignmentList if x[0] != rnum]
        # Remove from robots list
        self.robots.pop(robot_index)
        self.robotsInWarehouseCount -= 1
        _log.info('decommission: robot#%d removed from warehouse (fleet=%d)',
                   rnum, len(self.robots))
        # Rescale flow config for smaller fleet
        self._rescale_flow_config()

    def _rescale_flow_config(self):
        """Recompute robot-scaled flow parameters for current fleet size."""
        _n_robots = max(len(self.robots), self.robotsMaxQuantity, 1)
        _max_cap = max(self.packagesMaxQuantity, 1)
        self._flow_target_occ = max(0.10, min(0.40, (_n_robots * 8) / _max_cap))
        _ceiling = _n_robots * 10
        self._flow_import_cap = min(int(_n_robots * self._flow_pipeline_depth), _ceiling)
        self.packagesMaxMoveQuantity = max(6, min(_n_robots * 4, 120))

    def set_robot_count(self, target_count):
        """Adjust fleet size toward target_count.

        If target > current: raise robotsMaxQuantity (spawner handles the rest).
        If target < current: decommission excess robots, preferring idle then least-busy.
        _robot_target_count is set immediately so the UI reflects the target.
        """
        target_count = max(1, int(target_count))
        current = len(self.robots)
        self._robot_target_count = target_count
        self.robotsMaxQuantity = target_count
        if target_count > current:
            _log.info('set_robot_count: target=%d, raising robotsMaxQuantity to %d',
                       target_count, target_count)
        elif target_count < current:
            delta = current - target_count
            # Build candidate list: (index, busy_score) — idle robots first, then least-busy
            candidates = []
            for idx, r in enumerate(self.robots):
                if r.decommissioning:
                    continue
                busy = len(r.actionQueue) + (10 if r.carrying != -1 else 0)
                candidates.append((idx, busy))
            candidates.sort(key=lambda x: x[1])
            for idx, _busy in candidates[:delta]:
                self.decommission_robot(idx)
            _log.info('set_robot_count: target=%d, decommissioning %d robots',
                       target_count, min(delta, len(candidates)))

    def update_warehouse(self):
        """Main simulation tick.  Called once per sim frame (~40 Hz).

        Orchestrates the full update sequence: zone counts, flow control,
        zone-change reconciliation, package lifecycle (spawn/sort/assign/
        colour/export), charger spawning, robot lifecycle (spawn/assign/
        pathfind/move/battery/idle-park/decommission), occupancy arrays,
        and periodic summary logging.
        """
        #self.printDebugInfo()
        self.datetimeNow = datetime.now() # Update datetime
        self.timeElapsed = time.time() - self.timeStart
        self.update_zone_counts() # Recompute slot counts from zoneMap (supports dynamic zones)
        self.update_flow_control() # Adaptive import throttle based on congestion
        self.reconcile_zone_changes() # Process user-painted cells; partial-flush affected plans
        self.invalidate_stale_targets() # Cancel in-flight moves whose targets no longer match zoneMap
        self.recount_planned() # Authoritative recount of planned counts
        self.packages, self.packagesRollingCount, self.packagesInWarehouseCount, self.packagesLog, self.packageTargetsInWarehouse, self.packageExportCount, self.packageExportRollingCount = self.update_packages() # Update packages
        self.chargers, self.chargersRollingCount, self.chargersInWarehouse = self.update_chargers()
        self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsRollingCount, self.robotsTaskAssignmentList, self.robotsLog, self.chargers = self.update_robots() # Update robots
        self.packages = self.update_carried_packages()
        self.packages, self.packagesInImportCount, self.packagesInStorageCount, self.packagesInExportCount = self.update_packages_areas()
        self.packagesInWarehouse = self.update_packages_in_warehouse() # Update warehouse knowledge of packages
        self.chargersInWarehouse = self.update_chargers_in_warehouse()
        self.robotsInWarehouse = self.update_robots_in_warehouse()
        self.importSpaceAvailable, self.storageSpaceAvailable, self.exportSpaceAvailable = self.update_space_available()
        self.packagesLog, self.robotsLog = self.trim_logs()
        self.record_warehouse_data()
        #self.printDebugInfo()
        # ── Periodic summary log (~every 5 sec at 40 tps) ──
        if self.warehouseLoopCount % 200 == 0:
            _n_overdue = sum(1 for p in self.packages if p.timeToDeadline.total_seconds() < 0)
            _n_carried = sum(1 for p in self.packages if p.status == 'carried')
            _n_planned = sum(1 for p in self.packages if p.status == 'move planned')
            _n_idle_robots = sum(1 for r in self.robots if r.status == 'idle')
            _n_charging = sum(1 for r in self.robots if r.status == 'charging')
            _avg_drain = (sum(r.batteryDepletingRate * r.batteryDrainMultiplier for r in self.robots)
                          / max(len(self.robots), 1))
            _log.info(
                'tick=%d  pkgs=%d (imp=%d sto=%d exp=%d)  moveList=%d  movingList=%d  '
                'overdue=%d  carried=%d  planned=%d  exported=%d  '
                'robots: idle=%d charging=%d  avg_drain=%.3f  pressure=%.2f raw=%.2f  thresh=%.0f/%.0f%%  '
                'flow: cap=%d/%d  delivery=%.0fs  throughput=%.2f/s',
                self.warehouseLoopCount,
                len(self.packages), self.packagesInImportCount,
                self.packagesInStorageCount, self.packagesInExportCount,
                len(self.packagesMoveList), len(self.packagesMovingList),
                _n_overdue, _n_carried, _n_planned,
                self.packageExportRollingCount,
                _n_idle_robots, _n_charging, _avg_drain,
                self._charger_pressure, self._charger_pressure_raw,
                self._charge_threshold, self._charge_threshold_target,
                self._flow_import_cap, self.packagesMaxQuantity,
                self._flow_avg_delivery, self._flow_throughput)
        self.warehouseLoopCount += 1
        return self

    def update_zone_counts(self):
        """Recompute slot counts from zoneMap. Supports dynamic zone changes."""
        self.numberOfImportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_IMPORT))
        self.numberOfStorageSlots = int(np.count_nonzero(self.zoneMap == ZONE_STORAGE))
        self.numberOfExportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_EXPORT))
        self.packagesMaxQuantity  = self.numberOfImportSlots + self.numberOfStorageSlots + self.numberOfExportSlots
        self.laneMap = None

    def update_flow_control(self):
        """Adaptive import cap based on fleet utilisation.

        Robot-scaled: base target = pipeline_depth * n_robots.
        Adjustments:
          - Idle bonus: boost imports when robots have spare capacity
          - Overdue penalty: back off only when robots are busy AND overdue
          - Delivery-stress penalty: ease off when deliveries lag deadlines
          - Emergency floor: if majority idle, immediately raise cap
        """
        n_pkgs   = len(self.packages)
        n_robots = len(self.robots)
        max_cap  = self.packagesMaxQuantity
        if n_robots == 0 or max_cap == 0:
            self._flow_import_cap = max_cap
            return

        _robot_ceiling = n_robots * 10

        # ── Signals ──────────────────────────────────────────
        n_overdue = sum(1 for p in self.packages if p.timeToDeadline.total_seconds() < 0)
        overdue_ratio = n_overdue / max(n_pkgs, 1)

        n_idle_robots = sum(1 for r in self.robots if r.status == 'idle')
        idle_robot_ratio = n_idle_robots / n_robots

        if self._flow_avg_deadline > 0 and self._flow_avg_delivery > 0:
            delivery_stress = self._flow_avg_delivery / self._flow_avg_deadline
        else:
            delivery_stress = 0.0

        # ── Robot-scaled base target ─────────────────────────
        # Each robot needs a pipeline: waiting → assigned → carried → delivered.
        base_target = n_robots * self._flow_pipeline_depth

        # ── Adjustments ──────────────────────────────────────
        # Idle bonus: idle robots = wasted capacity = pull more work.
        # No overdue gate — if robots are idle they should get work regardless.
        idle_bonus = 0.0
        if idle_robot_ratio > 0.15:
            idle_bonus = min((idle_robot_ratio - 0.15) * 0.8, 0.50)

        # Overdue penalty: only when robots are busy (idle + overdue = routing
        # issue, not volume issue — don't starve the fleet further).
        overdue_penalty = 0.0
        if overdue_ratio > 0.10 and idle_robot_ratio < 0.20:
            overdue_penalty = min((overdue_ratio - 0.10) * 2.0, 0.40)

        # Delivery-stress penalty: ease off when deliveries are slow vs deadlines
        stress_penalty = 0.0
        if delivery_stress > 0.7:
            stress_penalty = min((delivery_stress - 0.7) * 0.4, 0.20)

        # ── Combine ──────────────────────────────────────────
        adj = 1.0 + idle_bonus - overdue_penalty - stress_penalty
        adj = max(0.30, min(1.50, adj))

        target_cap = int(base_target * adj)
        target_cap = max(n_robots, min(_robot_ceiling, target_cap))

        # Smooth: 15 % toward target each tick — responsive but stable
        self._flow_import_cap = self._flow_import_cap + 0.15 * (target_cap - self._flow_import_cap)
        self._flow_import_cap = int(max(n_robots, min(_robot_ceiling, self._flow_import_cap)))

        # Emergency floor: if majority of robots idle, jump to minimum viable
        if idle_robot_ratio >= 0.5 and self._flow_import_cap < n_robots * 3:
            self._flow_import_cap = n_robots * 3

        # Track average deadline window from recently spawned packages
        recent = [p for p in self.packages if (time.time() - p.createdAt) < 2.0]
        if recent:
            avg_dl = sum((p.deadline - datetime.fromtimestamp(p.createdAt)).total_seconds() for p in recent) / len(recent)
            a = self._flow_ema_alpha
            self._flow_avg_deadline = a * avg_dl + (1 - a) * self._flow_avg_deadline if self._flow_avg_deadline > 0 else avg_dl

    def invalidate_stale_targets(self):
        """Cancel in-flight moves whose target cells no longer match the intended zone."""
        expected_zone = {'import': ZONE_IMPORT, 'storage': ZONE_STORAGE, 'export': ZONE_EXPORT}
        DROPOFF_ACTIONS = {"move to target dropoff location", "drop off target package"}
        for p in self.packages:
            if p.status in ('move planned', 'carried') and p.areaTarget != 'none':
                tx, ty = p.xyLocationTarget
                actual = self.zoneMap[ty][tx]
                if actual != expected_zone.get(p.areaTarget, -1):
                    self.packageTargetsInWarehouse[ty][tx] = 0
                    p.areaTarget = 'none'
                    p.xyLocationTarget = p.xyLocation.copy()
                    if p.status == 'move planned':
                        p.status = 'idle'
                    elif p.status == 'carried':
                        # Abort carry — mirrors reconcile_zone_changes Case C
                        p.status = 'idle'
                        if p.packageNumber in self.packagesMovingList:
                            self.packagesMovingList.remove(p.packageNumber)
                        if p.carrier not in (None, -1):
                            ri = next((j for j, r in enumerate(self.robots)
                                       if r.robotNumber == p.carrier), None)
                            if ri is not None:
                                # Strip dropoff tasks from robot queue
                                to_remove = [k for k, t in enumerate(self.robots[ri].actionQueue)
                                             if t[1] in DROPOFF_ACTIONS]
                                for k in reversed(to_remove):
                                    self.robots[ri].actionQueue.pop(k)
                                    tai = next((j for j, x in enumerate(self.robotsTaskAssignmentList)
                                                if x[0] == self.robots[ri].robotNumber), None)
                                    if tai is not None:
                                        self.robotsTaskAssignmentList[tai][1] -= 1
                                self.robots[ri].carrying = -1
                                if not self.robots[ri].actionQueue:
                                    self.robots[ri].status = 'idle'
                                else:
                                    self.robots[ri].status = self.robots[ri].actionQueue[0][1]
                            p.carrier = -1
                        _log.info('invalidate_stale: force-dropped carried pkg#%d (target zone changed)',
                                  p.packageNumber)
                    if p.packageNumber in self.packagesMoveList:
                        self.packagesMoveList.remove(p.packageNumber)

    def reconcile_zone_changes(self):
        """Process cells painted by the user this tick.

        Surgically retires only plans that touch changed cells, frees the
        corresponding robot queue entries, and queues displaced packages onto
        packagesReorgList for priority re-scheduling this tick.
        No-ops immediately when no cells were painted.
        """
        if not self.pending_zone_changes:
            return

        changed_cells = self.pending_zone_changes
        self.pending_zone_changes = set()

        ZONE_NAMES      = {ZONE_NONE: 'neutral', ZONE_IMPORT: 'import',
                           ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}
        PICKUP_ACTIONS  = {"move to target pickup location", "pick up target package"}
        DROPOFF_ACTIONS = {"move to target dropoff location", "drop off target package"}

        def _strip_robot_tasks(robot_number, action_set, location=None):
            """Remove matching tasks from a robot's action queue and adjust counts."""
            ri = next((j for j, r in enumerate(self.robots) if r.robotNumber == robot_number), None)
            if ri is None:
                return
            r = self.robots[ri]
            if location is not None:
                new_q = [t for t in r.actionQueue
                         if not (t[1] in action_set and t[0] == location)]
            else:
                new_q = [t for t in r.actionQueue if t[1] not in action_set]
            removed = len(r.actionQueue) - len(new_q)
            if removed:
                r.actionQueue = new_q
                ta_idx = next((j for j, x in enumerate(self.robotsTaskAssignmentList)
                               if x[0] == robot_number), None)
                if ta_idx is not None:
                    self.robotsTaskAssignmentList[ta_idx][1] = max(
                        0, self.robotsTaskAssignmentList[ta_idx][1] - removed)
                r.status = r.actionQueue[0][1] if r.actionQueue else 'idle'

        def _robots_with_pickup_at(location):
            return [r.robotNumber for r in self.robots
                    if any(t[0] == location and t[1] in PICKUP_ACTIONS
                           for t in r.actionQueue)]

        for p in self.packages:
            px, py  = p.xyLocation
            tx, ty  = p.xyLocationTarget
            cur_chg = (px, py) in changed_cells
            tgt_chg = (tx, ty) in changed_cells

            if cur_chg and p.status == 'idle':
                # Case A: idle package on repainted cell — reclassify area only
                p.area = ZONE_NAMES.get(self.zoneMap[py][px], 'neutral')
                if p.packageNumber not in self.packagesReorgList:
                    self.packagesReorgList.append(p.packageNumber)

            elif cur_chg and p.status == 'move planned':
                # Case B: planned package on repainted cell — cancel plan + free inbound robot
                self.packageTargetsInWarehouse[ty][tx] = 0
                p.area = ZONE_NAMES.get(self.zoneMap[py][px], 'neutral')
                p.areaTarget = 'none'
                p.xyLocationTarget = p.xyLocation.copy()
                p.status = 'idle'
                if p.packageNumber in self.packagesMoveList:
                    self.packagesMoveList.remove(p.packageNumber)
                for rn in _robots_with_pickup_at([px, py]):
                    _strip_robot_tasks(rn, PICKUP_ACTIONS, [px, py])
                if p.packageNumber not in self.packagesReorgList:
                    self.packagesReorgList.append(p.packageNumber)

            elif tgt_chg and p.status == 'carried':
                # Case C: robot carrying package to now-invalid target — abort carry
                self.packageTargetsInWarehouse[ty][tx] = 0
                p.areaTarget = 'none'
                p.xyLocationTarget = p.xyLocation.copy()
                p.status = 'idle'
                if p.packageNumber in self.packagesMoveList:
                    self.packagesMoveList.remove(p.packageNumber)
                if p.packageNumber in self.packagesMovingList:
                    self.packagesMovingList.remove(p.packageNumber)
                if p.carrier not in (None, -1):
                    _strip_robot_tasks(p.carrier, DROPOFF_ACTIONS)
                    ri = next((j for j, r in enumerate(self.robots)
                               if r.robotNumber == p.carrier), None)
                    if ri is not None:
                        self.robots[ri].carrying = -1
                p.carrier = -1
                if p.packageNumber not in self.packagesReorgList:
                    self.packagesReorgList.append(p.packageNumber)

            elif tgt_chg and p.status == 'move planned' and not cur_chg:
                # Case D: assigned target now wrong zone — cancel plan + free inbound robot
                self.packageTargetsInWarehouse[ty][tx] = 0
                p.areaTarget = 'none'
                p.xyLocationTarget = p.xyLocation.copy()
                p.status = 'idle'
                if p.packageNumber in self.packagesMoveList:
                    self.packagesMoveList.remove(p.packageNumber)
                for rn in _robots_with_pickup_at([px, py]):
                    _strip_robot_tasks(rn, PICKUP_ACTIONS, [px, py])
                if p.packageNumber not in self.packagesReorgList:
                    self.packagesReorgList.append(p.packageNumber)

        self.recount_planned()
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False)

    def recount_planned(self):
        """Authoritative recount of planned counts from package list."""
        self.packagesPlannedInImportCount  = sum(1 for p in self.packages if p.areaTarget == 'import')
        self.packagesPlannedInStorageCount = sum(1 for p in self.packages if p.areaTarget == 'storage')
        self.packagesPlannedInExportCount  = sum(1 for p in self.packages if p.areaTarget == 'export')

    def update_space_available(self):
        # Space is available only if both planned and actual counts are below the limit
        self.importSpaceAvailable = (self.packagesPlannedInImportCount < self.numberOfImportSlots) and (self.packagesInImportCount < self.numberOfImportSlots)
        self.storageSpaceAvailable = (self.packagesPlannedInStorageCount < self.numberOfStorageSlots) and (self.packagesInStorageCount < self.numberOfStorageSlots)
        self.exportSpaceAvailable = (self.packagesPlannedInExportCount < self.numberOfExportSlots) and (self.packagesInExportCount < self.numberOfExportSlots)
        return self.importSpaceAvailable, self.storageSpaceAvailable, self.exportSpaceAvailable
    
    def trim_logs(self):
        if len(self.packagesLog) > self.logsMaxLength:
            self.packagesLog = self.packagesLog[len(self.packagesLog)-self.logsMaxLength::]
        if len(self.robotsLog) > self.logsMaxLength:
            self.robotsLog = self.robotsLog[len(self.robotsLog)-self.logsMaxLength::]
        return self.packagesLog, self.robotsLog

    def record_warehouse_data(self):
        # Time
        #self.warehouse_data.datetimeStamps = Timeseries_Functions.rollUpdate(self.warehouse_data.datetimeStamps, 1, self.datetimeNow)
        self.warehouse_data.timeElapseds = Timeseries_Functions.rollUpdate(self.warehouse_data.timeElapseds, 1, self.timeElapsed)
        # numPackages
        self.warehouse_data.numPackagesImported = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesImported, 1, self.packagesRollingCount)
        self.warehouse_data.numPackagesInWarehouse = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInWarehouse, 1, self.packagesInWarehouseCount)
        self.warehouse_data.numPackagesInImport = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInImport, 1, self.packagesInImportCount)
        self.warehouse_data.numPackagesInStorage = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInStorage, 1, self.packagesInStorageCount)
        self.warehouse_data.numPackagesInExport = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInExport, 1, self.packagesInExportCount)
        self.warehouse_data.numPackagesInMoveList = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInMoveList, 1, len(self.packagesMoveList))
        self.warehouse_data.numPackagesInMovingList = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesInMovingList, 1, len(self.packagesMovingList))
        self.warehouse_data.numPackagesExported = Timeseries_Functions.rollUpdate(self.warehouse_data.numPackagesExported, 1, self.packageExportRollingCount)
        # numChargers
        # numRobots
        self.warehouse_data.numRobotsInWarehouse = Timeseries_Functions.rollUpdate(self.warehouse_data.numRobotsInWarehouse, 1, self.robotsInWarehouseCount)
        return self

    def printDebugInfo(self):
        debug_len = 1
        if self.warehouseLoopCount % 1 == 0:
            print("--- debug ---")
            # Warehouse
            print("- warehouse:")
            packagesInWarehouseLen = np.count_nonzero(self.packagesInWarehouse)
            print("- pInWarehouseCount: {}, pInWarehouse: {}".format(self.packagesInWarehouseCount, packagesInWarehouseLen))
            print("- pInImport: {}, pInStorage: {}, pInExport: {}".format(self.packagesInImportCount, self.packagesInStorageCount, self.packagesInExportCount))
            print("- pPlannedInImport: {}. pPlannedInStorage: {}, pPlannedInExport: {}".format(self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount))
            print("- spacesInWarehouse: {}, {}, {}".format(self.importSpaceAvailable, self.storageSpaceAvailable, self.exportSpaceAvailable))
            # Packages
            len_packages = len(self.packages)
            print("- packages: {}, packagesRollingCount: {}".format(len_packages, self.packagesRollingCount))
            print("- packageExportCount: {}, packageExportRollingCount: {}".format(self.packageExportCount, self.packageExportRollingCount))
            #if len_packages > debug_len:
            #    len_packages = debug_len
            # Packages In Import
            print("- Packages In Import:")
            if (self.packagesMoveList) and (not self.packagesMovingList):
                orig_debug_len = debug_len
                debug_len = 1000
            i_count = 0
            if len_packages > 0:
                for i in range(0,len_packages):
                    if self.packages[i].area == "import":
                        if i_count < debug_len:
                            print("- package#: {}, tToDeadline: {}, area: {}, areaTgt: {}, status: {}".format(self.packages[i].packageNumber, self.packages[i].timeToDeadline, self.packages[i].area, self.packages[i].areaTarget, self.packages[i].status))
                            i_count += 1
            if (self.packagesMoveList) and (not self.packagesMovingList):
                debug_len = orig_debug_len
            # Packages In Storage
            print("- Packages In Storage:")
            i_count = 0
            if len_packages > 0:
                for i in range(0,len_packages):
                    if self.packages[i].area == "storage":
                        if i_count < debug_len:
                            print("- package#: {}, tToDeadline: {}, area: {}, areaTgt: {}, status: {}".format(self.packages[i].packageNumber, self.packages[i].timeToDeadline, self.packages[i].area, self.packages[i].areaTarget, self.packages[i].status))
                            i_count += 1
            # Packages In Export
            print("- Packages In Export:")
            i_count = 0
            if len_packages > 0:
                for i in range(0,len_packages):
                    if self.packages[i].area == "export":
                        if i_count < debug_len:
                            print("- package#: {}, tToDeadline: {}, area: {}, areaTgt: {}, status: {}".format(self.packages[i].packageNumber, self.packages[i].timeToDeadline, self.packages[i].area, self.packages[i].areaTarget, self.packages[i].status))
                            i_count += 1
                        # Carried Packages
            #packagesCarriedIndices = [y for y, x in enumerate(self.packages) if x.status == "carried"]
            #if packagesCarriedIndices:
            #    for i in packagesCarriedIndices:
            #        print("- packageCarried#: {}, xyLocation: {}, xyLocationTarget: {}, status: {}".format(self.packages[i].packageNumber, self.packages[i].xyLocation, self.packages[i].xyLocationTarget, self.packages[i].status))
            # packagesMoveList
            len_packagesMoveList = len(self.packagesMoveList)
            print("- packagesMoveList: {}".format(len_packagesMoveList))
            #print(self.packagesMoveList)
            # packagesMovingList
            len_packagesMovingList = len(self.packagesMovingList)
            print("- packagesMovingList: {}".format(len_packagesMovingList))
            #print(self.packagesMovingList)
            # Packages In MoveList but not in MovingList
            #len_packages = len(self.packages)
            #print("- Packages In MoveList but not in MovingList:")
            #i_count = 0
            #if len_packages > 0:
            #    for packageNumber in self.packagesMoveList:
            #        if packageNumber not in self.packagesMovingList:
            #            packageIndex = [y for y, x in enumerate(self.packages) if x.packageNumber == packageNumber]
            #            if packageIndex:
            #                packageIndex = packageIndex[0]
            #                #if i_count < debug_len:
            #                print("- package#: {}, tToDeadline: {}, area: {}, areaTgt: {}, status: {}".format(self.packages[packageIndex].packageNumber, self.packages[packageIndex].timeToDeadline, self.packages[packageIndex].area, self.packages[packageIndex].areaTarget, self.packages[packageIndex].status))
            #                i_count += 1
            #print("- # packages in MoveList but not in MovingList: {}".format(i_count))
            # Packages Log
            len_packagesLog = len(self.packagesLog)
            print("- packagesLog: {}".format(len_packagesLog))
            #if len_packagesLog > debug_len:
            #    for i in range(len_packagesLog-debug_len,len_packagesLog):
            #        print("- action: {}, datetimeNow: {}".format(self.packagesLog[i].action, self.packagesLog[i].datetimeNow))
            # Chargers
            #len_chargers = len(self.chargers)
            #print("- chargers: {}, rc: {}".format(len_chargers, self.chargersRollingCount))
            #if len_chargers > debug_len:
            #    len_chargers = debug_len
            #if len_chargers > 0:
            #    for i in range(0,len_chargers):
            #        print("- charger#: {}, status: {}, xyLocation: {}".format(self.chargers[i].chargerNumber, self.chargers[i].status, self.chargers[i].xyLocation))
            # Robots
            len_robots = len(self.robots)
            print("- robots: {}, rc: {}".format(len_robots, self.robotsRollingCount))
            #if len_robots > debug_len:
            #    len_robots = debug_len
            #if len_robots > 0:
            #    for i in range(0,len_robots):
            #        #print("- robot#: {}, batt%: {}, status = {}, lenActionQ: {}, xyLoc: {}, tgtxyLoc: {}".format(self.robots[i].robotNumber, self.robots[i].batteryPercent, self.robots[i].status, len(self.robots[i].actionQueue), self.robots[i].xyLocation, self.robots[i].xyLocationTarget))
            #        print("- robot#: {}, actionQueue: {}".format(self.robots[i].robotNumber, self.robots[i].actionQueue))
            # Robots Task Assignment List
            #print("- robotsTaskAssignmentList: {}".format(len(self.robotsTaskAssignmentList)))
            #print(self.robotsTaskAssignmentList)
            # Robots Log
            len_robotsLog = len(self.robotsLog)
            print("- robotsLog: {}".format(len_robotsLog))
            #if len_robotsLog > debug_len:
            #    for i in range(len_robotsLog-debug_len,len_robotsLog):
            #        print("- action: {}, datetimeNow: {}".format(self.robotsLog[i].action, self.robotsLog[i].datetimeNow))
            
            # CONDITIONAL CONTINUE
            #if self.packagesPlannedInStorageCount:
            #    time.sleep(500)
            #    exit()
        return

    # Update Packages
    def update_packages(self):
        #print("- importSpaceAvailable: {}".format(self.importSpaceAvailable))
        # Determine spawn zone: prefer import, fall back to storage, then export
        spawnZoneId = None
        if self.importSpaceAvailable and self.numberOfImportSlots > 0:
            spawnZoneId = ZONE_IMPORT
        elif self.numberOfImportSlots == 0 and self.storageSpaceAvailable and self.numberOfStorageSlots > 0:
            spawnZoneId = ZONE_STORAGE
        elif self.numberOfImportSlots == 0 and self.numberOfStorageSlots == 0 and self.exportSpaceAvailable and self.numberOfExportSlots > 0:
            spawnZoneId = ZONE_EXPORT
        if spawnZoneId is not None:
            # Use adaptive import cap instead of raw packagesMaxQuantity
            _effective_cap = min(self._flow_import_cap, self.packagesMaxQuantity)
            self.packagesRollingCount, self.packagesInImportCount, self.packagesInWarehouseCount, self.packages, self.packagesLog = Package_Functions.import_package(Package_Functions, self.zoneMap, self.chargersInWarehouse, self.packagesRollingCount, self.packagesInImportCount, self.packagesInWarehouseCount, self.packagesInWarehouse, self.packageTargetsInWarehouse, _effective_cap, self.packages, self.packagesLog, self.itemsList, self.addressesList, self.datetimeNow, spawnZoneId=spawnZoneId) # Import package
        self.packages = self.update_packages_timeToDeadline(self.datetimeNow) # Update package timeToDeadline
        self.packages.sort(key=lambda x: x.deadline, reverse=False) # Sort packages by deadline
        self._reconcile_stale_moving_list()  # Clean orphaned packagesMovingList entries
        self.packagesMoveList = self.sort_packagesMoveList_by_deadline()
        self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount = self.remove_immovables_from_packagesMoveList()
        self.packagesMoveList, self.packages = self.append_to_packagesMoveList() # Add to packagesMoveList based on deadline
        # Re-sort after append: packages appended above land at the end of the list,
        # but may be more urgent than existing entries (e.g. newly overdue).
        # Robot assignment iterates packagesMoveList front-to-back, so position == priority.
        self.packagesMoveList = self.sort_packagesMoveList_by_deadline()
        self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount = self.update_packages_targetLocation() # Decide to Move Package to Storage or Export
        self.packageTargetsInWarehouse = self.update_packageTargetsInWarehouse()
        self.packages = self.update_packages_colours(self.datetimeNow)
        self.packages, self.packageExportCount, self.packageExportRollingCount = self.package_export()

        return self.packages, self.packagesRollingCount, self.packagesInWarehouseCount, self.packagesLog, self.packageTargetsInWarehouse, self.packageExportCount, self.packageExportRollingCount

    def update_packages_timeToDeadline(self, datetimeNow):
        for i in range(len(self.packages)):
            self.packages[i].timeToDeadline = self.packages[i].deadline - datetimeNow
        return self.packages

    def sort_packagesMoveList_by_deadline(self):
        if not self.packagesMoveList:
            return self.packagesMoveList
        # Sort packagesMoveList by deadline
        packageInMoveListCount = 0
        for i in range(len(self.packages)):
            packageNumber = self.packages[i].packageNumber
            if (packageNumber in self.packagesMoveList):
                self.packagesMoveList.remove(packageNumber)
                self.packagesMoveList.insert(packageInMoveListCount, packageNumber)
                packageInMoveListCount += 1
        return self.packagesMoveList

    def _find_nearest_empty_cell(self, cx, cy):
        """BFS from (cx, cy) to find nearest cell with no package, charger, or robot.

        Returns [nx, ny] or [cx, cy] if nothing found (fallback to same cell).
        """
        W, H = self.warehouseWindowRes
        visited = set()
        q = deque()
        q.append((cx, cy))
        visited.add((cx, cy))
        while q:
            x, y = q.popleft()
            if (self.packagesInWarehouse[y][x] == 0
                    and self.chargersInWarehouse[y][x] == 0
                    and self.robotsInWarehouse[y][x] == 0):
                return [x, y]
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return [cx, cy]  # fallback

    def _find_nearest_neutral_cell(self, cx, cy):
        """BFS from (cx, cy) to find nearest ZONE_NONE cell with no package, charger, or robot.

        Used to park idle robots in neutral corridors after charging.
        Returns (nx, ny) or None if nothing found.
        """
        W, H = self.warehouseWindowRes
        visited = set()
        q = deque()
        q.append((cx, cy))
        visited.add((cx, cy))
        while q:
            x, y = q.popleft()
            if (self.zoneMap[y][x] == ZONE_NONE
                    and self.packagesInWarehouse[y][x] == 0
                    and self.chargersInWarehouse[y][x] == 0
                    and self.robotsInWarehouse[y][x] == 0):
                return (x, y)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return None

    def _reconcile_stale_moving_list(self):
        """Remove orphaned entries from packagesMovingList.

        A package can become stranded in packagesMovingList when a robot's
        pickup task is cancelled (e.g. low-battery preemption inserts a
        charging task and the pickup task is dropped, or reconcile_zone_changes
        strips tasks).  The package reverts to idle/carrier=-1 but is never
        removed from packagesMovingList, permanently blocking it from being
        re-queued.

        Ground-truth: a package belongs in packagesMovingList only if a robot
        currently has a pickup or dropoff task targeting it, OR the package
        status is 'carried'.
        """
        if not self.packagesMovingList:
            return
        # Build set of package locations that robots are actively working on
        _active_pickup_locs = set()
        for r in self.robots:
            for task in r.actionQueue:
                if task[1] in ("move to target pickup location", "pick up target package"):
                    _active_pickup_locs.add(tuple(task[0]))
        # Also include packages currently being carried
        _carried_pkgs = {p.packageNumber for p in self.packages if p.status == 'carried'}
        # Detect zombie carries: packages stuck on robots at 0% battery
        _zombie_carriers = {r.carrying for r in self.robots
                           if r.carrying != -1 and r.batteryPercent <= 0}
        for _zpkg in _zombie_carriers:
            _zidx = next((j for j, p in enumerate(self.packages)
                          if p.packageNumber == _zpkg), None)
            if _zidx is not None and self.packages[_zidx].status == 'carried':
                _zrobot_idx = next((j for j, r in enumerate(self.robots)
                                    if r.carrying == _zpkg), None)
                if _zrobot_idx is not None:
                    _zloc = self._find_nearest_empty_cell(
                        self.packages[_zidx].xyLocation[0], self.packages[_zidx].xyLocation[1])
                    self.packages[_zidx].xyLocation = _zloc
                    _log.warning('zombie carry: robot#%d at 0%% stuck with pkg#%d -- force-dropping at %s',
                                 self.robots[_zrobot_idx].robotNumber, _zpkg, _zloc)
                    self.packages[_zidx].status = 'idle'
                    self.packages[_zidx].carrier = -1
                    self.packages[_zidx].areaTarget = 'none'
                    self.packages[_zidx].xyLocationTarget = self.packages[_zidx].xyLocation.copy()
                    self.packages[_zidx].deliveredAt = time.time()
                    self.robots[_zrobot_idx].carrying = -1
                    if _zpkg in self.packagesMovingList:
                        self.packagesMovingList.remove(_zpkg)
                    if _zpkg in self.packagesMoveList:
                        self.packagesMoveList.remove(_zpkg)

        for pkg_num in list(self.packagesMovingList):
            if pkg_num in _carried_pkgs:
                continue
            pkg_idx = next((j for j, p in enumerate(self.packages) if p.packageNumber == pkg_num), None)
            if pkg_idx is None:
                # Package no longer exists
                self.packagesMovingList.remove(pkg_num)
                continue
            p = self.packages[pkg_idx]
            if p.status in ('carried',):
                continue
            # Check if any robot is heading to pick up this package
            pkg_loc = tuple(p.xyLocation)
            if pkg_loc not in _active_pickup_locs:
                # Orphaned: no robot is heading for this package
                _log.warning('stale movingList: pkg#%d status=%s area=%s loc=%s -- removed',
                             pkg_num, p.status, p.area, p.xyLocation)
                self.packagesMovingList.remove(pkg_num)
                # Also clean up packagesMoveList if present with stale state
                if pkg_num in self.packagesMoveList and p.status == 'idle' and p.areaTarget == 'none':
                    self.packagesMoveList.remove(pkg_num)

    def remove_immovables_from_packagesMoveList(self):
        if not self.packagesMoveList:
            return self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount
        # For idle packages in the move list that are plan-surplus for their target zone,
        # cancel only as many as are actually over-planned this tick.
        #
        # Surplus = planned_count - free_physical_slots
        #         = planned_count - (total_slots - actual_in_zone)
        #
        # A positive surplus means we have more pending plans than there are empty
        # slots to absorb them: cancel exactly that many (least-important first,
        # i.e. reversed move-list order).  Zero or negative surplus = nothing to do,
        # even if the space-available flag is False — physical slots still exist.
        # This prevents mass plan-cancellations triggered purely by the boolean gate.
        free_import  = max(0, self.numberOfImportSlots  - self.packagesInImportCount)
        free_storage = max(0, self.numberOfStorageSlots - self.packagesInStorageCount)
        free_export  = max(0, self.numberOfExportSlots  - self.packagesInExportCount)
        surplus = {
            'import':  max(0, self.packagesPlannedInImportCount  - free_import),
            'storage': max(0, self.packagesPlannedInStorageCount - free_storage),
            'export':  max(0, self.packagesPlannedInExportCount  - free_export),
        }
        evicted = {'import': 0, 'storage': 0, 'export': 0}
        _now_stale = time.time()
        _STALE_PLAN_TIMEOUT = 5.0  # seconds before an unassigned 'move planned' is reset
        for packageNumber in list(reversed(self.packagesMoveList)):  # snapshot — list is mutated inside loop
            if (packageNumber not in self.packagesMovingList):
                packageIndex = [y for y, x in enumerate(self.packages) if x.packageNumber == packageNumber]
                if packageIndex:
                    packageIndex = packageIndex[0]
                    packageStatus    = self.packages[packageIndex].status
                    packageAreaTarget = self.packages[packageIndex].areaTarget
                    removePackage = False
                    if (packageStatus == "idle"):
                        if (packageAreaTarget == "none"):
                            removePackage = True
                        elif packageAreaTarget in surplus and evicted[packageAreaTarget] < surplus[packageAreaTarget]:
                            removePackage = True
                    elif (packageStatus == "move planned"):
                        # Evict plans that have been waiting for a robot too long.
                        _planned_at = getattr(self.packages[packageIndex], 'plannedAt', None)
                        if _planned_at is not None and (_now_stale - _planned_at) > _STALE_PLAN_TIMEOUT:
                            removePackage = True
                            _log.debug('stale plan evict: pkg#%d areaTarget=%s age=%.1fs',
                                       packageNumber, packageAreaTarget, _now_stale - _planned_at)
                        # Remove Package
                        if (removePackage == True):
                            if (packageAreaTarget == "import"):
                                self.packagesPlannedInImportCount -= 1
                            elif (packageAreaTarget == "storage"):
                                self.packagesPlannedInStorageCount -= 1
                            elif (packageAreaTarget == "export"):
                                self.packagesPlannedInExportCount -= 1
                            # Only charge the surplus eviction budget for surplus-based evictions
                            # (idle packages). Stale-plan evictions are unconditional and must
                            # not reduce the budget, which would block legitimate surplus evictions.
                            if packageStatus == "idle" and packageAreaTarget in evicted:
                                evicted[packageAreaTarget] += 1
                            # Clear the phantom target cell so try_packageTargetLocation
                            # can reuse it this same tick.
                            tx, ty = self.packages[packageIndex].xyLocationTarget
                            if self.packageTargetsInWarehouse[ty][tx] == 1:
                                self.packageTargetsInWarehouse[ty][tx] = 0
                            self.packages[packageIndex].status = "idle"
                            self.packages[packageIndex].areaTarget = "none"
                            self.packagesMoveList.remove(packageNumber)
        return self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def append_to_packagesMoveList(self):
        """Queue idle packages for robot pickup.

        Pipeline depth is capped at ``available_robots * 3`` (floor 6) so the
        move queue stays proportional to fleet capacity.  Within that budget:

        1. Reorg list  — displaced packages from zone changes get front-of-queue.
        2. Overdue pass — past-deadline idle packages inserted at front.
        3. Normal pass  — remaining idle packages appended, limited by
           destination-slot headroom (export + storage free slots).

        Packages must pass import cooldown (2 s) and delivery cooldown (2 s)
        before being eligible.  Returns (packagesMoveList, packages).
        """
        # Dynamic planning depth: only plan as many moves as robots can realistically
        # execute soon.  available_robots * 2 gives a one-task pipeline buffer.
        available_robots = sum(1 for r in self.robots
                               if r.status not in ('charging', 'move to charging station')
                               and r.batteryPercent >= self._work_min_batt_pct)
        effective_move_cap = max(available_robots * 3, 6)  # floor of 6 to keep pipeline fed
        effective_move_cap = min(effective_move_cap, self.packagesMaxMoveQuantity)

        # Headroom = genuinely available destination slots this tick.
        # Cap new moveList entries to total_headroom so we don't queue more
        # packages than there are slots to assign targets to this tick.
        export_headroom  = max(0, self.numberOfExportSlots  - self.packagesInExportCount  - self.packagesPlannedInExportCount)
        storage_headroom = max(0, self.numberOfStorageSlots - self.packagesInStorageCount - self.packagesPlannedInStorageCount)
        total_headroom   = export_headroom + storage_headroom
        new_entries      = 0

        _now = time.time()
        _IMPORT_COOLDOWN = 2.0  # seconds a newly-spawned package must settle before planning
        _DELIVERY_COOLDOWN = 2.0  # seconds after drop-off before re-queuing for next move

        def _import_ready(p):
            """True when package has cleared its import cooldown."""
            age = _now - float(getattr(p, 'createdAt', _now))
            return p.area != 'import' or age >= _IMPORT_COOLDOWN

        def _delivery_ready(p):
            """True when package has settled after its last drop-off."""
            _delivered_at = getattr(p, 'deliveredAt', None)
            if _delivered_at is None:
                return True
            return (_now - float(_delivered_at)) >= _DELIVERY_COOLDOWN

        # Drain reorg list first — displaced packages get front-of-queue priority.
        # Reorg entries reclaim already-planned slot space so they don't consume
        # from new_entries (which caps against total_headroom for net-new work).
        for pkg_num in list(self.packagesReorgList):
            pkg_idx = next((i for i, p in enumerate(self.packages)
                            if p.packageNumber == pkg_num), None)
            if pkg_idx is not None:
                p = self.packages[pkg_idx]
                if (p.status == 'idle' and p.area != 'export'
                        and _import_ready(p)
                        and _delivery_ready(p)
                        and pkg_num not in self.packagesMoveList
                        and pkg_num not in self.packagesMovingList
                        and len(self.packagesMoveList) < effective_move_cap):
                    self.packagesMoveList.insert(0, pkg_num)
        self.packagesReorgList.clear()
        # Overdue-priority pass: idle packages past their deadline are inserted at the
        # front of the move list regardless of headroom budget.  update_packages_targetLocation
        # will either find a free export slot for them directly, or use the preemption path
        # below to displace the least-urgent non-overdue planned package.
        for _i in range(len(self.packages)):
            _p = self.packages[_i]
            if (_p.timeToDeadline.total_seconds() < 0
                    and _p.status == 'idle'
                    and _p.area != 'export'
                    and _import_ready(_p)
                    and _delivery_ready(_p)
                    and _p.packageNumber not in self.packagesMoveList
                    and _p.packageNumber not in self.packagesMovingList
                    and len(self.packagesMoveList) < effective_move_cap):
                self.packagesMoveList.insert(0, _p.packageNumber)
        # Add package to packagesMoveList if there is enough bandwidth
        if (len(self.packagesMoveList) < effective_move_cap):
            for i in range(len(self.packages)):
                if new_entries >= total_headroom:
                    break
                packageNumber = self.packages[i].packageNumber
                packageStatus = self.packages[i].status
                packageArea = self.packages[i].area
                packageAreaTarget = self.packages[i].areaTarget
                # Add to packagesMoveList Accordingly
                addPackage = False
                if (packageStatus == "idle") and _import_ready(self.packages[i]) and _delivery_ready(self.packages[i]):
                    if (packageNumber not in self.packagesMoveList) and (packageNumber not in self.packagesMovingList) and (packageArea != "export"):
                        if (packageArea == "import") and (storage_headroom > 0 or export_headroom > 0):
                            addPackage = True
                        elif (packageArea == "storage") and (export_headroom > 0):
                            addPackage = True
                        elif (packageArea == "neutral") and (storage_headroom > 0 or export_headroom > 0):
                            addPackage = True
                    if (addPackage == True):
                        self.packagesMoveList.append(packageNumber)
                        new_entries += 1
                # Cut packagesMoveList Loop if exceeded effective_move_cap
                if (len(self.packagesMoveList) >= effective_move_cap):
                    break
        return self.packagesMoveList, self.packages
    
    def update_packages_targetLocation(self):
        if not self.packagesMoveList:
            return self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount
        export_headroom  = max(0, self.numberOfExportSlots  - self.packagesInExportCount  - self.packagesPlannedInExportCount)
        storage_headroom = max(0, self.numberOfStorageSlots - self.packagesInStorageCount - self.packagesPlannedInStorageCount)

        # ── Redirect: cancel suboptimal storage targets for overdue packages ──
        # An overdue import package that was planned for storage (because export
        # was full last tick) keeps its "move planned" status indefinitely,
        # blocking the overdue-priority pass and the assignment loop from ever
        # re-routing it to export.  Cancel the plan so it can compete for export
        # headroom in this tick's assignment.
        for i in range(len(self.packages)):
            p = self.packages[i]
            if (p.packageNumber in self.packagesMoveList
                    and p.timeToDeadline.total_seconds() < 0
                    and p.status == 'move planned'
                    and p.areaTarget == 'storage'
                    and p.packageNumber not in self.packagesMovingList):
                tx, ty = p.xyLocationTarget
                self.packageTargetsInWarehouse[ty][tx] = 0
                p.xyLocationTarget = p.xyLocation.copy()
                p.areaTarget = 'none'
                p.status = 'idle'
                self.packagesPlannedInStorageCount = max(0, self.packagesPlannedInStorageCount - 1)
                storage_headroom += 1
                _log.info('overdue redirect: pkg#%d storage->idle (ttd=%.1fs)',
                          p.packageNumber, p.timeToDeadline.total_seconds())

        # ── Main pass: assign targets in deadline order (overdue first) ──
        # Routing policy: only send to export when deadline is near (<=30s) or
        # already overdue.  Packages with plenty of time go to storage first,
        # keeping export lean and reserved for urgent packages.
        _EXPORT_URGENCY_SECS = 30  # seconds before deadline to qualify for export
        for i in range(len(self.packages)):
            packageNumber = self.packages[i].packageNumber
            packageStatus = self.packages[i].status
            packageArea = self.packages[i].area
            packageAreaTarget = self.packages[i].areaTarget
            if (packageNumber in self.packagesMoveList):
                if (packageStatus == 'idle'):
                    movePackage = False
                    ttd = self.packages[i].timeToDeadline.total_seconds()
                    is_urgent = ttd <= _EXPORT_URGENCY_SECS  # overdue or near-deadline
                    _ref_xy = self.packages[i].xyLocation
                    _mode = self._pkg_target_mode
                    # Try to store package in Export (only if urgent)
                    if (movePackage == False) and (export_headroom > 0) and is_urgent:
                        if (packageArea != 'export') and (packageAreaTarget != 'export'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_EXPORT, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse, reference_xy=_ref_xy, mode=_mode)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'export'
                                self.packagesPlannedInExportCount += 1
                                export_headroom -= 1
                    # Try to store package in Storage
                    if (movePackage == False) and (storage_headroom > 0):
                        if (packageArea != "storage") and (packageAreaTarget != 'storage'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_STORAGE, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse, reference_xy=_ref_xy, mode=_mode)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'storage'
                                self.packagesPlannedInStorageCount += 1
                                storage_headroom -= 1
                    # Fallback: non-urgent package but storage full — try export anyway
                    if (movePackage == False) and (export_headroom > 0) and not is_urgent:
                        if (packageArea != 'export') and (packageAreaTarget != 'export'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_EXPORT, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse, reference_xy=_ref_xy, mode=_mode)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'export'
                                self.packagesPlannedInExportCount += 1
                                export_headroom -= 1
                    # Move the package (or take no action)
                    if (movePackage == True):
                        self.packages[i].xyLocationTarget = xyLocation.copy()
                        self.packages[i].status = 'move planned'
                        self.packages[i].plannedAt = time.time()
                        self.packageTargetsInWarehouse[xyLocation[1]][xyLocation[0]] = 1

        # ── Overdue preemption pass (runs AFTER the main pass) ──
        # The main pass processed packages by deadline (overdue first).  If an
        # overdue package couldn't get an export target because headroom was 0,
        # non-overdue packages later in the loop may have received export targets.
        # Those targets now exist as 'move planned' and can be preempted here.
        for i in range(len(self.packages)):
            p = self.packages[i]
            if (p.packageNumber in self.packagesMoveList
                    and p.status == 'idle'
                    and p.areaTarget == 'none'
                    and p.timeToDeadline.total_seconds() < 0
                    and p.area != 'export'):
                _evict_idx = None
                _evict_deadline = p.deadline
                for _j in range(len(self.packages)):
                    _pj = self.packages[_j]
                    if (_j != i
                            and _pj.status == 'move planned'
                            and _pj.areaTarget == 'export'
                            and _pj.packageNumber not in self.packagesMovingList
                            and _pj.timeToDeadline.total_seconds() > 0
                            and _pj.deadline > _evict_deadline):
                        _evict_deadline = _pj.deadline
                        _evict_idx = _j
                if _evict_idx is not None:
                    _ep = self.packages[_evict_idx]
                    _log.info('overdue preempt: pkg#%d (ttd=%.1fs) evicts pkg#%d (ttd=%.1fs)',
                              p.packageNumber, p.timeToDeadline.total_seconds(),
                              _ep.packageNumber, _ep.timeToDeadline.total_seconds())
                    _etx, _ety = _ep.xyLocationTarget
                    self.packageTargetsInWarehouse[_ety][_etx] = 0
                    _ep.xyLocationTarget = _ep.xyLocation.copy()
                    _ep.areaTarget = 'none'
                    _ep.status = 'idle'
                    self.packagesPlannedInExportCount = max(0, self.packagesPlannedInExportCount - 1)
                    export_headroom += 1
                    movePackage, xyLocation = Package_Functions.try_packageTargetLocation(
                        self.zoneMap, ZONE_EXPORT, self.packagesInWarehouse,
                        self.packageTargetsInWarehouse, self.chargersInWarehouse,
                        reference_xy=p.xyLocation, mode=self._pkg_target_mode)
                    if movePackage:
                        p.areaTarget = 'export'
                        p.xyLocationTarget = xyLocation.copy()
                        p.status = 'move planned'
                        p.plannedAt = time.time()
                        self.packageTargetsInWarehouse[xyLocation[1]][xyLocation[0]] = 1
                        self.packagesPlannedInExportCount += 1
                        export_headroom -= 1

        # Sweep: remove moveList entries that couldn't get a target this tick.
        for pkgNum in list(self.packagesMoveList):
            if pkgNum in self.packagesMovingList:
                continue
            idx = next((i for i, p in enumerate(self.packages) if p.packageNumber == pkgNum), None)
            if idx is not None and self.packages[idx].status == 'idle' and self.packages[idx].areaTarget == 'none':
                self.packagesMoveList.remove(pkgNum)
        return self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def update_packageTargetsInWarehouse(self):
        self.packageTargetsInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        if not self.packagesMoveList:
            return self.packageTargetsInWarehouse
        for packageNumber in self.packagesMoveList:
            packageIndex = [y for y, x in enumerate(self.packages) if x.packageNumber == packageNumber]
            if packageIndex:
                packageIndex = packageIndex[0]
                packageXYLocationTarget = self.packages[packageIndex].xyLocationTarget
                self.packageTargetsInWarehouse[packageXYLocationTarget[1]][packageXYLocationTarget[0]] = 1
        return self.packageTargetsInWarehouse
    
    def update_carried_packages(self):
        if not self.packages:
            return self.packages
        for i in range(len(self.packages)):
            if self.packages[i].status == "carried":
                packageNumber = self.packages[i].packageNumber
                robotNumber = self.packages[i].carrier
                robotIndex = [y for y, x in enumerate(self.robots) if x.robotNumber == robotNumber]
                if not robotIndex:
                    # Carrier robot is gone — orphan the package so it can be replanned
                    self.packages[i].status = 'idle'
                    self.packages[i].carrier = -1
                    self.packages[i].areaTarget = 'none'
                    self.packages[i].xyLocationTarget = self.packages[i].xyLocation.copy()
                    self.packages[i].deliveredAt = time.time()
                    _log.warning('orphan carried pkg#%d: carrier robot#%d not found -- reset to idle',
                                 packageNumber, robotNumber)
                    continue
                robotIndex = robotIndex[0]
                robotCarrying = self.robots[robotIndex].carrying
                if (robotCarrying == packageNumber) and (self.packages[i].xyLocation != self.robots[robotIndex].xyLocation):
                    newLocation = [self.robots[robotIndex].xyLocation[0],self.robots[robotIndex].xyLocation[1]]
                    self.packages[i].xyLocation = newLocation
                #print("- robotLocation: {}".format(self.robots[robotIndex].xyLocation))
                #print("- packageLocation: {}".format(self.packages[i].xyLocation))
        return self.packages

    def update_packages_areas(self):
        """Authoritative recount: derive area and counts from zoneMap positions."""
        zoneNames = {ZONE_IMPORT: 'import', ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}
        self.packagesInImportCount = 0
        self.packagesInStorageCount = 0
        self.packagesInExportCount = 0
        for p in self.packages:
            x, y = p.xyLocation
            zoneId = self.zoneMap[y][x]
            p.area = zoneNames.get(zoneId, 'neutral')
            if p.area == 'import':
                self.packagesInImportCount += 1
            elif p.area == 'storage':
                self.packagesInStorageCount += 1
            elif p.area == 'export':
                self.packagesInExportCount += 1
        self.packagesInWarehouseCount = len(self.packages)
        return self.packages, self.packagesInImportCount, self.packagesInStorageCount, self.packagesInExportCount

    def update_packages_colours(self, datetimeNow):
        """Assign BGR colour to each package based on status and urgency.

        Colour palette (BGR):
            Idle, on-time          — slate blue   (160, 140, 110)
            Move planned, on-time  — lime green   (60, 210, 60)
            Carried (in transit)   — cyan         (220, 180, 50)
            Overdue, no plan       — magenta      (130, 40, 210)
            Overdue, assigned      — pink         (130, 100, 220)
            Export waiting 10–60s  — lavender     (190, 130, 170)
            Export urgent <10s     — bright lav.  (210, 100, 200)

        A 1-second delivery fade blends from cyan toward idle colour after
        drop-off so recently-delivered packages don't snap to grey.
        """
        _now = time.time()
        _DELIVERY_FADE_SECS = 1.0  # seconds for cyan→idle colour transition after drop-off
        for i in range(len(self.packages)):
            p = self.packages[i]
            _overdue = p.deadline < datetimeNow
            _assigned = p.packageNumber in self.packagesMoveList or p.packageNumber in self.packagesMovingList
            _carried = p.status == 'carried'

            if _carried:
                # In transit — cyan
                p.colour = [220, 180, 50]
            elif _overdue and not _assigned:
                # Overdue, no plan — magenta
                p.colour = [130, 40, 210]
            elif _overdue and _assigned:
                # Overdue, assigned — pink
                p.colour = [130, 100, 220]
            elif _assigned:
                # On-time, assigned/en-route — lime green
                p.colour = [60, 210, 60]
            else:
                # Idle, on-time — slate blue
                p.colour = [160, 140, 110]

            # Delivery fade: blend from cyan toward idle colour over _DELIVERY_FADE_SECS
            _delivered_at = getattr(p, 'deliveredAt', None)
            if _delivered_at is not None and not _carried and not _assigned:
                _age = _now - float(_delivered_at)
                if _age < _DELIVERY_FADE_SECS:
                    _t = _age / _DELIVERY_FADE_SECS  # 0→1
                    # Blend: cyan (220,180,50) → current colour
                    _base = p.colour
                    p.colour = [
                        int(220 + (_base[0] - 220) * _t),
                        int(180 + (_base[1] - 180) * _t),
                        int(50  + (_base[2] - 50)  * _t),
                    ]

            # Export zone urgency overrides (idle packages only)
            if p.area == "export" and p.status == "idle":
                if timedelta(seconds=10) < p.timeToDeadline < timedelta(seconds=60):
                    # Export waiting — lavender
                    p.colour = [190, 130, 170]
                elif p.timeToDeadline < timedelta(seconds=10):
                    # Export urgent — bright lavender
                    p.colour = [210, 100, 200]
        return self.packages
    
    def package_export(self):
        self.packageExportCount = 0
        lenPackages = len(self.packages)
        _now = time.time()
        i = 0
        while 1:
            if (not self.packages):
                break
            try:
                packageStatus = self.packages[i].status
            except IndexError:
                break
            packageTimeToDeadline = self.packages[i].timeToDeadline
            packageArea = self.packages[i].area
            packageAreaTarget = self.packages[i].areaTarget
            packageLocation = self.packages[i].xyLocation
            packageTargetLocation = self.packages[i].xyLocationTarget
            canExport = (packageArea == "export") or (self.numberOfExportSlots == 0)
            _delivered_at = getattr(self.packages[i], 'deliveredAt', None)
            _cooldown_ok = (_delivered_at is not None) and (_now - _delivered_at >= 2.0)
            if canExport and _cooldown_ok and (packageTimeToDeadline < timedelta(seconds=0)) and (packageStatus == "idle"):
                #print("- packageTimeToDeadline: {}".format(packageTimeToDeadline))
                _delivery_time = _now - self.packages[i].createdAt
                self._flow_export_times.append(_delivery_time)
                if len(self._flow_export_times) > 200:
                    self._flow_export_times.pop(0)
                # Update delivery-time EMA
                a = self._flow_ema_alpha
                self._flow_avg_delivery = a * _delivery_time + (1 - a) * self._flow_avg_delivery if self._flow_avg_delivery > 0 else _delivery_time
                # Update throughput EMA (seconds since last export)
                _gap = _now - self._flow_last_export_t
                if _gap > 0:
                    _inst_rate = 1.0 / _gap
                    self._flow_throughput = a * _inst_rate + (1 - a) * self._flow_throughput if self._flow_throughput > 0 else _inst_rate
                self._flow_last_export_t = _now
                _log.debug('export: pkg#%d  deadline_margin=%s  loc=%s  delivery=%.0fs',
                           self.packages[i].packageNumber, packageTimeToDeadline, packageLocation, _delivery_time)
                self.packages.pop(i)
                self.packageExportCount += 1
                self.packageExportRollingCount += 1
                i -= 1
            i += 1
        return self.packages, self.packageExportCount, self.packageExportRollingCount

    # Update Warehouse
    def update_packages_in_warehouse(self):
        self.packagesInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        for i in range(len(self.packages)):
            packageXY = self.packages[i].xyLocation
            self.packagesInWarehouse[packageXY[1]][packageXY[0]] = 1
        return self.packagesInWarehouse
    
    def update_robots_in_warehouse(self):
        self.robotsInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        for i in range(len(self.robots)):
            robotXY = self.robots[i].xyLocation
            self.robotsInWarehouse[robotXY[1]][robotXY[0]] = 1
        return self.robotsInWarehouse
    
    def update_chargers_in_warehouse(self):
        self.chargersInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        for i in range(len(self.chargers)):
            chargerXY = self.chargers[i].xyLocation
            self.chargersInWarehouse[chargerXY[1]][chargerXY[0]] = 1
        return self.chargersInWarehouse
    
    # Update Robots
    def update_robots(self):
        # Import Robot
        self.robotsRollingCount, self.robotsInWarehouseCount, self.robots, self.robotsTaskAssignmentList = Robot_Functions.import_robot(Robot_Functions, self.robotSpawnMap, self.robotsRollingCount, self.robotsInWarehouseCount, self.robotsMaxQuantity, self.robotsInWarehouse, self.robots, self.robotsTaskAssignmentList, self.chargersInWarehouse) # Import Robot
        # Update self-checks
        self.robots = self.update_robots_batteryPercent() # Deplete batteryPercent
        self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers, self.packagesMoveList = self.update_robots_checkBattery() # Self-check batteryPercent
        # Update actionQueue / task assignments
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        self.packagesMovingList, self.robots, self.robotsTaskAssignmentList, self.robotsLog = self.update_robots_assign_package_availability() # Assign packagesMoveList to robots by availability
        #self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        # Update automatically
        self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers = self.update_robots_charging()
        # Update location-contingent logic
        self.robots, self.robotsTaskAssignmentList, self.robotsLog = self.update_robots_target_reached()
        self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsTaskAssignmentList, self.robotsLog = self.update_robots_target_labouring()
        #self.robots = self.update_robots_area() # Update area based on xyLocation, and areaTarget based on target package area.
        # Update xyLocation
        self.robots = self.update_robots_xyLocationTarget()
        self.robots = self.update_robots_xyLocation()
        # Update robotsInWarehouse
        self.robotsInWarehouse = self.update_robots_in_warehouse()
        # ── Queue idle-moves for truly idle robots sitting in zones ──
        # Done after all assignments so we only move robots that got no work this tick.
        for _ii in range(len(self.robots)):
            _r = self.robots[_ii]
            if (_r.status == 'idle'
                    and not _r.actionQueue
                    and not _r.decommissioning
                    and self.zoneMap[int(_r.xyLocation[1])][int(_r.xyLocation[0])] != ZONE_NONE):
                self._queue_idle_move(_ii)
        # ── Remove decommissioned robots that reached the perimeter ──
        _to_remove = [i for i, r in enumerate(self.robots) if getattr(r, '_pending_removal', False)]
        for i in reversed(_to_remove):
            self._remove_robot(i)
        # Debug
        #print(self.robots[0].actionQueue)
        #print(self.robots[0].batteryPercent)
        #print(self.robots[0].status)
        return self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsRollingCount, self.robotsTaskAssignmentList, self.robotsLog, self.chargers

    def update_chargers(self):
        # Import Charger
        self.chargersRollingCount, self.chargers = Charger_Functions.import_charger(Charger_Functions, self.chargerSpawnMap, self.chargersRollingCount, self.chargersMaxQuantity, self.chargersInWarehouse, self.chargers, self.packagesInWarehouse)
        return self.chargers, self.chargersRollingCount, self.chargersInWarehouse

    def update_robots_batteryPercent(self):
        # State-aware drain multipliers
        DRAIN_IDLE     = 0.1   # stationary, minimal systems draw
        DRAIN_MOVING   = 1.0   # baseline motor cost
        DRAIN_CARRYING = 1.5   # extra load while transporting a package
        DRAIN_CHARGING = 0.0   # no drain while docked at charger
        # Fleet-aware Power Policy: when chargers are scarce, robots conserve energy.
        # Apply damped smoothing so drain changes are gradual and less exaggerated.
        _eco_target = 1.0 - self._power_policy_eco_strength * max(0.0, self._charger_pressure - 0.5) * 2.0
        self._power_policy_eco += self._power_policy_alpha_eco * (_eco_target - self._power_policy_eco)
        _eco = float(np.clip(self._power_policy_eco, 1.0 - self._power_policy_eco_strength, 1.0))
        for i in range(len(self.robots)):
            r = self.robots[i]
            if r.status == 'charging':
                mult = DRAIN_CHARGING
            elif r.xyLocation != r.xyLocationTarget and float(getattr(r, 'velocity', 0.0)) > 0.02:
                mult = DRAIN_CARRYING if r.carrying != -1 else DRAIN_MOVING
            else:
                mult = DRAIN_IDLE
            r.batteryDrainMultiplier = mult
            r.batteryPercent -= round(r.batteryDepletingRate * mult * _eco, 4)
            r.batteryPercent = np.round(r.batteryPercent, 2)
        return self.robots
    
    def update_robots_checkBattery(self):
        # --- Reconcile charger reservations ---
        # Build ground-truth set of charger locations currently claimed by robot action queues.
        actively_claimed = set()
        for r in self.robots:
            for task in r.actionQueue:
                if task[1] in ("move to charging station", "charging"):
                    actively_claimed.add(tuple(task[0]))
        # Any charger stuck in "charging planned" with no robot heading there is a stale
        # reservation — free it so it can be reassigned.
        for charger in self.chargers:
            if charger.status == "charging planned" and tuple(charger.xyLocation) not in actively_claimed:
                charger.status = "idle"
        # --- End reconciliation ---
        # --- Adaptive Power Policy metrics ---
        _n_chargers = max(len(self.chargers), 1)
        _n_busy = sum(1 for c in self.chargers if c.status != 'idle')
        self._charger_pressure_raw = _n_busy / _n_chargers
        _alpha_p = self._power_policy_alpha_pressure
        _alpha_t = self._power_policy_alpha_threshold
        self._charger_pressure += (_alpha_p * (self._charger_pressure_raw - self._charger_pressure))
        self._avg_fleet_battery = (sum(r.batteryPercent for r in self.robots)
                                   / max(len(self.robots), 1))
        # Dynamic charge threshold: base + pressure-scaled span.
        # The active threshold follows the target smoothly to avoid abrupt policy flips.
        self._charge_threshold_target = self._charge_threshold_base + self._charge_threshold_span * self._charger_pressure_raw
        self._charge_threshold += (_alpha_t * (self._charge_threshold_target - self._charge_threshold))
        self._charge_threshold = np.clip(
            self._charge_threshold,
            self._charge_threshold_base,
            self._charge_threshold_base + self._charge_threshold_span,
        )
        # --- End Adaptive Power Policy metrics ---
        for i in range(len(self.robots)):
            if self.robots[i].decommissioning:
                continue
            # Distance-aware threshold: if the nearest charger is far,
            # the robot must start heading there earlier.
            _travel_cost_i = 0.0
            _avail_tmp, _c_tmp = self.find_available_charger(self.robots[i])
            if _avail_tmp:
                _travel_cost_i = self._estimate_charge_travel_cost(
                    self.robots[i], self.chargers[_c_tmp].xyLocation)
            _effective_threshold = self._charge_threshold + _travel_cost_i
            if self.robots[i].batteryPercent <= _effective_threshold:
                if self.robots[i].status != "charging":
                    if self.robots[i].actionQueue:
                        robotQueuedCharging = [x for x in self.robots[i].actionQueue if "move to charging station" in x]
                        if not robotQueuedCharging:
                            chargerAvailable, c = self.find_available_charger(self.robots[i])
                            if (chargerAvailable == True):
                                chargingStationLocation = self.chargers[c].xyLocation
                                #print('- chargingStationLocation: {}'.format(chargingStationLocation))
                                if ("dropoff" not in self.robots[i].status):
                                    _prev_task = self.robots[i].actionQueue[1][1] if len(self.robots[i].actionQueue) > 1 else 'none'
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, chargingStationLocation, "move to charging station")
                                    self._compute_robot_path(i, chargingStationLocation)
                                    self.chargers[c].status = "charging planned"
                                    _log.info('battery preempt: robot#%d batt=%.1f%% -> charger %s (displaced: %s)',
                                              self.robots[i].robotNumber, self.robots[i].batteryPercent,
                                              chargingStationLocation, _prev_task)
                                    # Return Package to packagesMoveList
                                    #robotActionToReturn = self.robots[i].actionQueue[1]
                                    #packagePosition = robotActionToReturn[0]
                                    #print("- packagePosition in robot.actionQueue (PRIOR): {}".format([x for x in self.robots[i].actionQueue if x[0] == packagePosition]))
                                    #packageIndex = [i for i, x in enumerate(self.packages) if x.xyLocation == packagePosition][0]
                                    #packageNumber = self.packages[packageIndex].packageNumber
                                    #self.packagesMoveList.append(packageNumber)
                                    #self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, robotActionToReturn[1])
                                    #print("- Returned: packageIndex: {}, packageNumber: {}, xyLocation: {}".format(packageIndex, self.packages[packageIndex].packageNumber, self.packages[packageIndex].xyLocation))
                                    #print("- packageNumber in self.packagesMoveList: {}".format(packageNumber in self.packagesMovelist))
                                    #print("- packagePosition in robot.actionQueue (POST): {}".format([x for x in self.robots[i].actionQueue if x[0] == packagePosition]))
                                    # Done Returning
                                    self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                #print('- Robot #{}: Battery low'.format(i))
                                #print('- robotTaskAssignmentIndex: {}'.format(robotTaskAssignmentIndex))
                    else:
                        chargerAvailable, c = self.find_available_charger(self.robots[i])
                        if (chargerAvailable == True):
                            chargingStationLocation = self.chargers[c].xyLocation
                            if ("dropoff" not in self.robots[i].status):
                                self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, chargingStationLocation, "move to charging station")
                                self._compute_robot_path(i, chargingStationLocation)
                                self.chargers[c].status = "charging planned"
                                self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                _log.info('battery preempt: robot#%d batt=%.1f%% -> charger %s (idle robot)',
                                          self.robots[i].robotNumber, self.robots[i].batteryPercent,
                                          chargingStationLocation)
                            #print('- Robot #{}: Battery low'.format(i))
                            #print('- robotTaskAssignmentIndex: {}'.format(robotTaskAssignmentIndex))
            # Opportunistic top-off: idle robots charge proactively when chargers plentiful
            elif (self.robots[i].batteryPercent <= 70
                  and self._charger_pressure < 0.3
                  and self.robots[i].status == 'idle'
                  and not self.robots[i].actionQueue):
                chargerAvailable, c = self.find_available_charger(self.robots[i])
                if chargerAvailable:
                    _csl = self.chargers[c].xyLocation
                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, _csl, "move to charging station")
                    self._compute_robot_path(i, _csl)
                    self.chargers[c].status = "charging planned"
                    self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False)
                    _log.debug('opportunistic top-off: robot#%d batt=%.1f%% pressure=%.2f -> charger %s',
                               self.robots[i].robotNumber, self.robots[i].batteryPercent,
                               self._charger_pressure, _csl)
            if self.robots[i].batteryPercent <= self._critical_batt_pct:
                # Battery critically low — emergency-drop carried package so it
                # doesn't get permanently stuck on a dying robot.
                if self.robots[i].carrying != -1:
                    _epkg = self.robots[i].carrying
                    _epidx = next((j for j, p in enumerate(self.packages)
                                   if p.packageNumber == _epkg), None)
                    if _epidx is not None:
                        _eloc = self._find_nearest_empty_cell(
                            self.robots[i].xyLocation[0], self.robots[i].xyLocation[1])
                        self.packages[_epidx].xyLocation = _eloc
                        _log.warning('emergency drop: robot#%d batt=%.1f%% force-dropping pkg#%d at %s',
                                     self.robots[i].robotNumber, self.robots[i].batteryPercent,
                                     _epkg, _eloc)
                        self.packages[_epidx].status = 'idle'
                        self.packages[_epidx].carrier = -1
                        self.packages[_epidx].areaTarget = 'none'
                        self.packages[_epidx].xyLocationTarget = self.packages[_epidx].xyLocation.copy()
                        self.packages[_epidx].deliveredAt = time.time()
                        self.packageTargetsInWarehouse[self.packages[_epidx].xyLocation[1]][self.packages[_epidx].xyLocation[0]] = 0
                        self.robots[i].carrying = -1
                        # Clean up move/moving lists
                        if _epkg in self.packagesMovingList:
                            self.packagesMovingList.remove(_epkg)
                        if _epkg in self.packagesMoveList:
                            self.packagesMoveList.remove(_epkg)
                        # Strip delivery tasks from robot queue
                        for _tname in ('move to target dropoff location', 'drop off target package',
                                       'move to target pickup location', 'pick up target package'):
                            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, _tname)
            self.robots[i].batteryPercent = Functions.ensure_limit_1d(self.robots[i].batteryPercent, 0, 100)
        
        #print(self.robotsTaskAssignmentList[0][0])
        #print(i)
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers, self.packagesMoveList
    
    @staticmethod
    def _estimate_charge_travel_cost(robot, charger_loc):
        """Estimate battery % consumed travelling to *charger_loc*.

        Uses Chebyshev distance (grid steps) × per-step drain at normal
        speed, with a 1.5× safety margin so the robot doesn't cut it
        razor-thin. The estimate is intentionally pessimistic.
        """
        _dist = max(
            abs(int(robot.xyLocation[0]) - int(charger_loc[0])),
            abs(int(robot.xyLocation[1]) - int(charger_loc[1])),
        )
        # DRAIN_MOVING = 1.0 multiplier; carrying is 1.5
        _drain_mult = 1.5 if robot.carrying != -1 else 1.0
        _cost_per_step = float(robot.batteryDepletingRate) * _drain_mult
        return _cost_per_step * _dist * 1.5   # 50 % safety margin

    def find_available_charger(self, requesting_robot=None):
        # Ground-truth: any charger location already in a robot's action queue is claimed,
        # regardless of the charger's status field (guards against stale status).
        claimed = set()
        for r in self.robots:
            for task in r.actionQueue:
                if task[1] in ("move to charging station", "charging"):
                    claimed.add(tuple(task[0]))
        # Exclude chargers physically occupied by another robot (prevents stacking).
        occupied = set()
        for r in self.robots:
            if r is not requesting_robot:
                occupied.add(tuple(r.xyLocation))
        # Return the nearest available charger (Chebyshev distance).
        _best_c = -1
        _best_dist = float('inf')
        _rx = int(requesting_robot.xyLocation[0]) if requesting_robot else 0
        _ry = int(requesting_robot.xyLocation[1]) if requesting_robot else 0
        for c in range(len(self.chargers)):
            loc = tuple(self.chargers[c].xyLocation)
            if self.chargers[c].status == "idle" and loc not in claimed and loc not in occupied:
                _d = max(abs(int(loc[0]) - _rx), abs(int(loc[1]) - _ry))
                if _d < _best_dist:
                    _best_dist = _d
                    _best_c = c
        if _best_c >= 0:
            return True, _best_c
        return False, -1

    def _plan_speed_profile(self, robot):
        """Pre-compute an ideal speed envelope for every waypoint in *robot.path*.

        Three kinematic passes are combined element-wise (minimum):
          1) Forward   – acceleration from the robot's current velocity.
          2) Goal-stop – deceleration to zero at the final waypoint.
          3) Corner    – deceleration to safe cornering speed at each turn.

        The result is stored as *robot.pathPlan*, a list of dicts aligned 1-to-1
        with *robot.path*.  Each entry contains:
          speed     – planned ceiling speed at that waypoint (cells/tick)
          turn_deg  – turn angle (degrees, 0 = straight, 180 = U-turn)
          phase     – human-readable label: cruise / accel / corner / decel / stop
        """
        path = robot.path
        if not path:
            robot.pathPlan = []
            return

        n = len(path)
        max_v = float(robot.maxVelocity)
        if robot.carrying != -1:
            max_v *= 0.92
        accel = max(float(robot.accelerationRate), 1e-6)
        decel = max(float(robot.decelerationRate), 1e-6)

        # ── segment distances (Chebyshev, >= 1) ──
        seg_dist = []
        for k in range(n - 1):
            dx = abs(int(path[k + 1][0]) - int(path[k][0]))
            dy = abs(int(path[k + 1][1]) - int(path[k][1]))
            seg_dist.append(max(dx, dy, 1))

        # ── turn angles ──
        turn_angles = [0.0] * n
        for k in range(1, n - 1):
            v1x = int(path[k][0]) - int(path[k - 1][0])
            v1y = int(path[k][1]) - int(path[k - 1][1])
            v2x = int(path[k + 1][0]) - int(path[k][0])
            v2y = int(path[k + 1][1]) - int(path[k][1])
            m1 = math.hypot(v1x, v1y)
            m2 = math.hypot(v2x, v2y)
            if m1 > 1e-6 and m2 > 1e-6:
                dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (m1 * m2)))
                turn_angles[k] = math.degrees(math.acos(dot))

        # ── corner target speeds ──
        # The live velocity-smoothing loop already handles tick-by-tick
        # acceleration and deceleration.  The speed profile only needs to
        # tell the live controller about upcoming corners (which require
        # look-ahead that the per-tick loop cannot do).
        plan = []
        for k in range(n):
            t = turn_angles[k]
            if t >= 120.0:
                spd = max_v * 0.20
            elif t >= 90.0:
                spd = max_v * 0.28
            elif t >= 60.0:
                spd = max_v * 0.40
            elif t >= 25.0:
                spd = max_v * 0.52
            else:
                spd = max_v

            # Ensure last waypoint has enough speed to step into it.
            if k == n - 1:
                spd = max(spd, max(float(robot.minMovingVelocity), accel))
                phase = 'stop'
            elif t >= 25.0:
                phase = 'corner'
            else:
                phase = 'cruise'
            plan.append({
                'speed': round(spd, 4),
                'turn_deg': round(turn_angles[k], 1),
                'phase': phase,
            })
        robot.pathPlan = plan

    def _compute_robot_path(self, robot_index, goal_xy, cooldown_ticks=0):
        """Compute an A* path from robot's current position to goal_xy.

        Uses the dense A* pathfinder (8-direction, octile heuristic).  Other
        robot positions are injected as soft-cost cells (cost 5.0) so the
        planner routes around occupied paths where possible.

        After pathfinding, assigns speed profile via ``_plan_speed_profile``
        (corner-severity speed limits) and syncs ``xyLocationTarget`` to the
        first waypoint.  Sets ``replanCooldownTicks`` to prevent immediate
        re-planning on the next tick.
        """
        robot = self.robots[robot_index]
        _goal = [int(goal_xy[0]), int(goal_xy[1])]
        _grid_w, _grid_h = self.warehouseWindowRes[0], self.warehouseWindowRes[1]
        _path = []
        if getattr(self, 'lanePlanner', None) is not None and getattr(self, 'laneMap', None) is not None:
            _path = self.lanePlanner.find_path(
                robot.xyLocation,
                _goal,
                self.laneMap,
                self.chargersInWarehouse,
            )
        if not _path:
            # Fallback to geometric A* while lane infrastructure is still maturing.
            _planner = AStarPathfinder()
            # Pass current robot positions as soft-cost cells so A* routes
            # around other robots (avoids head-on paths).
            _other_robots = set()
            for j, r in enumerate(self.robots):
                if j != robot_index:
                    _other_robots.add((int(r.xyLocation[0]), int(r.xyLocation[1])))
            _path = _planner.find_path(
                robot.xyLocation,
                _goal,
                _grid_w,
                _grid_h,
                self.chargersInWarehouse,
                _other_robots,
            )
        robot.path = _path
        robot.pathTarget = _goal
        robot.pathAge = 0
        robot.replanCooldownTicks = int(max(0, cooldown_ticks))
        # Reset fractional movement so stale velocity doesn't cause a
        # wrong-direction step on the first tick of a fresh path.
        robot.movementProgress = 0.0
        # Immediately sync xyLocationTarget so movement doesn't spend a tick
        # heading toward the old destination while the new path is active.
        if _path:
            robot.xyLocationTarget = list(_path[0])
        else:
            robot.xyLocationTarget = _goal
        self._plan_speed_profile(robot)
        if not _path and robot.xyLocation != _goal:
            _log.debug('A* no path: robot#%d from=%s to=%s',
                       robot.robotNumber, list(robot.xyLocation), _goal)

    def update_robots_assign_package_availability(self):
        if not self.packagesMoveList:
            return self.packagesMovingList, self.robots, self.robotsTaskAssignmentList, self.robotsLog
        if not self.packages:
            return self.packagesMovingList, self.robots, self.robotsTaskAssignmentList, self.robotsLog
        
        #print('-- pre-update: ')
        #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
        #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
        #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
        
        if self.packagesMoveList:
            for packageNumber in self.packagesMoveList:
                if (packageNumber not in self.packagesMovingList):
                    packageIndex = [i for i, x in enumerate(self.packages) if x.packageNumber == packageNumber]
                    if packageIndex:
                        packageIndex = packageIndex[0]
                        packageArea = self.packages[packageIndex].area
                        packageAreaTarget = self.packages[packageIndex].areaTarget
                        packageStatus = self.packages[packageIndex].status
                        if (packageStatus == "move planned"):
                            #print('- packageIndex: {}'.format(packageIndex))
                            packagePosition = self.packages[packageIndex].xyLocation
                            #print('- packagePosition: {}'.format(str(packagePosition)))
                            robotTaskQuantity = self.robotsTaskAssignmentList[0][1]
                            #print('- robotTaskQuantity: {}'.format(str(robotTaskQuantity)))
                            # Treat robots with only a "move to idle" task as available (0 real tasks)
                            _candidate_num = self.robotsTaskAssignmentList[0][0]
                            _candidate_r = next((r for r in self.robots if r.robotNumber == _candidate_num), None)
                            _effective_task_qty = robotTaskQuantity
                            if (_candidate_r and robotTaskQuantity == 1
                                    and _candidate_r.actionQueue
                                    and _candidate_r.actionQueue[0][1] == "move to idle"):
                                _effective_task_qty = 0
                            if (_effective_task_qty < self.robotsTaskAssignmentMaxQuantity):
                                robotNumberInList = _candidate_num
                                # Skip decommissioning robots
                                _candidate_robot = next((r for r in self.robots if r.robotNumber == robotNumberInList), None)
                                if _candidate_robot and _candidate_robot.decommissioning:
                                    continue
                                robotIndex = [y for y, x in enumerate(self.robots) if x.robotNumber == robotNumberInList][0]
                                # Power Policy: don't assign new work below work-safe battery floor.
                                if self.robots[robotIndex].batteryPercent < self._work_min_batt_pct:
                                    continue
                                # Trip feasibility: estimate if robot can complete round trip
                                _rx, _ry = self.robots[robotIndex].xyLocation
                                _px, _py = packagePosition
                                _tx, _ty = self.packages[packageIndex].xyLocationTarget
                                _trip_dist = abs(_rx - _px) + abs(_ry - _py) + abs(_px - _tx) + abs(_py - _ty)
                                _trip_cost = _trip_dist * self.robots[robotIndex].batteryDepletingRate * 1.25
                                _batt_avail = self.robots[robotIndex].batteryPercent - self._charge_threshold
                                if _trip_cost > _batt_avail:
                                    continue
                                #print('- self.robotsTaskAssignmentList: {}'.format(self.robotsTaskAssignmentList))
                                #print('- self.robots[{}].actionQueue: {}'.format(robotNumber, self.robots[robotNumber].actionQueue))
                                # Cancel idle-move if that's the only thing queued — it shouldn't block real work
                                if (self.robots[robotIndex].actionQueue
                                        and len(self.robots[robotIndex].actionQueue) == 1
                                        and self.robots[robotIndex].actionQueue[0][1] == "move to idle"):
                                    self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(robotIndex, "move to idle")
                                if self.robots[robotIndex].actionQueue:
                                    robotQueuedCharging = [x for x in self.robots[robotIndex].actionQueue if "move to charging station" in x]
                                    if not robotQueuedCharging:
                                        self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
                                        self._compute_robot_path(robotIndex, packagePosition)
                                        self.packagesMovingList.append(packageNumber)
                                        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                        _log.debug('task assign: robot#%d -> pickup pkg#%d at %s',
                                                   self.robots[robotIndex].robotNumber, packageNumber, packagePosition)
                                else:
                                    self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
                                    self._compute_robot_path(robotIndex, packagePosition)
                                    self.packagesMovingList.append(packageNumber)
                                    self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                    _log.debug('task assign: robot#%d -> pickup pkg#%d at %s',
                                               self.robots[robotIndex].robotNumber, packageNumber, packagePosition)

            # If extra work is possible (i.e., export is blocked, storage is not entirely filled, and import packages can not be fit into MoveList), 
        #print('-- post-update: ')
        #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
        #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
        #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))

        return self.packagesMovingList, self.robots, self.robotsTaskAssignmentList, self.robotsLog
    
    def update_robots_xyLocationTarget(self):
        """High-level task routing only.  Detects when the task goal changes
        (actionQueue front differs from pathTarget) and triggers a full replan.
        All waypoint consumption and xyLocationTarget advancement is handled
        in update_robots_xyLocation for a single source of truth.
        """
        for i in range(len(self.robots)):
            robot = self.robots[i]
            if robot.actionQueue:
                actualTargetLocation = [int(robot.actionQueue[0][0][0]),
                                        int(robot.actionQueue[0][0][1])]
                if robot.pathTarget != actualTargetLocation:
                    robot.blockedTicks = 0
                    robot.movementProgress = 0.0
                    self._compute_robot_path(i, actualTargetLocation)
            else:
                robot.path = []
                robot.pathPlan = []
                robot.pathTarget = None
                robot.pathAge = 0
                robot.replanCooldownTicks = 0
                robot.blockedTicks = 0
        return self.robots

    @staticmethod
    def _normalize_deg(angle):
        """Normalize angle into [-180, 180) for stable turn math."""
        return ((angle + 180.0) % 360.0) - 180.0

    def _rotate_toward_deg(self, current, target, max_step):
        """Rotate current heading toward target heading by at most max_step degrees."""
        err = self._normalize_deg(target - current)
        if abs(err) <= max_step:
            return target
        return current + (max_step if err > 0 else -max_step)

    @staticmethod
    def _robot_is_stopped(robot, vel_eps=0.02, progress_eps=0.05):
        """True when robot is effectively stationary for interaction-safe transitions."""
        return abs(robot.velocity) <= vel_eps and robot.movementProgress <= progress_eps

    def update_robots_xyLocation(self):
        """Move each robot one cell per tick along its planned path.

        Path-directed stepping: the next cell is determined by path[0]
        (or a direct unit vector toward the target), NOT by cardinal-
        quantising the heading angle.  This eliminates the heading→cardinal
        mismatch that caused flickering / jitter / oscillation.

        Heading rotates cosmetically toward the step direction for display.
        Speed is governed by the pre-computed pathPlan envelope, approach
        braking, and battery limits.
        """
        _occupied = set(tuple(r.xyLocation) for r in self.robots)
        _w = self.warehouseWindowRes[0]
        _h = self.warehouseWindowRes[1]

        for i in range(len(self.robots)):
            robot = self.robots[i]
            robot.pathAge += 1
            robot.replanCooldownTicks = max(0, robot.replanCooldownTicks - 1)
            _hist = list(robot.recentLocations)
            _start = tuple(robot.xyLocation)
            _occupied.discard(_start)

            # ── Consume already-reached waypoints ──────────────────────
            while robot.path:
                if (int(robot.path[0][0]) == int(robot.xyLocation[0])
                        and int(robot.path[0][1]) == int(robot.xyLocation[1])):
                    robot.path.pop(0)
                    if robot.pathPlan:
                        robot.pathPlan.pop(0)
                else:
                    break

            # ── Set immediate movement target ──────────────────────────
            if robot.path:
                robot.xyLocationTarget = list(robot.path[0])
            elif robot.actionQueue:
                robot.xyLocationTarget = [int(robot.actionQueue[0][0][0]),
                                          int(robot.actionQueue[0][0][1])]

            # ── At destination? ────────────────────────────────────────
            if robot.xyLocation == robot.xyLocationTarget:
                robot.desiredVelocity = 0.0
                robot.velocity = max(0.0, robot.velocity - robot.decelerationRate)
                if robot.velocity < 0.01:
                    robot.velocity = 0.0
                robot.movementProgress = 0.0
                robot.blockedTicks = 0
            else:
                _moved = False
                _cx = int(robot.xyLocation[0])
                _cy = int(robot.xyLocation[1])
                _tx = int(robot.xyLocationTarget[0])
                _ty = int(robot.xyLocationTarget[1])

                # ── Step direction: unit-clamped toward target cell ────
                _dx = max(-1, min(1, _tx - _cx))
                _dy = max(-1, min(1, _ty - _cy))
                _step_angle = (math.degrees(math.atan2(_dy, _dx))
                               if (_dx or _dy)
                               else float(robot.direction))

                # ── Heading (snap to step direction for display) ─────
                robot.direction = round(_step_angle, 2)
                robot.targetDirection = robot.direction
                robot.degrees = round(robot.direction)
                robot.xyLocationDiff = [_dx, _dy]
                _cardinal = Functions.find_cardinal(robot.degrees)
                robot.cardinal = _cardinal
                robot.lastCardinal = _cardinal

                # ── Desired speed ──────────────────────────────────────
                _desired = float(robot.maxVelocity)
                if robot.carrying != -1:
                    _desired *= 0.92

                # Battery limp mode
                _lt = self._limp_mode_batt_pct
                if robot.batteryPercent < _lt:
                    _desired = min(
                        _desired,
                        0.25 + 0.75 * max(robot.batteryPercent, 0)
                        / max(_lt, 1))

                # Approach braking: only slow down in the last few cells
                # to the final task goal.  The live velocity smoothing
                # handles gradual deceleration; this just ensures we don't
                # overshoot at full speed.
                _brake_pt = list(robot.xyLocationTarget)
                if robot.actionQueue:
                    _brake_pt = robot.actionQueue[0][0]
                elif robot.pathTarget:
                    _brake_pt = robot.pathTarget
                _dist_goal = max(
                    abs(int(_brake_pt[0]) - _cx),
                    abs(int(_brake_pt[1]) - _cy))
                if _dist_goal <= 1:
                    _desired = min(_desired, float(robot.minMovingVelocity))
                elif _dist_goal <= 3:
                    _desired = min(_desired, float(robot.maxVelocity) * 0.5)

                # Pre-computed speed envelope (corners, accel, decel)
                if robot.pathPlan:
                    _desired = min(
                        _desired,
                        robot.pathPlan[0].get('speed', _desired))

                # ── Velocity smoothing ─────────────────────────────────
                robot.desiredVelocity = round(_desired, 3)
                if robot.velocity < _desired:
                    robot.velocity = min(
                        _desired,
                        robot.velocity + robot.accelerationRate)
                else:
                    robot.velocity = max(
                        _desired,
                        robot.velocity - robot.decelerationRate)
                if (_desired > 0
                        and 0 < robot.velocity < robot.minMovingVelocity):
                    robot.velocity = robot.minMovingVelocity

                # ── Movement step ──────────────────────────────────────
                robot.movementProgress += robot.velocity
                if int(robot.movementProgress) >= 1 and (_dx or _dy):
                    _nx, _ny = _cx + _dx, _cy + _dy
                    _nc = (_nx, _ny)

                    if not (0 <= _nx < _w and 0 <= _ny < _h):
                        # Out of bounds — path error
                        robot.movementProgress = 0.0
                        robot.blockedTicks += 1
                    elif _nc in _occupied:
                        # Another robot occupies the cell — check for head-on
                        _blocker = None
                        for _other in self.robots:
                            if (int(_other.xyLocation[0]) == _nx
                                    and int(_other.xyLocation[1]) == _ny):
                                _blocker = _other
                                break
                        # Head-on detection: blocker is heading toward us
                        _head_on = False
                        if _blocker is not None:
                            _bdx = int(getattr(_blocker, 'xyLocationDiff', [0,0])[0])
                            _bdy = int(getattr(_blocker, 'xyLocationDiff', [0,0])[1])
                            # Opposite directions: our step + their step = (0,0)
                            if (_dx + _bdx == 0 and _dy + _bdy == 0
                                    and (_bdx or _bdy)):
                                _head_on = True
                        # Right-of-way: lower robotNumber yields sideways
                        if _head_on and robot.robotNumber < _blocker.robotNumber:
                            # Try perpendicular sidestep
                            _yielded = False
                            # Perpendicular directions to our travel
                            _perps = [(-_dy, _dx), (_dy, -_dx)]
                            for _pdx, _pdy in _perps:
                                _sx, _sy = _cx + _pdx, _cy + _pdy
                                if (0 <= _sx < _w and 0 <= _sy < _h
                                        and (_sx, _sy) not in _occupied):
                                    robot.xyLocation[0] = _sx
                                    robot.xyLocation[1] = _sy
                                    _occupied.add((_sx, _sy))
                                    robot.movementProgress -= 1.0
                                    robot.blockedTicks = 0
                                    _moved = True
                                    _yielded = True
                                    # Replan path from new position
                                    if robot.replanCooldownTicks <= 0:
                                        _goal = (list(robot.actionQueue[0][0])
                                                 if robot.actionQueue
                                                 else list(robot.xyLocationTarget))
                                        self._compute_robot_path(i, _goal, cooldown_ticks=4)
                                    break
                            if not _yielded:
                                robot.velocity = max(
                                    0.0, robot.velocity - robot.decelerationRate)
                                robot.movementProgress = min(
                                    robot.movementProgress, 0.35)
                                robot.blockedTicks += 1
                        else:
                            # Normal blocking — wait
                            robot.velocity = max(
                                0.0, robot.velocity - robot.decelerationRate)
                            robot.movementProgress = min(
                                robot.movementProgress, 0.35)
                            robot.blockedTicks += 1
                    else:
                        # Successful step
                        robot.xyLocation[0] = _nx
                        robot.xyLocation[1] = _ny
                        _occupied.add(_nc)
                        robot.movementProgress -= 1.0
                        robot.blockedTicks = 0
                        _moved = True

                        # Advance waypoints after step
                        while robot.path:
                            if (int(robot.path[0][0]) == _nx
                                    and int(robot.path[0][1]) == _ny):
                                robot.path.pop(0)
                                if robot.pathPlan:
                                    robot.pathPlan.pop(0)
                            else:
                                break
                        if robot.path:
                            robot.xyLocationTarget = list(robot.path[0])
                        elif robot.actionQueue:
                            robot.xyLocationTarget = [int(robot.actionQueue[0][0][0]),
                                                      int(robot.actionQueue[0][0][1])]
                        if robot.xyLocation == robot.xyLocationTarget:
                            robot.movementProgress = 0.0
                else:
                    # Not enough movement budget yet — count as blocked.
                    robot.blockedTicks += 1

                # ── Recovery ───────────────────────────────────────────
                if not _moved:
                    _bt = robot.blockedTicks
                    _has_path = bool(robot.path)

                    # Path drift: next waypoint is far → path is stale
                    if (_has_path
                            and max(abs(int(robot.path[0][0]) - _cx),
                                    abs(int(robot.path[0][1]) - _cy)) > 2
                            and robot.replanCooldownTicks <= 0):
                        _goal = (list(robot.actionQueue[0][0])
                                 if robot.actionQueue
                                 else list(robot.xyLocationTarget))
                        self._compute_robot_path(i, _goal, cooldown_ticks=6)
                        _has_path = bool(robot.path)

                    # Local unstick: try any free neighbour closer to
                    # target (only when far enough to avoid drift,
                    # unless the target itself is adjacent and free).
                    if _bt >= 12:
                        _target_d = max(abs(_tx - _cx), abs(_ty - _cy))
                        _best_uc = None
                        _best_ud = _target_d
                        for _udx, _udy in ((1,0),(1,1),(0,1),(-1,1),
                                            (-1,0),(-1,-1),(0,-1),(1,-1)):
                            _ux, _uy = _cx + _udx, _cy + _udy
                            if not (0 <= _ux < _w and 0 <= _uy < _h):
                                continue
                            if (_ux, _uy) in _occupied:
                                continue
                            _ud = max(abs(_tx - _ux), abs(_ty - _uy))
                            if _ud < _best_ud:
                                _best_ud = _ud
                                _best_uc = (_ux, _uy)
                        if _best_uc:
                            _sd = math.degrees(math.atan2(
                                _best_uc[1] - _cy,
                                _best_uc[0] - _cx))
                            robot.direction = self._rotate_toward_deg(
                                float(robot.direction), _sd, 180.0)
                            robot.xyLocation[0] = _best_uc[0]
                            robot.xyLocation[1] = _best_uc[1]
                            _occupied.add(_best_uc)
                            robot.movementProgress = 0.0
                            robot.blockedTicks = 0
                            _moved = True

                    # Waypoint skip (need ≥2 so there's a next target)
                    if (not _moved and _bt >= 16
                            and _has_path and len(robot.path) >= 2):
                        robot.path.pop(0)
                        if robot.pathPlan:
                            robot.pathPlan.pop(0)
                        robot.xyLocationTarget = list(robot.path[0])
                        robot.blockedTicks = 0
                        _log.debug(
                            'waypoint skip: robot#%d -> now %s',
                            robot.robotNumber, robot.xyLocationTarget)

                    # Full replan
                    _replan_at = 8 if _has_path else 14
                    if (not _moved
                            and _bt >= _replan_at
                            and robot.pathAge >= 4
                            and robot.replanCooldownTicks <= 0
                            and robot.xyLocation != robot.xyLocationTarget):
                        _goal = (list(robot.actionQueue[0][0])
                                 if robot.actionQueue
                                 else list(robot.xyLocationTarget))
                        self._compute_robot_path(i, _goal, cooldown_ticks=6)
                        _log.debug(
                            'replan: robot#%d blocked=%d at %s',
                            robot.robotNumber, _bt,
                            list(robot.xyLocation))

            # ── History ────────────────────────────────────────────────
            _hist.append(tuple(robot.xyLocation))
            if len(_hist) > 8:
                _hist = _hist[-8:]
            robot.recentLocations = _hist

            # ── Oscillation detection ──────────────────────────────────
            if robot.actionQueue and len(_hist) >= 6 and robot.pathAge >= 4:
                _a, _b, _c, _d, _e, _f = _hist[-6:]
                _abab = ((_a == _c == _e) and (_b == _d == _f)
                         and (_a != _b))
                _cluster = (len(set(_hist[-6:])) <= 2
                            and robot.blockedTicks >= 6)
                if (_abab or _cluster) and robot.replanCooldownTicks <= 0:
                    _goal = list(robot.actionQueue[0][0])
                    self._compute_robot_path(i, _goal, cooldown_ticks=6)
                    robot.velocity = min(robot.velocity, 0.08)
                    robot.movementProgress = 0.0
                    _log.debug(
                        'oscillation replan: robot#%d at %s',
                        robot.robotNumber, list(robot.xyLocation))

            _occupied.add(tuple(robot.xyLocation))

        return self.robots
    
    def update_robots_charging(self):
        for i in range(len(self.robots)):
            if self.robots[i].actionQueue:
                #print('- robotStatus: {}'.format(robotStatus))
                if (self.robots[i].status == "move to charging station"):
                    robotTargetLocation = self.robots[i].actionQueue[0][0]
                    #print("- robotTargetLocation: {}".format(robotTargetLocation))
                    _cs_matches = [x for x in self.chargers if x.xyLocation == robotTargetLocation]
                    if not _cs_matches:
                        continue  # charger moved or removed; skip safely
                    chargingStation = _cs_matches[0]
                    chargingStationNumber = chargingStation.chargerNumber
                    chargingStationLocation = chargingStation.xyLocation
                    chargingStationIndex = [i for i, x in enumerate(self.chargers) if x.chargerNumber == chargingStationNumber][0]
                    #print('- chargingStationLocation: {}'.format(chargingStationLocation))
                    if (self.robots[i].xyLocation == chargingStationLocation): # Just arriving at charging station
                        if not self._robot_is_stopped(self.robots[i]):
                            self.robots[i].desiredVelocity = 0.0
                            continue
                        _occupied_by_other = any(
                            (ri != i and tuple(self.robots[ri].xyLocation) == tuple(chargingStationLocation))
                            for ri in range(len(self.robots))
                        )
                        if _occupied_by_other:
                            # Charger cell is physically occupied; keep waiting without overlap.
                            continue
                        if chargingStation.status == "charging":
                            # Another robot beat us to this charger; pop our task and let
                            # checkBattery re-assign us to a different charger next tick.
                            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "move to charging station")
                        else:
                            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "move to charging station", chargingStationLocation, "charging")
                            self.chargers[chargingStationIndex].status = "charging"
                        #chargingActionIndex = [i2 for i2, x in enumerate(self.robots[i].actionQueue) if "move to charging station" in x][0]
                        #self.robots[i].actionQueue.pop(chargingActionIndex)
                        #robotTaskAssignmentIndex = [x for x in self.robotsTaskAssignmentList if x[0] == i][0][0]
                        #self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] -= 1
                        ##print('- self.robots[i].actionQueue: {}'.format(self.robots[i].actionQueue))
                        #self.robots[i].actionQueue.insert(0, [chargingStationLocation, "charging"])
                        #self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] += 1
                        ##print('- self.robots[i].actionQueue: {}'.format(self.robots[i].actionQueue))
                        #self.robots[i].status = "charging"
                        ##print('- self.robots[i].status: {}'.format(self.robots[i].status))
                        #self.robots[i].robotLog.append(Robot_Log('charging', self.datetimeNow))
                        #self.robotsLog.append(Robots_Log(self.robots[i].robotNumber, self.robots[i], 'charging', self.datetimeNow))
                elif (self.robots[i].status == "charging"):
                    robotTargetLocation = self.robots[i].actionQueue[0][0]
                    _cs_matches = [x for x in self.chargers if x.xyLocation == robotTargetLocation]
                    if not _cs_matches:
                        continue  # charger moved or removed; skip safely
                    chargingStation = _cs_matches[0]
                    chargingStationNumber = chargingStation.chargerNumber
                    chargingStationLocation = chargingStation.xyLocation
                    chargingStationIndex = [i for i, x in enumerate(self.chargers) if x.chargerNumber == chargingStationNumber][0]
                    if (self.robots[i].xyLocation == chargingStationLocation):
                        if self.robots[i].batteryPercent <= 98: # Charge
                            self.robots[i].batteryPercent = round(self.robots[i].batteryPercent + self.robots[i].batteryChargingRate, 1)
                        else: # Just leaving charging station
                            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "charging")
                            self.chargers[chargingStationIndex].status = "idle"
                            #chargingActionIndex = [i2 for i2, x in enumerate(self.robots[i].actionQueue) if "charging" in x][0]
                            #self.robots[i].actionQueue.pop(chargingActionIndex)
                            #robotTaskAssignmentIndex = [x for x in self.robotsTaskAssignmentList if x[0] == i][0][0]
                            ##print('- robotTaskAssignmentIndex: {}'.format(robotTaskAssignmentIndex))
                            #self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] -= 1
                            #if not self.robots[i].actionQueue:
                            #    self.robots[i].status = 'idle'
                            #    self.robots[i].robotLog.append(Robot_Log('idle', self.datetimeNow))
                            #    self.robotsLog.append(Robots_Log(self.robots[i].robotNumber, self.robots[i], 'idle', self.datetimeNow))
                            #else:
                            #    #print('- self.robots[i].actionQueue: {}'.format(self.robots[i].actionQueue))
                            #    robotNewStatus = self.robots[i].actionQueue[0][1]
                            #    self.robots[i].status = robotNewStatus
                            #    #print('- self.robots[i].status: {}'.format(self.robots[i].status))
                            #    self.robots[i].robotLog.append(Robot_Log(robotNewStatus, self.datetimeNow))
                            #    self.robotsLog.append(Robots_Log(self.robots[i].robotNumber, self.robots[i], robotNewStatus, self.datetimeNow))
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers

    def update_robots_target_reached(self):
        for i in range(len(self.robots)):
            if self.robots[i].actionQueue:
                currentLocation = self.robots[i].xyLocation
                targetLocation = self.robots[i].xyLocationTarget
                if (currentLocation == targetLocation):
                    if not self._robot_is_stopped(self.robots[i]):
                        self.robots[i].desiredVelocity = 0.0
                        continue
                    if (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "move to target pickup location"):
                        packageIndex = [y for y, x in enumerate(self.packages) if x.xyLocation == currentLocation]
                        if packageIndex:
                            packageIndex = packageIndex[0]
                            packageLocation = self.packages[packageIndex].xyLocation
                            packageLocationTarget = self.packages[packageIndex].xyLocationTarget
                            #print("- packageIndex: {}".format(packageIndex))
                            #print("- robotLocation: {}".format(currentLocation))
                            #print("- packageLocation: {}".format(packageLocation))
                            #print("- packageLocationTarget: {}".format(packageLocationTarget))
                            #print("- self.robots[i].status: {}".format(self.robots[i].status))
                            #print("- self.packages[packageIndex].status: {}".format(self.packages[packageIndex].status))
                            #print("- self.packages[packageIndex].xyLocation: {}".format(self.packages[packageIndex].xyLocation))
                            if currentLocation == packageLocation:
                                if (self.robots[i].status != "carried") and (self.packages[packageIndex].status != "carried"):
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "move to target pickup location", currentLocation, "pick up target package")
                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "move to target dropoff location"):
                        # Find the package this robot is carrying by robotNumber, not by cell
                        # (an idle package already sitting at the dropoff cell would fool a
                        # location-based search and block the carried→dropoff transition).
                        _carrying = self.robots[i].carrying
                        packageIndex = next((y for y, x in enumerate(self.packages)
                                            if x.packageNumber == _carrying), None)
                        if packageIndex is not None:
                            packageLocation = self.packages[packageIndex].xyLocation
                            packageLocationTarget = self.packages[packageIndex].xyLocationTarget
                            if currentLocation == packageLocationTarget:
                                if (self.robots[i].status != "carried") and (self.packages[packageIndex].status == "carried"):
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "move to target dropoff location", currentLocation, "drop off target package")
                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "move to idle"):
                        # Robot reached neutral idle spot — just pop the task
                        self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "move to idle")
                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "move to exit"):
                        # Decommissioned robot reached the perimeter — mark for removal
                        self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "move to exit")
                        self.robots[i]._pending_removal = True
        
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog
    
    # upon reaching export:
    # update packagesMovingList
    # update packagesMoveList
    
    def update_robots_target_labouring(self):
        for i in range(len(self.robots)):
            if self.robots[i].actionQueue:
                currentLocation = self.robots[i].xyLocation
                targetLocation = self.robots[i].xyLocationTarget
                if currentLocation == targetLocation:
                    if not self._robot_is_stopped(self.robots[i]):
                        self.robots[i].desiredVelocity = 0.0
                        continue
                    robotNumber = self.robots[i].robotNumber
                    if (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "pick up target package"):
                        packageIndex = [y for y, x in enumerate(self.packages) if x.xyLocation == currentLocation]
                        if packageIndex:
                            packageIndex = packageIndex[0]
                            packageNumber = self.packages[packageIndex].packageNumber
                            packageLocationTarget = self.packages[packageIndex].xyLocationTarget
                            if (self.robots[i].status != "carried") and (self.packages[packageIndex].status != "carried"):
                                # Block pickup if battery too low — robot can't reliably deliver
                                if self.robots[i].batteryPercent < self._work_min_batt_pct:
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "pick up target package")
                                    _log.info('pickup blocked: robot#%d batt=%.1f%% too low for pkg#%d',
                                              self.robots[i].robotNumber, self.robots[i].batteryPercent, packageNumber)
                                    continue
                                #print('--- pick up target package ---')
                                #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
                                #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
                                #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
                                #print('- packageNumber: {}'.format(str(packageNumber)))
                                #print('- robotNumber: {}'.format(str(self.robots[i].robotNumber)))
                                #print('- actionQueue: {}'.format(str(self.robots[i].actionQueue)))
                                self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "pick up target package", packageLocationTarget, "move to target dropoff location")
                                self.robots[i].carrying = packageNumber
                                self.packages[packageIndex].status = "carried"
                                self.packages[packageIndex].carrier = robotNumber
                                _log.debug('pickup: robot#%d picked pkg#%d at %s -> dropoff %s',
                                           robotNumber, packageNumber, list(currentLocation), list(packageLocationTarget))
                                #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
                                #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
                                #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
                                #print('- packageNumber: {}'.format(str(packageNumber)))
                                #print('- robotNumber: {}'.format(str(self.robots[i].robotNumber)))
                                #print('- actionQueue: {}'.format(str(self.robots[i].actionQueue)))

                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "drop off target package"):
                        # Find the package being dropped off by robot.carrying, not by
                        # xyLocationTarget — an idle package previously dropped at the
                        # same export cell may share the same target coord and would
                        # be found first by a location search.
                        _carrying = self.robots[i].carrying
                        packageIndex = next((y for y, x in enumerate(self.packages)
                                            if x.packageNumber == _carrying), None)
                        if packageIndex is not None:
                            packageNumber = self.packages[packageIndex].packageNumber
                            packageLocationTarget = self.packages[packageIndex].xyLocationTarget
                            if (self.robots[i].status != "carried") and (self.packages[packageIndex].status == "carried"):
                                #print('--- drop off target package ---')
                                #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
                                #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
                                #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
                                #print('- packageNumber: {}'.format(str(packageNumber)))
                                #print('- robotNumber: {}'.format(str(self.robots[i].robotNumber)))
                                #print('- actionQueue: {}'.format(str(self.robots[i].actionQueue)))
                                self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "drop off target package")
                                self.robots[i].carrying = -1
                                self.packagesMoveList, self.packagesMovingList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount = self.package_dropoff(packageNumber)
                                _log.debug('dropoff: robot#%d dropped pkg#%d at %s  area=%s',
                                           self.robots[i].robotNumber, packageNumber, list(currentLocation),
                                           self.packages[next((y for y, x in enumerate(self.packages) if x.packageNumber == packageNumber), -1)].area if any(x.packageNumber == packageNumber for x in self.packages) else 'exported')
                                #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
                                #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
                                #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
                                #print('- packageNumber: {}'.format(str(packageNumber)))
                                #print('- robotNumber: {}'.format(str(self.robots[i].robotNumber)))
                                #print('- actionQueue: {}'.format(str(self.robots[i].actionQueue)))
                                #exit()
        
        return self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsTaskAssignmentList, self.robotsLog
    
    def package_find_carrier_robotNumber(self, packageNumber):
        robotIndex = [i for i, x in enumerate(self.robots) if x.carrying == packageNumber]
        if robotIndex:
            robotIndex = robotIndex[0]
            robotNumber = self.robots[robotIndex].robotNumber
            return robotNumber
        return -1

    def robot_find_carrying_packageNumber(self, robotNumber):
        packageIndex = [i for i, x in enumerate(self.packages) if x.carrier == robotNumber]
        if packageIndex:
            packageIndex = packageIndex[0]
            packageNumber = self.packages[packageIndex].packageNumber
            return packageNumber
        return -1

    def package_dropoff(self, packageNumber):
        try:
            # Update Move & MovingLists
            packageMoveListIndex = self.packagesMoveList.index(packageNumber)
            packageMovingListIndex = self.packagesMovingList.index(packageNumber)
            self.packagesMoveList.pop(packageMoveListIndex)
            self.packagesMovingList.pop(packageMovingListIndex)
            # Update Planned Counts
            packageIndex = [i for i, x in enumerate(self.packages) if x.packageNumber == packageNumber][0]
            if self.packages[packageIndex].areaTarget == "import":
                self.packagesPlannedInImportCount -= 1
            elif self.packages[packageIndex].areaTarget == "storage":
                self.packagesPlannedInStorageCount -= 1
            elif self.packages[packageIndex].areaTarget == "export":
                self.packagesPlannedInExportCount -= 1
            # Update Package
            self.packages[packageIndex].status = "idle"
            self.packages[packageIndex].areaTarget = "none"
            self.packages[packageIndex].carrier = -1
            self.packages[packageIndex].deliveredAt = time.time()
        except (ValueError, IndexError):
            print("- packageNumber #{} does not exist in move/moving lists!".format(packageNumber))
            print("- packagesMoveList: {}".format(self.packagesMoveList))
            print("- packagesMovingList: {}".format(self.packagesMovingList))
            print("- lenPackages: {}".format(len(self.packages)))
            print("- packageNumber: {}".format(packageNumber))
            # Update Planned Counts
            packageIndex = [i for i, x in enumerate(self.packages) if x.packageNumber == packageNumber][0]
            if self.packages[packageIndex].areaTarget == "import":
                self.packagesPlannedInImportCount -= 1
            elif self.packages[packageIndex].areaTarget == "storage":
                self.packagesPlannedInStorageCount -= 1
            elif self.packages[packageIndex].areaTarget == "export":
                self.packagesPlannedInExportCount -= 1
            # Update Package
            packageLocation = self.packages[packageIndex].xyLocation
            packageStatus = self.packages[packageIndex].status
            print("- packageIndex: {}, xyLocation: {}, status: {}".format(packageIndex, packageLocation, packageStatus))
            self.packages[packageIndex].status = "error"
            self.packages[packageIndex].areaTarget = "none"
            self.packages[packageIndex].carrier = -1
            self.packages[packageIndex].deliveredAt = time.time()
        return self.packagesMoveList, self.packagesMovingList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def robot_insert_task(self, robotIndex, robotNewActionIndex, robotNewLocation, robotNewAction):
        robotNumber = self.robots[robotIndex].robotNumber
        self.robots[robotIndex].actionQueue.insert(robotNewActionIndex, [robotNewLocation, robotNewAction])
        robotTaskAssignmentIndex = [y for y, x in enumerate(self.robotsTaskAssignmentList) if x[0] == robotNumber][0]
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] += 1
        self.robots[robotIndex].status = robotNewAction
        self.robots[robotIndex].robotLog.append(Robot_Log(robotNewAction, self.datetimeNow))
        self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], robotNewAction, self.datetimeNow))
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        #print('- self.robots[{}].robotNumber: {}'.format(robotIndex, self.robots[robotIndex].robotNumber))
        #print('- self.robots[{}].actionQueue: {}'.format(robotIndex, self.robots[robotIndex].actionQueue))
        #print('- thisTaskAssignment: {}'.format(self.robotsTaskAssignmentList[robotTaskAssignmentIndex]))
        return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
    
    def robot_pop_task(self, robotIndex, robotActionToPop):
        robotNumber = self.robots[robotIndex].robotNumber
        _matches = [i2 for i2, x in enumerate(self.robots[robotIndex].actionQueue) if robotActionToPop in x]
        if not _matches:
            # Action not found — queue is already in the correct state; nothing to pop.
            return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
        robotActionIndex = _matches[0]
        self.robots[robotIndex].actionQueue.pop(robotActionIndex)
        robotTaskAssignmentIndex = [y for y, x in enumerate(self.robotsTaskAssignmentList) if x[0] == robotNumber][0]
        #print('- robotTaskAssignmentIndex: {}'.format(robotTaskAssignmentIndex))
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] -= 1
        if not self.robots[robotIndex].actionQueue:
            self.robots[robotIndex].status = "idle"
            self.robots[robotIndex].robotLog.append(Robot_Log('idle', self.datetimeNow))
            self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], 'idle', self.datetimeNow))
            # Decommissioning: queue exit task instead of staying idle
            if self.robots[robotIndex].decommissioning:
                self._queue_exit_task(robotIndex)
        else:
            #print('- self.robots[robotIndex].actionQueue: {}'.format(self.robots[robotIndex].actionQueue))
            robotNewStatus = self.robots[robotIndex].actionQueue[0][1]
            self.robots[robotIndex].status = robotNewStatus
            #print('- self.robots[robotIndex].status: {}'.format(self.robots[robotIndex].status))
            self.robots[robotIndex].robotLog.append(Robot_Log(robotNewStatus, self.datetimeNow))
            self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], robotNewStatus, self.datetimeNow))
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        #print('- self.robots[{}].robotNumber: {}'.format(robotIndex, self.robots[robotIndex].robotNumber))
        #print('- self.robots[{}].actionQueue: {}'.format(robotIndex, self.robots[robotIndex].actionQueue))
        #print('- thisTaskAssignment: {}'.format(self.robotsTaskAssignmentList[robotTaskAssignmentIndex]))
        return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
        
    def robot_replace_task(self, robotIndex, robotActionToPop, robotNewLocation, robotNewAction):
        robotNumber = self.robots[robotIndex].robotNumber
        _matches = [i2 for i2, x in enumerate(self.robots[robotIndex].actionQueue) if robotActionToPop in x]
        if not _matches:
            # Action not found — nothing to replace; caller should handle this case.
            return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
        robotActionIndex = _matches[0]
        self.robots[robotIndex].actionQueue.pop(robotActionIndex)
        robotTaskAssignmentIndex = [y for y, x in enumerate(self.robotsTaskAssignmentList) if x[0] == robotNumber][0]
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] -= 1
        #print('- self.robots[robotIndex].actionQueue: {}'.format(self.robots[robotIndex].actionQueue))
        self.robots[robotIndex].actionQueue.insert(robotActionIndex, [robotNewLocation, robotNewAction])
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] += 1
        #print('- self.robots[robotIndex].actionQueue: {}'.format(self.robots[robotIndex].actionQueue))
        self.robots[robotIndex].status = robotNewAction
        #print('- self.robots[robotIndex].status: {}'.format(self.robots[robotIndex].status))
        self.robots[robotIndex].robotLog.append(Robot_Log(robotNewAction, self.datetimeNow))
        self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], robotNewAction, self.datetimeNow))
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        #print('- self.robots[{}].robotNumber: {}'.format(robotIndex, self.robots[robotIndex].robotNumber))
        #print('- self.robots[{}].actionQueue: {}'.format(robotIndex, self.robots[robotIndex].actionQueue))
        #print('- thisTaskAssignment: {}'.format(self.robotsTaskAssignmentList[robotTaskAssignmentIndex]))
        return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
