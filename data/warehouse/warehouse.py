"""Core warehouse simulation engine — tick loop, flow control, scheduling, and pathfinding."""
from datetime import datetime, timedelta
import math
import random
import logging
from collections import deque
import numpy as np
import cv2 
import time

from data.functions import Functions
from data.functions_timeseries import Timeseries_Functions
from data.warehouse.warehouse_init import Warehouse_Init
from data.warehouse.package import Package
from data.warehouse.package_functions import Package_Functions
from data.warehouse.charger import Charger
from data.warehouse.charger_functions import Charger_Functions
from data.warehouse.robot import Robot, Robot_Log
from data.warehouse.robot_functions import Robot_Functions
from data.warehouse.pathfinder import AStarPathfinder
from data.warehouse.road_builder import RoadBuildJob
from data.warehouse.warehouse_data import Warehouse_Data
from data.warehouse.warehouse_log import Robots_Log
from data.constants import (ZONE_NONE, ZONE_IMPORT, ZONE_STORAGE, ZONE_EXPORT,
                            ZONE_NAMES, SIM_TICK_RATE, STALL_TICKS,
                            ROBOT_CARRY_SPEED_PENALTY, BATTERY_FULL_PCT)

# ── Module-level logger ────────────────────────────────────────────────
_log = logging.getLogger('warehouse')
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    _fh = logging.FileHandler('warehouse.log', mode='a', encoding='utf-8')
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
        Pathfinding   — dense A* with road preference, traffic congestion, and speed profiling
        Road network  — Physarum-inspired organic builder (hotspots → MST → tiered roads → plazas)
        Decommission  — graceful robot retirement with perimeter exit
    """
    _POWER_POLICY_MODES = ('eco', 'balanced', 'performance')
    _FLOW_POLICY_MODES = ('steady', 'balanced', 'throughput')
    _PKG_TARGET_MODES = ('random', 'nearest', 'zone_edge')

    @property
    def _sim_time(self):
        """Current simulation time in seconds (tick-based, deterministic)."""
        return self.warehouseLoopCount / SIM_TICK_RATE

    def __init__(self,
                    # Window Resolution, Window Center
                    windowRes = None, windowBackgroundColour = None, windowCenter = None, windowArray = None, itemsList = None, addressesList = None,
                    # Warehouse
                    warehouseLoopCount = 0, datetimeNow = None, timeStart = None, timeElapsed = 0, warehouseWindowRes = None, warehouseBackgroundColour = None, warehouseWindowCenter = None, warehouseWindowArray = None, warehousePerimeterCoordinates = None,
                    colourOfImportAreas = (10,30,10), colourOfStorageAreas = (30,10,10), colourOfExportAreas = (10,10,30),
                    logsMaxLength = 10000, dataMaxLength = 200, warehouse_data = None,
                    # Space Availability
                    packagesInImportCount = 0, packagesInStorageCount = 0, packagesInExportCount = 0, packagesInWarehouseCount = 0,
                    packagesPlannedInImportCount = 0, packagesPlannedInStorageCount = 0, packagesPlannedInExportCount = 0,
                    importSpaceAvailable = True, storageSpaceAvailable = True, exportSpaceAvailable = True,
                    packageExportCount = 0, packageExportRollingCount = 0,
                    # Packages
                    packageDimensionsLimit = 1, packagesActionsList = None,
                    packages = None, packagesLog = None, packagesMaxQuantity = 2000,
                    packagesMoveList = None, packagesMaxMoveQuantity = 120, packagesRollingCount = 0,
                    packagesInWarehouse = None, packageTargetsInWarehouse = None,
                    # Robots
                    packagesMovingList = None, robotsActionsList = None,
                    robots = None, robotsLog = None, robotsInWarehouse = None, robotsMaxQuantity = 60, robotsRollingCount = 0, robotsInWarehouseCount = 0,
                    robotsTaskAssignmentList = None, robotsTaskAssignmentStyle = 0, robotsTaskAssignmentMaxQuantity = 1,
                    numRobotsIdle = 0, numRobotsMoving = 0, numRobotsCharging = 0,
                    # Charging station(s)
                    chargersActionsList = None,
                    chargers = None, chargersMaxQuantity = 10, chargersInWarehouse = None, chargersRollingCount = 0,
                    # Spawn maps (list of [x,y] coords; None = use default perimeter)
                    chargerSpawnMap = None, robotSpawnMap = None,
                    _debug_invariants = False
                ) -> None:
        print("--- Warehouse Init ---")
        # Window Resolution, Window Center
        self.windowRes = windowRes if windowRes is not None else []
        self.windowBackgroundColour = windowBackgroundColour if windowBackgroundColour is not None else []
        self.windowCenter = windowCenter if windowCenter is not None else []
        self.windowArray = windowArray if windowArray is not None else []
        self.itemsList = itemsList if itemsList is not None else []
        self.addressesList = addressesList if addressesList is not None else []
        # Warehouse
        self.warehouseLoopCount = warehouseLoopCount
        self.datetimeNow = datetimeNow if datetimeNow is not None else datetime.now()
        self.timeStart = timeStart if timeStart is not None else time.time()
        self.timeElapsed = timeElapsed
        self.warehouseBackgroundColour = warehouseBackgroundColour if warehouseBackgroundColour is not None else []
        self.warehouseWindowRes = warehouseWindowRes if warehouseWindowRes is not None else []
        self.warehouseWindowCenter = warehouseWindowCenter if warehouseWindowCenter is not None else []
        self.warehouseWindowArray = warehouseWindowArray if warehouseWindowArray is not None else []
        self.warehousePerimeterCoordinates = warehousePerimeterCoordinates if warehousePerimeterCoordinates is not None else []
        self.logsMaxLength = logsMaxLength
        self.dataMaxLength = dataMaxLength
        self.warehouse_data = warehouse_data if warehouse_data is not None else []
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
        self.packagesActionsList = packagesActionsList if packagesActionsList is not None else ["idle", "carried"]
        self.packages = packages if packages is not None else []
        # Bounded ring buffer — auto-trims to logsMaxLength so long runs don't thrash the allocator.
        # Coerce to deque even when caller passes a plain list (e.g. save/load).
        self.packagesLog = deque(packagesLog, maxlen=logsMaxLength) if packagesLog is not None else deque(maxlen=logsMaxLength)
        self.packagesMaxQuantity = packagesMaxQuantity
        self.packagesMoveList = packagesMoveList if packagesMoveList is not None else []
        self.packagesMaxMoveQuantity = packagesMaxMoveQuantity
        self.packagesRollingCount = packagesRollingCount
        self.packagesInWarehouse = packagesInWarehouse if packagesInWarehouse is not None else []
        self.packageTargetsInWarehouse = packageTargetsInWarehouse if packageTargetsInWarehouse is not None else []
        # Robots
        self.packagesMovingList = packagesMovingList if packagesMovingList is not None else []
        self.packagesReorgList = []          # pkg numbers needing priority re-scheduling after a zone change
        self.pending_zone_changes = set()    # (gx,gy) cells painted this tick, consumed by reconcile
        self.robotsActionsList = robotsActionsList if robotsActionsList is not None else ["none", "idle", "charging", "move to charging station", "move to target pickup location", "move to target dropoff location", "pickup target package", "dropoff target package", "move to exit", "move to idle"]
        self.robots = robots if robots is not None else []
        # Bounded ring buffer — see packagesLog above.
        self.robotsLog = deque(robotsLog, maxlen=logsMaxLength) if robotsLog is not None else deque(maxlen=logsMaxLength)
        self.robotsInWarehouse = robotsInWarehouse if robotsInWarehouse is not None else []
        self.robotsMaxQuantity = robotsMaxQuantity
        self._robot_target_count = robotsMaxQuantity
        self.robotsRollingCount = robotsRollingCount
        self.robotsInWarehouseCount = robotsInWarehouseCount
        self.robotsTaskAssignmentList = robotsTaskAssignmentList if robotsTaskAssignmentList is not None else []
        self.robotsTaskAssignmentStyle = robotsTaskAssignmentStyle
        self.robotsTaskAssignmentMaxQuantity = robotsTaskAssignmentMaxQuantity
        # Charging Stations
        self.chargersActionsList = chargersActionsList if chargersActionsList is not None else ["none", "idle", "charging planned", "charging"]
        self.chargers = chargers if chargers is not None else []
        self.chargersMaxQuantity = chargersMaxQuantity
        self.chargersInWarehouse = chargersInWarehouse if chargersInWarehouse is not None else []
        self.chargersRollingCount = chargersRollingCount
        # Spawn maps (None = use default perimeter coords, set after perimeter init)
        self._chargerSpawnMap_override = chargerSpawnMap
        self._robotSpawnMap_override   = robotSpawnMap
        self._debug_invariants = _debug_invariants
        
        # Init Warehouse Screen
        self.windowCenter = Functions.get_screencenter(self.windowRes)
        self.warehouseWindowArray = self.windowArray
        self.warehousePerimeterCoordinates, self.warehousePerimeterCoordinatesMinusOne = Warehouse_Init.init_warehousePerimeter(self.windowRes)
        self.warehouseWindowRes, self.warehouseWindowCenter, self.packagesInWarehouse, self.robotsInWarehouse, self.chargersInWarehouse, self.idleAreasInWarehouse, self.packageTargetsInWarehouse = Warehouse_Init.init_warehouseWindow(self.windowRes, self.windowCenter)
        self.zoneMap = Warehouse_Init.init_zoneMap(self.warehouseWindowRes)
        self.zoneMap_original = self.zoneMap.copy()  # reference for zone restoration
        # ── Objective traffic heatmap (simulation-authoritative) ──
        # traffic_total: lifetime cumulative visit count (never decays)
        # traffic_ema:   exponential moving average, fixed 60 s half-life
        # No fleet-scaling or gain boosts — values are objective step counts.
        _gw, _gh = self.windowRes
        self.traffic_total = np.zeros((_gh, _gw), dtype=np.uint32)
        self.traffic_ema   = np.zeros((_gh, _gw), dtype=np.float32)
        self._traffic_last_time = 0.0  # sim-time of last traffic update
        # ── Road map (derived from traffic_total, display-only for now) ──
        self.road_map = np.zeros((_gh, _gw), dtype=np.uint8)
        self._road_map_prev = np.zeros((_gh, _gw), dtype=np.uint8)
        self.plaza_map = np.zeros((_gh, _gw), dtype=np.uint8)
        self._plaza_map_prev = np.zeros((_gh, _gw), dtype=np.uint8)
        self.road_hotspots = []   # list of (x, y) centroids
        self.road_edges    = []   # MST edge index-pairs into road_hotspots
        self._road_flow    = None # persistent flow accumulator for incremental updates
        self._road_job     = None # in-progress RoadBuildJob (frame-distributed)
        self._road_rebuild_count = 0        # how many times we've rebuilt
        self._road_last_rebuild = 0.0       # sim-time of last rebuild
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
        self._flow_last_export_t = 0.0  # sim-time of last export
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
        target = r.birthLocation
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
        """Remove a decommissioned robot that has reached the perimeter.

        Force-drops any package the robot is still carrying (instead of
        crashing on assertion) and sweeps every queue/list that may still
        reference the robot or its dropped package, so a single edge-case
        can't leave the warehouse in a half-consistent state.
        """
        r = self.robots[robot_index]
        rnum = r.robotNumber
        # Force-drop any carried package: reset to idle at the robot's current cell.
        if r.carrying != -1:
            _carried_pn = r.carrying
            _pkg_idx = self._find_package(_carried_pn)
            if _pkg_idx is not None:
                _pkg = self.packages[_pkg_idx]
                self._set_package_idle(_pkg, location=list(r.xyLocation))
                _log.error('decommission: robot#%d removed while carrying pkg#%d - force-dropped at %s',
                           rnum, _carried_pn, list(r.xyLocation))
            else:
                _log.error('decommission: robot#%d had carrying=%d but package not found',
                           rnum, _carried_pn)
            r.carrying = -1
            # Sweep stale references to the dropped package out of move/moving lists.
            self.packagesMoveList = [pn for pn in self.packagesMoveList if pn != _carried_pn]
            self.packagesMovingList = [pn for pn in self.packagesMovingList if pn != _carried_pn]
            self.packagesReorgList = [pn for pn in self.packagesReorgList if pn != _carried_pn]
        # Remove from task assignment list
        self.robotsTaskAssignmentList = [
            x for x in self.robotsTaskAssignmentList if x[0] != rnum]
        # Remove from robots list
        self.robots.pop(robot_index)
        self.robotsInWarehouseCount -= 1
        # Invalidate entity-lookup caches so any subsequent _find_robot doesn't return a stale index.
        self._robot_idx = {}
        self._pkg_idx = {}
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

    # ── Traffic heatmap ─────────────────────────────────────────────
    _TRAFFIC_EMA_HL = 60.0  # seconds – fixed half-life for EMA layer

    def _maybe_rebuild_roads(self):
        """Step the frame-distributed road builder.

        If a job is running, advance it by a few A* paths (~3).  When it
        finishes, harvest the result.  If no job is running, check whether
        enough time has passed to start a new one.

        This keeps per-frame cost at ~3-5 ms instead of ~70 ms all at once.
        """
        # ── advance in-progress job ─────────────────────────────────────
        if self._road_job is not None:
            self._road_job.step()          # default 3 A* paths
            if self._road_job.done:
                # Compute the new maps first, then swap all related fields
                # together so no caller can ever observe a half-updated state
                # (single-threaded, but documents intent + makes future
                # extraction to a renderer thread safer).
                _new_road_map, _new_hotspots, _new_edges, _new_flow, _new_plaza_map = self._road_job.result()
                self._road_map_prev = self.road_map.copy()
                self._plaza_map_prev = self.plaza_map.copy()
                self.road_map = _new_road_map
                self.road_hotspots = _new_hotspots
                self.road_edges = _new_edges
                self._road_flow = _new_flow
                self.plaza_map = _new_plaza_map
                self._apply_road_zone_erasure()
                self._road_job = None
            return

        # ── start new job? ──────────────────────────────────────────────
        now = self._sim_time
        interval = min(2.0 + self._road_rebuild_count * 0.2, 6.0)
        if now - self._road_last_rebuild < interval:
            return
        if self.traffic_total.max() == 0:
            return
        # Wait until every robot has charged at least once before building roads
        if self.robots and not all(r.hasCharged for r in self.robots):
            return
        self._road_last_rebuild = now
        self._road_rebuild_count += 1
        self._road_job = RoadBuildJob(
            self.traffic_total, self.zoneMap,
            self.traffic_ema, self.chargersInWarehouse,
            self._road_flow)
        self._road_job.step()  # start first batch immediately

    def _apply_road_zone_erasure(self):
        """Erase zones under road/plaza cells; restore zones on freed cells.

        When a road/plaza is built on a zone cell, the zone is cleared to
        neutral.  When a road/plaza disappears, the zone is restored from
        the original map — but only if that zone type is below its efficient
        target proportion (20 % import, 45 % storage, 35 % export).
        Idle packages on road/plaza cells are force-added to the move list.
        """
        _TARGET_RATIOS = {
            ZONE_IMPORT:  0.20,
            ZONE_STORAGE: 0.45,
            ZONE_EXPORT:  0.35,
        }

        cur_covered = (self.road_map > 0) | (self.plaza_map > 0)
        prev_covered = (self._road_map_prev > 0) | (self._plaza_map_prev > 0)

        # ── 1. Erase zones under NEW road/plaza cells ──────────────────
        new_roads = cur_covered & ~prev_covered
        erase = new_roads & (self.zoneMap != 0)
        if erase.any():
            self.zoneMap[erase] = 0

        # Also ensure ALL current road/plaza cells are neutral
        still_covered = cur_covered & (self.zoneMap != 0)
        if still_covered.any():
            self.zoneMap[still_covered] = 0

        # ── 2. Restore zones on FREED cells (was road, now isn't) ──────
        freed = prev_covered & ~cur_covered
        if freed.any():
            # Which freed cells had zones in the original map?
            orig_zones = self.zoneMap_original[freed]
            restorable = orig_zones > 0
            if restorable.any():
                # Current zone counts (excluding road/plaza cells)
                n_import  = int(np.count_nonzero(self.zoneMap == ZONE_IMPORT))
                n_storage = int(np.count_nonzero(self.zoneMap == ZONE_STORAGE))
                n_export  = int(np.count_nonzero(self.zoneMap == ZONE_EXPORT))
                n_total   = max(n_import + n_storage + n_export, 1)

                # Deficit: how far below target each zone type is
                deficit = {}
                for zid, target_ratio in _TARGET_RATIOS.items():
                    current_count = {ZONE_IMPORT: n_import, ZONE_STORAGE: n_storage, ZONE_EXPORT: n_export}[zid]
                    target_count = target_ratio * n_total
                    deficit[zid] = max(target_count - current_count, 0)

                # Restore freed cells — prioritize zones with largest deficit
                freed_ys, freed_xs = np.where(freed)
                for fy, fx in zip(freed_ys, freed_xs):
                    orig_z = self.zoneMap_original[fy, fx]
                    if orig_z == 0:
                        continue
                    # Only restore if this zone type has a deficit
                    if deficit.get(orig_z, 0) > 0:
                        self.zoneMap[fy, fx] = orig_z
                        deficit[orig_z] -= 1

        self.update_zone_counts()

        # ── 3. Evict idle packages sitting on road/plaza cells ─────────
        for pkg in self.packages:
            if pkg.status != 'idle':
                continue
            px, py = pkg.xyLocation
            if not cur_covered[py, px]:
                continue
            if (pkg.packageNumber in self.packagesMoveList
                    or pkg.packageNumber in self.packagesMovingList):
                continue
            self.packagesMoveList.insert(0, pkg.packageNumber)

    def _update_traffic_heatmap(self):
        """Accumulate objective robot traffic into simulation-authoritative heatmaps.

        traffic_total: +1 per moving robot per cell per tick (never decays).
        traffic_ema:   exponential moving average, fixed 60 s half-life.
        No fleet-scaling or gain boosts — values are objective step counts.
        """
        _STATIC = {'idle', 'charging'}
        now = self._sim_time
        dt = max(now - self._traffic_last_time, 0.0)
        self._traffic_last_time = now
        # Time-anchored exponential decay
        if dt > 0:
            self.traffic_ema *= 0.5 ** (dt / self._TRAFFIC_EMA_HL)
        for rb in self.robots:
            # Only count robots that are actively moving and have carried at least once
            if (rb.velocity > 0 and rb.hasCarried):
                rx, ry = rb.xyLocation
                self.traffic_total[ry, rx] += 1
                self.traffic_ema[ry, rx]   += 1.0

    def update_warehouse(self):
        """Main simulation tick.  Called once per sim frame (~40 Hz).

        Orchestrates the full update sequence: zone counts, flow control,
        zone-change reconciliation, package lifecycle (spawn/sort/assign/
        colour/export), charger spawning, robot lifecycle (spawn/assign/
        pathfind/move/battery/idle-park/decommission), occupancy arrays,
        and periodic summary logging.
        """
        self.datetimeNow = datetime.now() # Update datetime
        self.timeElapsed = time.time() - self.timeStart
        self._update_traffic_heatmap()
        self._maybe_rebuild_roads()
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
        if self._debug_invariants:
            self._check_invariants()
        # Always-on cheap invariant + auto-repair (every 100 ticks ~= every 2.5s sim time at 40 Hz).
        if (self.warehouseLoopCount % 100) == 0:
            self._check_invariants_lightweight()
        return self

    def _check_invariants(self):
        """Validate internal consistency (debug-only, off by default).

        Logs errors for any detected inconsistencies without crashing.
        """
        _pkg_ids = {p.packageNumber for p in self.packages}
        # 1. Every ID in packagesMoveList must exist in packages
        for pn in self.packagesMoveList:
            if pn not in _pkg_ids:
                _log.error('INVARIANT: packagesMoveList contains pkg#%d not in packages', pn)
        # 2. Every ID in packagesMovingList must exist and be carried or have active pickup
        _active_pickups = set()
        for r in self.robots:
            for t in r.actionQueue:
                if 'pickup' in t[1]:
                    _active_pickups.update(
                        p.packageNumber for p in self.packages
                        if p.xyLocation == t[0] and p.status in ('move planned', 'carried'))
        for pn in self.packagesMovingList:
            if pn not in _pkg_ids:
                _log.error('INVARIANT: packagesMovingList contains pkg#%d not in packages', pn)
                continue
            p = self.packages[self._find_package(pn)]
            if p.status != 'carried' and pn not in _active_pickups:
                _log.error('INVARIANT: pkg#%d in movingList but status=%s with no active pickup', pn, p.status)
        # 3. No package with carrier != -1 while status is idle
        for p in self.packages:
            if p.carrier not in (None, -1) and p.status == 'idle':
                _log.error('INVARIANT: pkg#%d status=idle but carrier=%s', p.packageNumber, p.carrier)
        # 4. Every robot carrying must have a matching carried package
        for r in self.robots:
            if r.carrying != -1:
                pi = self._find_package(r.carrying)
                if pi is None:
                    _log.error('INVARIANT: robot#%d carrying pkg#%d but package not found', r.robotNumber, r.carrying)
                elif self.packages[pi].status != 'carried':
                    _log.error('INVARIANT: robot#%d carrying pkg#%d but status=%s', r.robotNumber, r.carrying, self.packages[pi].status)
        # 5. Planned counts match actual areaTarget counts
        _planned_imp = sum(1 for p in self.packages if p.areaTarget == 'import')
        _planned_sto = sum(1 for p in self.packages if p.areaTarget == 'storage')
        _planned_exp = sum(1 for p in self.packages if p.areaTarget == 'export')
        if _planned_imp != self.packagesPlannedInImportCount:
            _log.error('INVARIANT: plannedImport count=%d vs actual=%d', self.packagesPlannedInImportCount, _planned_imp)
        if _planned_sto != self.packagesPlannedInStorageCount:
            _log.error('INVARIANT: plannedStorage count=%d vs actual=%d', self.packagesPlannedInStorageCount, _planned_sto)
        if _planned_exp != self.packagesPlannedInExportCount:
            _log.error('INVARIANT: plannedExport count=%d vs actual=%d', self.packagesPlannedInExportCount, _planned_exp)

    def _check_invariants_lightweight(self):
        """Cheap always-on invariant sweep with auto-repair (no raises).

        Catches the few corruption shapes that have actually shown up in
        practice: carrier/carrying asymmetry, idle-with-stale-areaTarget
        zombies, and entity coords outside the grid. Repairs in place and
        logs at error level so issues remain visible.
        """
        try:
            _w = int(self.warehouseWindowRes[0])
            _h = int(self.warehouseWindowRes[1])
        except Exception:
            return
        # 1. Carrier <-> carrying symmetry. Auto-repair: drop the dangling carry.
        for r in self.robots:
            if r.carrying == -1:
                continue
            pi = self._find_package(r.carrying)
            if pi is None:
                _log.error('INVARIANT-LITE: robot#%d carrying pkg#%d (not found) - clearing',
                           r.robotNumber, r.carrying)
                r.carrying = -1
                continue
            p = self.packages[pi]
            if p.status != 'carried' or p.carrier != r.robotNumber:
                _log.error('INVARIANT-LITE: robot#%d carries pkg#%d but pkg.status=%s carrier=%s - resyncing',
                           r.robotNumber, p.packageNumber, p.status, p.carrier)
                p.status = 'carried'
                p.carrier = r.robotNumber
        # 2. Reverse: package marked carried must point at a real robot.
        for p in self.packages:
            if p.status != 'carried':
                continue
            if p.carrier in (None, -1) or self._find_robot(p.carrier) is None:
                _log.error('INVARIANT-LITE: pkg#%d status=carried but carrier=%s missing - dropping to idle',
                           p.packageNumber, p.carrier)
                self._set_package_idle(p)
        # 3. Idle package with stale areaTarget plan = "zombie idle". Auto-repair.
        for p in self.packages:
            if p.status == 'idle' and p.areaTarget != 'none':
                _log.error('INVARIANT-LITE: pkg#%d idle but areaTarget=%s - clearing stale plan',
                           p.packageNumber, p.areaTarget)
                self._set_package_idle(p)
        # 4. Entity coordinates within grid bounds.
        for r in self.robots:
            try:
                _x, _y = int(r.xyLocation[0]), int(r.xyLocation[1])
            except Exception:
                _log.error('INVARIANT-LITE: robot#%d xyLocation=%r unreadable',
                           r.robotNumber, r.xyLocation)
                continue
            if not (0 <= _x < _w and 0 <= _y < _h):
                _log.error('INVARIANT-LITE: robot#%d out-of-grid at (%d,%d) - clamping',
                           r.robotNumber, _x, _y)
                r.xyLocation = [max(0, min(_w - 1, _x)), max(0, min(_h - 1, _y))]
        for p in self.packages:
            try:
                _x, _y = int(p.xyLocation[0]), int(p.xyLocation[1])
            except Exception:
                continue
            if not (0 <= _x < _w and 0 <= _y < _h):
                _log.error('INVARIANT-LITE: pkg#%d out-of-grid at (%d,%d) - clamping',
                           p.packageNumber, _x, _y)
                p.xyLocation = [max(0, min(_w - 1, _x)), max(0, min(_h - 1, _y))]

    def update_zone_counts(self):
        """Recompute slot counts from zoneMap. Supports dynamic zone changes."""
        self.numberOfImportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_IMPORT))
        self.numberOfStorageSlots = int(np.count_nonzero(self.zoneMap == ZONE_STORAGE))
        self.numberOfExportSlots  = int(np.count_nonzero(self.zoneMap == ZONE_EXPORT))
        self.packagesMaxQuantity  = self.numberOfImportSlots + self.numberOfStorageSlots + self.numberOfExportSlots

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
        recent = [p for p in self.packages if (self._sim_time - p.createdAt) < 2.0]
        if recent:
            # For recently-spawned packages, timeToDeadline still approximates the full span
            avg_dl = sum(p.timeToDeadline.total_seconds() for p in recent) / len(recent)
            a = self._flow_ema_alpha
            self._flow_avg_deadline = a * avg_dl + (1 - a) * self._flow_avg_deadline if self._flow_avg_deadline > 0 else avg_dl

    def invalidate_stale_targets(self):
        """Cancel in-flight moves whose target cells no longer match the intended zone."""
        expected_zone = {'import': ZONE_IMPORT, 'storage': ZONE_STORAGE, 'export': ZONE_EXPORT}
        PICKUP_ACTIONS  = {"move to target pickup location", "pick up target package"}
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
                        # Strip inbound robot pickup tasks + clean lists
                        if p.packageNumber in self.packagesMovingList:
                            self.packagesMovingList.remove(p.packageNumber)
                        px, py = p.xyLocation
                        for r in self.robots:
                            if any(t[0] == [px, py] and t[1] in PICKUP_ACTIONS
                                   for t in r.actionQueue):
                                new_q = [t for t in r.actionQueue
                                         if not (t[0] == [px, py] and t[1] in PICKUP_ACTIONS)]
                                removed = len(r.actionQueue) - len(new_q)
                                if removed:
                                    r.actionQueue = new_q
                                    tai = next((j for j, x in enumerate(self.robotsTaskAssignmentList)
                                                if x[0] == r.robotNumber), None)
                                    if tai is not None:
                                        self.robotsTaskAssignmentList[tai][1] = max(
                                            0, self.robotsTaskAssignmentList[tai][1] - removed)
                                    r.status = r.actionQueue[0][1] if r.actionQueue else 'idle'
                    elif p.status == 'carried':
                        # Abort carry — mirrors reconcile_zone_changes Case C
                        p.status = 'idle'
                        if p.packageNumber in self.packagesMovingList:
                            self.packagesMovingList.remove(p.packageNumber)
                        if p.carrier not in (None, -1):
                            ri = self._find_robot(p.carrier)
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

        # Snapshot then clear so any callback that fires mid-iteration (paint
        # events fire inside cv2.waitKey, single-threaded) accumulates into a
        # fresh set instead of mutating what we're iterating.
        changed_cells = set(self.pending_zone_changes)
        self.pending_zone_changes = set()

        PICKUP_ACTIONS  = {"move to target pickup location", "pick up target package"}
        DROPOFF_ACTIONS = {"move to target dropoff location", "drop off target package"}

        def _strip_robot_tasks(robot_number, action_set, location=None):
            """Remove matching tasks from a robot's action queue and adjust counts."""
            ri = self._find_robot(robot_number)
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
            try:
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
                        ri = self._find_robot(p.carrier)
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
            except Exception:
                # One bad cell must not leave the warehouse half-reconciled. Log and continue.
                _log.exception('reconcile_zone_changes: failed for pkg#%s - skipping',
                               getattr(p, 'packageNumber', '?'))

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
        # No-op: packagesLog / robotsLog are bounded deques (maxlen=logsMaxLength) and self-trim on append.
        # Retained for backwards-compatibility with existing call sites.
        return self.packagesLog, self.robotsLog

    def record_warehouse_data(self):
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
        self.warehouse_data.numRobotsInWarehouse = Timeseries_Functions.rollUpdate(self.warehouse_data.numRobotsInWarehouse, 1, self.robotsInWarehouseCount)
        return self

    # Update Packages
    def update_packages(self):
        self._rebuild_entity_indices()
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
            _prev_count = len(self.packages)
            self.packagesRollingCount, self.packagesInImportCount, self.packagesInWarehouseCount, self.packages, self.packagesLog = Package_Functions.import_package(Package_Functions, self.zoneMap, self.chargersInWarehouse, self.packagesRollingCount, self.packagesInImportCount, self.packagesInWarehouseCount, self.packagesInWarehouse, self.packageTargetsInWarehouse, _effective_cap, self.packages, self.packagesLog, self.itemsList, self.addressesList, self.datetimeNow, spawnZoneId=spawnZoneId) # Import package
            # Stamp new packages with sim-time
            for _pi in range(_prev_count, len(self.packages)):
                self.packages[_pi].createdAt = self._sim_time
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
        # Rebuild in deadline order: self.packages is already sorted by deadline,
        # so iterating it preserves the ordering contract (front = most urgent).
        move_set = set(self.packagesMoveList)
        sorted_in_move = [p.packageNumber for p in self.packages if p.packageNumber in move_set]
        # Defensive: keep any moveList IDs not found in packages (shouldn't happen)
        remainder = [pn for pn in self.packagesMoveList if pn not in move_set]
        self.packagesMoveList = sorted_in_move + remainder
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
            _zidx = self._find_package(_zpkg)
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
                    self.packages[_zidx].deliveredAt = self._sim_time
                    self.robots[_zrobot_idx].carrying = -1
                    if _zpkg in self.packagesMovingList:
                        self.packagesMovingList.remove(_zpkg)
                    if _zpkg in self.packagesMoveList:
                        self.packagesMoveList.remove(_zpkg)

        for pkg_num in list(self.packagesMovingList):
            if pkg_num in _carried_pkgs:
                continue
            pkg_idx = self._find_package(pkg_num)
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
        _now_stale = self._sim_time
        _STALE_PLAN_TIMEOUT = 5.0  # seconds before an unassigned 'move planned' is reset
        for packageNumber in list(reversed(self.packagesMoveList)):  # snapshot — list is mutated inside loop
            if (packageNumber not in self.packagesMovingList):
                packageIndex = self._find_package(packageNumber)
                if packageIndex is not None:
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
                        _planned_at = self.packages[packageIndex].plannedAt
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

        _now = self._sim_time
        _IMPORT_COOLDOWN = 2.0  # seconds a newly-spawned package must settle before planning
        _DELIVERY_COOLDOWN = 2.0  # seconds after drop-off before re-queuing for next move

        def _import_ready(p):
            """True when package has cleared its import cooldown."""
            age = _now - float(p.createdAt)
            return p.area != 'import' or age >= _IMPORT_COOLDOWN

        def _delivery_ready(p):
            """True when package has settled after its last drop-off."""
            _delivered_at = p.deliveredAt
            if _delivered_at is None:
                return True
            return (_now - float(_delivered_at)) >= _DELIVERY_COOLDOWN

        # Drain reorg list first — displaced packages get front-of-queue priority.
        # Reorg entries reclaim already-planned slot space so they don't consume
        # from new_entries (which caps against total_headroom for net-new work).
        for pkg_num in list(self.packagesReorgList):
            pkg_idx = self._find_package(pkg_num)
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
                        self.packages[i].plannedAt = self._sim_time
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
                        p.plannedAt = self._sim_time
                        self.packageTargetsInWarehouse[xyLocation[1]][xyLocation[0]] = 1
                        self.packagesPlannedInExportCount += 1
                        export_headroom -= 1

        # Sweep: remove moveList entries that couldn't get a target this tick.
        for pkgNum in list(self.packagesMoveList):
            if pkgNum in self.packagesMovingList:
                continue
            idx = self._find_package(pkgNum)
            if idx is not None and self.packages[idx].status == 'idle' and self.packages[idx].areaTarget == 'none':
                self.packagesMoveList.remove(pkgNum)
        return self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def update_packageTargetsInWarehouse(self):
        self.packageTargetsInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        if not self.packagesMoveList:
            return self.packageTargetsInWarehouse
        for packageNumber in self.packagesMoveList:
            packageIndex = self._find_package(packageNumber)
            if packageIndex is not None:
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
                robotIndex = self._find_robot(robotNumber)
                if robotIndex is None:
                    # Carrier robot is gone — orphan the package so it can be replanned
                    self.packages[i].status = 'idle'
                    self.packages[i].carrier = -1
                    self.packages[i].areaTarget = 'none'
                    self.packages[i].xyLocationTarget = self.packages[i].xyLocation.copy()
                    self.packages[i].deliveredAt = self._sim_time
                    _log.warning('orphan carried pkg#%d: carrier robot#%d not found -- reset to idle',
                                 packageNumber, robotNumber)
                    continue
                robotCarrying = self.robots[robotIndex].carrying
                if (robotCarrying == packageNumber) and (self.packages[i].xyLocation != self.robots[robotIndex].xyLocation):
                    newLocation = [self.robots[robotIndex].xyLocation[0],self.robots[robotIndex].xyLocation[1]]
                    self.packages[i].xyLocation = newLocation
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
        _now = self._sim_time
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
            _delivered_at = p.deliveredAt
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
        _now = self._sim_time
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
            _delivered_at = self.packages[i].deliveredAt
            # 1e-6 epsilon guards against float-rounding flake when both times derive from sim_time accumulation.
            _cooldown_ok = (_delivered_at is not None) and (_now - _delivered_at >= 2.0 - 1e-6)
            if canExport and _cooldown_ok and (packageTimeToDeadline < timedelta(seconds=0)) and (packageStatus == "idle"):
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
        _h, _w = self.packagesInWarehouse.shape
        for i in range(len(self.packages)):
            packageXY = self.packages[i].xyLocation
            _x, _y = int(packageXY[0]), int(packageXY[1])
            if 0 <= _x < _w and 0 <= _y < _h:
                self.packagesInWarehouse[_y][_x] = 1
            else:
                _log.error('update_packages_in_warehouse: pkg#%d out-of-grid at (%d,%d) - skipped',
                           self.packages[i].packageNumber, _x, _y)
        return self.packagesInWarehouse
    
    def update_robots_in_warehouse(self):
        self.robotsInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        _h, _w = self.robotsInWarehouse.shape
        for i in range(len(self.robots)):
            robotXY = self.robots[i].xyLocation
            _x, _y = int(robotXY[0]), int(robotXY[1])
            if 0 <= _x < _w and 0 <= _y < _h:
                self.robotsInWarehouse[_y][_x] = 1
            else:
                _log.error('update_robots_in_warehouse: robot#%d out-of-grid at (%d,%d) - skipped',
                           self.robots[i].robotNumber, _x, _y)
        return self.robotsInWarehouse
    
    def update_chargers_in_warehouse(self):
        self.chargersInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        _h, _w = self.chargersInWarehouse.shape
        for i in range(len(self.chargers)):
            chargerXY = self.chargers[i].xyLocation
            _x, _y = int(chargerXY[0]), int(chargerXY[1])
            if 0 <= _x < _w and 0 <= _y < _h:
                self.chargersInWarehouse[_y][_x] = 1
            else:
                _log.error('update_chargers_in_warehouse: charger#%d out-of-grid at (%d,%d) - skipped',
                           self.chargers[i].chargerNumber, _x, _y)
        return self.chargersInWarehouse
    
    # Update Robots
    def update_robots(self):
        self._rebuild_entity_indices()
        # Import Robot
        _prev_robot_count = len(self.robots)
        self.robotsRollingCount, self.robotsInWarehouseCount, self.robots, self.robotsTaskAssignmentList = Robot_Functions.import_robot(Robot_Functions, self.robotSpawnMap, self.robotsRollingCount, self.robotsInWarehouseCount, self.robotsMaxQuantity, self.robotsInWarehouse, self.robots, self.robotsTaskAssignmentList, self.chargersInWarehouse) # Import Robot
        # Stamp new robots with sim-time
        for _ri in range(_prev_robot_count, len(self.robots)):
            self.robots[_ri].createdAt = self._sim_time
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
        _to_remove = [i for i, r in enumerate(self.robots) if r._pending_removal]
        for i in reversed(_to_remove):
            self._remove_robot(i)
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
            elif r.xyLocation != r.xyLocationTarget and float(r.velocity) > 0.02:
                mult = DRAIN_CARRYING if r.carrying != -1 else DRAIN_MOVING
            else:
                mult = DRAIN_IDLE
            r.batteryDrainMultiplier = mult
            r.batteryPercent -= round(r.batteryDepletingRate * mult * _eco, 4)
            r.batteryPercent = np.round(r.batteryPercent, 2)
        return self.robots
    
    def update_robots_checkBattery(self):
        self._reconcile_charger_reservations()
        self._update_charge_policy_metrics()
        for i in range(len(self.robots)):
            if self.robots[i].decommissioning:
                continue
            self._dispatch_charging_if_needed(i)
            self._emergency_battery_drop(i)
            self.robots[i].batteryPercent = Functions.ensure_limit_1d(self.robots[i].batteryPercent, 0, 100)
        
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers, self.packagesMoveList

    def _reconcile_charger_reservations(self):
        """Free chargers stuck in 'charging planned' with no robot heading there."""
        actively_claimed = set()
        for r in self.robots:
            for task in r.actionQueue:
                if task[1] in ("move to charging station", "charging"):
                    actively_claimed.add(tuple(task[0]))
        for charger in self.chargers:
            if charger.status == "charging planned" and tuple(charger.xyLocation) not in actively_claimed:
                charger.status = "idle"

    def _update_charge_policy_metrics(self):
        """Recompute adaptive power-policy pressure and threshold each tick."""
        _n_chargers = max(len(self.chargers), 1)
        _n_busy = sum(1 for c in self.chargers if c.status != 'idle')
        self._charger_pressure_raw = _n_busy / _n_chargers
        _alpha_p = self._power_policy_alpha_pressure
        _alpha_t = self._power_policy_alpha_threshold
        self._charger_pressure += (_alpha_p * (self._charger_pressure_raw - self._charger_pressure))
        self._avg_fleet_battery = (sum(r.batteryPercent for r in self.robots)
                                   / max(len(self.robots), 1))
        self._charge_threshold_target = self._charge_threshold_base + self._charge_threshold_span * self._charger_pressure_raw
        self._charge_threshold += (_alpha_t * (self._charge_threshold_target - self._charge_threshold))
        self._charge_threshold = np.clip(
            self._charge_threshold,
            self._charge_threshold_base,
            self._charge_threshold_base + self._charge_threshold_span,
        )

    def _dispatch_charging_if_needed(self, i):
        """Send robot *i* to a charger if battery is below the effective threshold."""
        robot = self.robots[i]
        # Distance-aware threshold: if the nearest charger is far,
        # the robot must start heading there earlier.
        _travel_cost_i = 0.0
        _avail_tmp, _c_tmp = self.find_available_charger(robot)
        if _avail_tmp:
            _travel_cost_i = self._estimate_charge_travel_cost(
                robot, self.chargers[_c_tmp].xyLocation)
        _effective_threshold = self._charge_threshold + _travel_cost_i
        if robot.batteryPercent <= _effective_threshold:
            if robot.status != "charging":
                robotQueuedCharging = [x for x in robot.actionQueue if "move to charging station" in x]
                if not robotQueuedCharging:
                    chargerAvailable, c = self.find_available_charger(robot)
                    if chargerAvailable and "dropoff" not in robot.status:
                        chargingStationLocation = self.chargers[c].xyLocation
                        _prev_task = robot.actionQueue[1][1] if len(robot.actionQueue) > 1 else 'none'
                        self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, chargingStationLocation, "move to charging station")
                        self._compute_robot_path(i, chargingStationLocation)
                        self.chargers[c].status = "charging planned"
                        _log.info('battery preempt: robot#%d batt=%.1f%% -> charger %s (displaced: %s)',
                                  robot.robotNumber, robot.batteryPercent,
                                  chargingStationLocation, _prev_task)
                        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False)
        # Opportunistic top-off: idle robots charge proactively when chargers plentiful
        elif (robot.batteryPercent <= 70
              and self._charger_pressure < 0.3
              and robot.status == 'idle'
              and not robot.actionQueue):
            chargerAvailable, c = self.find_available_charger(robot)
            if chargerAvailable:
                _csl = self.chargers[c].xyLocation
                self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, _csl, "move to charging station")
                self._compute_robot_path(i, _csl)
                self.chargers[c].status = "charging planned"
                self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False)
                _log.debug('opportunistic top-off: robot#%d batt=%.1f%% pressure=%.2f -> charger %s',
                           robot.robotNumber, robot.batteryPercent,
                           self._charger_pressure, _csl)

    def _emergency_battery_drop(self, i):
        """Force-drop the carried package if battery is critically low."""
        robot = self.robots[i]
        if robot.batteryPercent > self._critical_batt_pct:
            return
        if robot.carrying == -1:
            return
        _epkg = robot.carrying
        _epidx = self._find_package(_epkg)
        if _epidx is None:
            return
        _eloc = self._find_nearest_empty_cell(robot.xyLocation[0], robot.xyLocation[1])
        self.packages[_epidx].xyLocation = _eloc
        _log.warning('emergency drop: robot#%d batt=%.1f%% force-dropping pkg#%d at %s',
                     robot.robotNumber, robot.batteryPercent, _epkg, _eloc)
        self.packages[_epidx].status = 'idle'
        self.packages[_epidx].carrier = -1
        self.packages[_epidx].areaTarget = 'none'
        self.packages[_epidx].xyLocationTarget = self.packages[_epidx].xyLocation.copy()
        self.packages[_epidx].deliveredAt = self._sim_time
        self.packageTargetsInWarehouse[self.packages[_epidx].xyLocation[1]][self.packages[_epidx].xyLocation[0]] = 0
        robot.carrying = -1
        # Clean up move/moving lists
        if _epkg in self.packagesMovingList:
            self.packagesMovingList.remove(_epkg)
        if _epkg in self.packagesMoveList:
            self.packagesMoveList.remove(_epkg)
        # Strip delivery tasks from robot queue
        for _tname in ('move to target dropoff location', 'drop off target package',
                       'move to target pickup location', 'pick up target package'):
            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, _tname)
    
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
            max_v *= ROBOT_CARRY_SPEED_PENALTY
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
            road_map=self.road_map if self.road_map is not None else None,
            traffic_ema=self.traffic_ema,
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
        
        if self.packagesMoveList:
            for packageNumber in self.packagesMoveList:
                if (packageNumber not in self.packagesMovingList):
                    packageIndex = self._find_package(packageNumber)
                    if packageIndex is not None:
                        packageArea = self.packages[packageIndex].area
                        packageAreaTarget = self.packages[packageIndex].areaTarget
                        packageStatus = self.packages[packageIndex].status
                        if (packageStatus == "move planned"):
                            packagePosition = self.packages[packageIndex].xyLocation
                            robotTaskQuantity = self.robotsTaskAssignmentList[0][1]
                            # Treat robots with only a "move to idle" task as available (0 real tasks)
                            _candidate_num = self.robotsTaskAssignmentList[0][0]
                            _cidx = self._find_robot(_candidate_num)
                            _candidate_r = self.robots[_cidx] if _cidx is not None else None
                            _effective_task_qty = robotTaskQuantity
                            if (_candidate_r and robotTaskQuantity == 1
                                    and _candidate_r.actionQueue
                                    and _candidate_r.actionQueue[0][1] == "move to idle"):
                                _effective_task_qty = 0
                            if (_effective_task_qty < self.robotsTaskAssignmentMaxQuantity):
                                robotNumberInList = _candidate_num
                                # Skip decommissioning robots
                                _cidx2 = self._find_robot(robotNumberInList)
                                _candidate_robot = self.robots[_cidx2] if _cidx2 is not None else None
                                if _candidate_robot and _candidate_robot.decommissioning:
                                    continue
                                robotIndex = self._find_robot(robotNumberInList)
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

            # ── Stall pause (pickup / dropoff / charger engage-disengage)
            if robot.stallTicks > 0:
                robot.stallTicks -= 1
                robot.velocity = 0.0
                robot.desiredVelocity = 0.0
                robot.movementProgress = 0.0
                continue

            _occupied.discard(tuple(robot.xyLocation))
            self._robot_consume_waypoints(i)

            # ── At destination? ────────────────────────────────────────
            if robot.xyLocation == robot.xyLocationTarget:
                robot.desiredVelocity = 0.0
                robot.velocity = max(0.0, robot.velocity - robot.decelerationRate)
                if robot.velocity < 0.01:
                    robot.velocity = 0.0
                robot.movementProgress = 0.0
                robot.blockedTicks = 0
            else:
                _moved = self._robot_step_movement(i, _occupied, _w, _h)
                if not _moved:
                    self._robot_recover_stuck(i, _occupied, _w, _h)

            # ── History ────────────────────────────────────────────────
            _hist = list(robot.recentLocations)
            _hist.append(tuple(robot.xyLocation))
            if len(_hist) > 8:
                _hist = _hist[-8:]
            robot.recentLocations = _hist

            self._robot_detect_oscillation(i)
            _occupied.add(tuple(robot.xyLocation))

        return self.robots

    def _robot_consume_waypoints(self, i):
        """Pop already-reached waypoints and set immediate movement target."""
        robot = self.robots[i]
        while robot.path:
            if (int(robot.path[0][0]) == int(robot.xyLocation[0])
                    and int(robot.path[0][1]) == int(robot.xyLocation[1])):
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

    def _robot_step_movement(self, i, _occupied, _w, _h):
        """Compute desired speed, attempt one movement step, handle collisions.

        Returns True if the robot successfully moved this tick.
        """
        robot = self.robots[i]
        _moved = False
        _cx = int(robot.xyLocation[0])
        _cy = int(robot.xyLocation[1])
        _tx = int(robot.xyLocationTarget[0])
        _ty = int(robot.xyLocationTarget[1])

        # Step direction: unit-clamped toward target cell
        _dx = max(-1, min(1, _tx - _cx))
        _dy = max(-1, min(1, _ty - _cy))
        _step_angle = (math.degrees(math.atan2(_dy, _dx))
                       if (_dx or _dy)
                       else float(robot.direction))

        # Heading (snap to step direction for display)
        robot.direction = round(_step_angle, 2)
        robot.targetDirection = robot.direction
        robot.degrees = round(robot.direction)
        robot.xyLocationDiff = [_dx, _dy]
        _cardinal = Functions.find_cardinal(robot.degrees)
        robot.cardinal = _cardinal
        robot.lastCardinal = _cardinal

        # Desired speed
        _desired = float(robot.maxVelocity)
        if robot.carrying != -1:
            _desired *= ROBOT_CARRY_SPEED_PENALTY

        # Tiered speed based on road tier and zone
        _road_tier = self.road_map[_cy, _cx]
        if _road_tier >= 2:           # collector / arterial: full speed
            pass
        elif _road_tier == 1:         # branch road: slight slowdown
            _desired *= 0.85
        elif self.zoneMap[_cy, _cx] == 0:  # neutral corridor, no road
            _desired *= 0.70
        else:                         # inside zone, no road (shelves)
            _desired *= 0.35

        # Battery limp mode
        _lt = self._limp_mode_batt_pct
        if robot.batteryPercent < _lt:
            _desired = min(
                _desired,
                0.25 + 0.75 * max(robot.batteryPercent, 0)
                / max(_lt, 1))

        # Approach braking
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

        # Velocity smoothing
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
                and 0 < robot.velocity < robot.minMovingVelocity - 1e-6):
            robot.velocity = robot.minMovingVelocity

        # Movement step
        _diagonal = (_dx != 0 and _dy != 0)
        _step_cost = 1.414 if _diagonal else 1.0
        robot.movementProgress += robot.velocity
        if robot.movementProgress >= _step_cost and (_dx or _dy):
            _nx, _ny = _cx + _dx, _cy + _dy
            _nc = (_nx, _ny)

            if not (0 <= _nx < _w and 0 <= _ny < _h):
                robot.movementProgress = 0.0
                robot.blockedTicks += 1
            elif _nc in _occupied:
                _blocker = None
                for _other in self.robots:
                    if (int(_other.xyLocation[0]) == _nx
                            and int(_other.xyLocation[1]) == _ny):
                        _blocker = _other
                        break
                _head_on = False
                if _blocker is not None:
                    _bdx = int(_blocker.xyLocationDiff[0])
                    _bdy = int(_blocker.xyLocationDiff[1])
                    if (_dx + _bdx == 0 and _dy + _bdy == 0
                            and (_bdx or _bdy)):
                        _head_on = True
                if _head_on and robot.robotNumber < _blocker.robotNumber:
                    _perps = [(-_dy, _dx), (_dy, -_dx)]
                    for _pdx, _pdy in _perps:
                        _sx, _sy = _cx + _pdx, _cy + _pdy
                        if (0 <= _sx < _w and 0 <= _sy < _h
                                and (_sx, _sy) not in _occupied):
                            robot.xyLocation[0] = _sx
                            robot.xyLocation[1] = _sy
                            _occupied.add((_sx, _sy))
                            robot.movementProgress -= _step_cost
                            robot.blockedTicks = 0
                            _moved = True
                            if robot.replanCooldownTicks <= 0:
                                _goal = (list(robot.actionQueue[0][0])
                                         if robot.actionQueue
                                         else list(robot.xyLocationTarget))
                                self._compute_robot_path(i, _goal, cooldown_ticks=4)
                            break
                    if not _moved:
                        robot.velocity = max(
                            0.0, robot.velocity - robot.decelerationRate)
                        robot.movementProgress = min(
                            robot.movementProgress, 0.35)
                        robot.blockedTicks += 1
                else:
                    robot.velocity = max(
                        0.0, robot.velocity - robot.decelerationRate)
                    robot.movementProgress = min(
                        robot.movementProgress, 0.35)
                    robot.blockedTicks += 1
            else:
                robot.xyLocation[0] = _nx
                robot.xyLocation[1] = _ny
                _occupied.add(_nc)
                robot.movementProgress -= _step_cost
                robot.blockedTicks = 0
                _moved = True

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
            robot.blockedTicks += 1

        return _moved

    def _robot_recover_stuck(self, i, _occupied, _w, _h):
        """Try to unstick a blocked robot via local sidestep, waypoint skip, or replan."""
        robot = self.robots[i]
        _cx = int(robot.xyLocation[0])
        _cy = int(robot.xyLocation[1])
        _tx = int(robot.xyLocationTarget[0])
        _ty = int(robot.xyLocationTarget[1])
        _bt = robot.blockedTicks
        _has_path = bool(robot.path)
        _moved = False

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

        # Local unstick: try any free neighbour closer to target
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

    def _robot_detect_oscillation(self, i):
        """Check recent location history for ABAB patterns and replan if needed."""
        robot = self.robots[i]
        _hist = list(robot.recentLocations)
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
    
    def update_robots_charging(self):
        for i in range(len(self.robots)):
            if not self.robots[i].actionQueue:
                continue
            status = self.robots[i].status
            if status == "move to charging station":
                self._robot_dock_charger(i)
            elif status == "charging":
                self._robot_charge_tick(i)
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers

    def _find_charger_at(self, location):
        """Return (charger, index) for the charger at *location*, or (None, -1)."""
        for ci, c in enumerate(self.chargers):
            if c.xyLocation == location:
                return c, ci
        return None, -1

    def _rebuild_entity_indices(self):
        """Rebuild O(1) lookup dicts for robots and packages."""
        self._robot_idx = {r.robotNumber: i for i, r in enumerate(self.robots)}
        self._pkg_idx = {p.packageNumber: i for i, p in enumerate(self.packages)}

    def _find_robot(self, robot_number):
        """Return index of robot with given robotNumber, or None.  O(1) amortised."""
        idx = getattr(self, '_robot_idx', {}).get(robot_number)
        if idx is not None and idx < len(self.robots) and self.robots[idx].robotNumber == robot_number:
            return idx
        self._robot_idx = {r.robotNumber: i for i, r in enumerate(self.robots)}
        return self._robot_idx.get(robot_number)

    def _find_package(self, package_number):
        """Return index of package with given packageNumber, or None.  O(1) amortised."""
        idx = getattr(self, '_pkg_idx', {}).get(package_number)
        if idx is not None and idx < len(self.packages) and self.packages[idx].packageNumber == package_number:
            return idx
        self._pkg_idx = {p.packageNumber: i for i, p in enumerate(self.packages)}
        return self._pkg_idx.get(package_number)

    def _set_package_idle(self, pkg, location=None):
        """Atomically reset a package to a clean idle state.

        Sets status, carrier, areaTarget, and target location together so a
        package can never end up 'idle' with a stale plan still attached
        (the 'zombie idle' state). ``location`` defaults to the package's
        current xyLocation; pass an override (e.g. dropping robot's cell)
        to relocate the package in the same operation.
        """
        try:
            pkg.status = 'idle'
            pkg.carrier = -1
            pkg.areaTarget = 'none'
            if location is not None:
                pkg.xyLocation = list(location)
            pkg.xyLocationTarget = list(pkg.xyLocation)
        except Exception:
            _log.exception('_set_package_idle failed for pkg#%s',
                           getattr(pkg, 'packageNumber', '?'))

    def _robot_dock_charger(self, i):
        """Handle a robot arriving at its target charging station."""
        robot = self.robots[i]
        target_loc = robot.actionQueue[0][0]
        charger, ci = self._find_charger_at(target_loc)
        if charger is None:
            return  # charger moved or removed; skip safely
        if robot.xyLocation != charger.xyLocation:
            return  # not there yet
        if not self._robot_is_stopped(robot):
            robot.desiredVelocity = 0.0
            return
        # Check if another robot is physically on the charger cell
        if any(ri != i and tuple(self.robots[ri].xyLocation) == tuple(charger.xyLocation)
               for ri in range(len(self.robots))):
            return  # occupied; wait
        if charger.status == "charging":
            # Another robot beat us; pop and let checkBattery re-assign next tick
            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "move to charging station")
        else:
            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "move to charging station", charger.xyLocation, "charging")
            self.chargers[ci].status = "charging"
            robot.stallTicks = STALL_TICKS
            robot.hasCharged = True

    def _robot_charge_tick(self, i):
        """Increment battery while docked; undock when full."""
        robot = self.robots[i]
        target_loc = robot.actionQueue[0][0]
        charger, ci = self._find_charger_at(target_loc)
        if charger is None:
            return  # charger moved or removed; skip safely
        if robot.xyLocation != charger.xyLocation:
            return
        if robot.batteryPercent <= BATTERY_FULL_PCT:
            robot.batteryPercent = round(robot.batteryPercent + robot.batteryChargingRate, 1)
        else:
            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "charging")
            self.chargers[ci].status = "idle"
            robot.stallTicks = STALL_TICKS

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
                            if currentLocation == packageLocation:
                                if (self.robots[i].status != "carried") and (self.packages[packageIndex].status != "carried"):
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "move to target pickup location", currentLocation, "pick up target package")
                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "move to target dropoff location"):
                        # Find the package this robot is carrying by robotNumber, not by cell
                        # (an idle package already sitting at the dropoff cell would fool a
                        # location-based search and block the carried→dropoff transition).
                        _carrying = self.robots[i].carrying
                        packageIndex = self._find_package(_carrying)
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
                    action = self.robots[i].actionQueue[0][1]
                    if action == "pick up target package" and self.robots[i].actionQueue[0][0] == currentLocation:
                        self._robot_do_pickup(i)
                    elif action == "drop off target package" and self.robots[i].actionQueue[0][0] == currentLocation:
                        self._robot_do_dropoff(i)
        
        return self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsTaskAssignmentList, self.robotsLog

    def _robot_do_pickup(self, i):
        """Execute package pickup: battery check, grab, set carrier, stall."""
        robot = self.robots[i]
        currentLocation = robot.xyLocation
        robotNumber = robot.robotNumber
        packageIndex = [y for y, x in enumerate(self.packages) if x.xyLocation == currentLocation]
        if not packageIndex:
            return
        packageIndex = packageIndex[0]
        pkg = self.packages[packageIndex]
        packageNumber = pkg.packageNumber
        packageLocationTarget = pkg.xyLocationTarget
        if robot.status == "carried" or pkg.status == "carried":
            return
        # Block pickup if battery too low — robot can't reliably deliver
        if robot.batteryPercent < self._work_min_batt_pct:
            self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "pick up target package")
            _log.info('pickup blocked: robot#%d batt=%.1f%% too low for pkg#%d',
                      robotNumber, robot.batteryPercent, packageNumber)
            return
        self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_replace_task(i, "pick up target package", packageLocationTarget, "move to target dropoff location")
        robot.carrying = packageNumber
        robot.stallTicks = STALL_TICKS
        robot.hasCarried = True
        pkg.status = "carried"
        pkg.carrier = robotNumber
        _log.debug('pickup: robot#%d picked pkg#%d at %s -> dropoff %s',
                   robotNumber, packageNumber, list(currentLocation), list(packageLocationTarget))

    def _robot_do_dropoff(self, i):
        """Execute package dropoff: release carrier, stall, clean up move lists."""
        robot = self.robots[i]
        currentLocation = robot.xyLocation
        _carrying = robot.carrying
        packageIndex = self._find_package(_carrying)
        if packageIndex is None:
            return
        pkg = self.packages[packageIndex]
        packageNumber = pkg.packageNumber
        if robot.status == "carried" or pkg.status != "carried":
            return
        self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_pop_task(i, "drop off target package")
        robot.carrying = -1
        robot.stallTicks = STALL_TICKS
        self.packagesMoveList, self.packagesMovingList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount = self.package_dropoff(packageNumber)
        _log.debug('dropoff: robot#%d dropped pkg#%d at %s  area=%s',
                   robot.robotNumber, packageNumber, list(currentLocation),
                   self.packages[self._find_package(packageNumber)].area if self._find_package(packageNumber) is not None else 'exported')
    
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
        # Tolerant list cleanup: remove from whichever lists contain the package
        if packageNumber in self.packagesMoveList:
            self.packagesMoveList.remove(packageNumber)
        if packageNumber in self.packagesMovingList:
            self.packagesMovingList.remove(packageNumber)
        # Update Planned Counts
        packageIndex = self._find_package(packageNumber)
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
        self.packages[packageIndex].deliveredAt = self._sim_time
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
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] -= 1
        if not self.robots[robotIndex].actionQueue:
            self.robots[robotIndex].status = "idle"
            self.robots[robotIndex].robotLog.append(Robot_Log('idle', self.datetimeNow))
            self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], 'idle', self.datetimeNow))
            # Decommissioning: queue exit task instead of staying idle
            if self.robots[robotIndex].decommissioning:
                self._queue_exit_task(robotIndex)
        else:
            robotNewStatus = self.robots[robotIndex].actionQueue[0][1]
            self.robots[robotIndex].status = robotNewStatus
            self.robots[robotIndex].robotLog.append(Robot_Log(robotNewStatus, self.datetimeNow))
            self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], robotNewStatus, self.datetimeNow))
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
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
        self.robots[robotIndex].actionQueue.insert(robotActionIndex, [robotNewLocation, robotNewAction])
        self.robotsTaskAssignmentList[robotTaskAssignmentIndex][1] += 1
        self.robots[robotIndex].status = robotNewAction
        self.robots[robotIndex].robotLog.append(Robot_Log(robotNewAction, self.datetimeNow))
        self.robotsLog.append(Robots_Log(self.robots[robotIndex].robotNumber, self.robots[robotIndex], robotNewAction, self.datetimeNow))
        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
        return self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog
