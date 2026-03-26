from datetime import date, datetime, timedelta
import math
import random
import logging
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
                    packagesMovingList = [], robotsActionsList = ["none", "idle", "charging", "move to charging station", "move to target pickup location", "move to target dropoff location", "pickup target package", "dropoff target package"],
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
        # Adaptive power management state (recomputed every tick)
        self._charger_pressure = 0.0    # ratio of busy chargers (0.0-1.0)
        self._avg_fleet_battery = 100.0
        self._charge_threshold = 30     # dynamic, recomputed in checkBattery

        # ── Flow-control metrics ──
        self._flow_export_times = []     # recent delivery durations (seconds)
        self._flow_avg_delivery = 0.0    # EMA of delivery time (seconds)
        self._flow_avg_deadline = 0.0    # EMA of time-to-deadline at spawn (seconds)
        self._flow_throughput   = 0.0    # exports per second (EMA)
        self._flow_last_export_t = time.time()
        self._flow_import_cap   = int(self.packagesMaxQuantity * 0.5)  # start at target occupancy
        self._flow_ema_alpha    = 0.02   # smoothing factor for EMAs
        self._flow_target_occ   = 0.50   # target warehouse occupancy (50%)
    
    def update_warehouse(self):
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
                'robots: idle=%d charging=%d  avg_drain=%.3f  pressure=%.2f  thresh=%.0f%%  '
                'flow: cap=%d/%d  delivery=%.0fs  throughput=%.2f/s',
                self.warehouseLoopCount,
                len(self.packages), self.packagesInImportCount,
                self.packagesInStorageCount, self.packagesInExportCount,
                len(self.packagesMoveList), len(self.packagesMovingList),
                _n_overdue, _n_carried, _n_planned,
                self.packageExportRollingCount,
                _n_idle_robots, _n_charging, _avg_drain,
                self._charger_pressure, self._charge_threshold,
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

    def update_flow_control(self):
        """Compute adaptive import cap targeting ~50%% warehouse occupancy.

        Base target = packagesMaxQuantity * target_occupancy (0.50).
        Adjustments:
          - Overdue penalty: reduce cap when packages are late
          - Delivery-stress penalty: reduce cap when deliveries are slow vs deadlines
          - Idle-robot bonus: raise cap slightly when robots have spare capacity
          - Occupancy error: proportional correction toward the target occupancy
        """
        n_pkgs   = len(self.packages)
        n_robots = len(self.robots)
        max_cap  = self.packagesMaxQuantity
        if n_robots == 0 or max_cap == 0:
            self._flow_import_cap = max_cap
            return

        base_target = max_cap * self._flow_target_occ

        # ── Signals ──────────────────────────────────────────
        n_overdue = sum(1 for p in self.packages if p.timeToDeadline.total_seconds() < 0)
        overdue_ratio = n_overdue / max(n_pkgs, 1)

        n_idle_robots = sum(1 for r in self.robots if r.status == 'idle')
        idle_robot_ratio = n_idle_robots / n_robots

        current_occ = n_pkgs / max_cap
        occ_error = current_occ - self._flow_target_occ  # positive = above target

        if self._flow_avg_deadline > 0 and self._flow_avg_delivery > 0:
            delivery_stress = self._flow_avg_delivery / self._flow_avg_deadline
        else:
            delivery_stress = 0.0

        # ── Adjustments ──────────────────────────────────────
        # Overdue penalty: up to 50% reduction at heavy overdue
        overdue_penalty = min(overdue_ratio * 2.5, 0.50)

        # Delivery-stress penalty: penalise when avg delivery > 60% of deadline window
        stress_penalty = max(0.0, (delivery_stress - 0.6)) * 0.5

        # Occupancy proportional correction: nudge cap when occupancy drifts from target
        # +10% above target → 20% reduction; −10% below target → 20% increase
        occ_correction = occ_error * 2.0

        # Idle-robot bonus: raise cap when robots are idle and no overdue pressure
        idle_bonus = 0.0
        if overdue_ratio < 0.05 and idle_robot_ratio > 0.3:
            idle_bonus = min((idle_robot_ratio - 0.3) * 0.4, 0.20)

        # ── Combine into multiplier ──────────────────────────
        adj = 1.0 - overdue_penalty - stress_penalty - occ_correction + idle_bonus
        adj = max(0.20, min(1.30, adj))  # clamp

        target_cap = int(base_target * adj)
        target_cap = max(n_robots, min(max_cap, target_cap))

        # Smooth: move 5% toward target each tick to avoid oscillation
        self._flow_import_cap = int(self._flow_import_cap + 0.05 * (target_cap - self._flow_import_cap))
        self._flow_import_cap = max(n_robots, min(max_cap, self._flow_import_cap))

        # Track average deadline window from recently spawned packages
        recent = [p for p in self.packages if (time.time() - p.createdAt) < 2.0]
        if recent:
            avg_dl = sum((p.deadline - datetime.fromtimestamp(p.createdAt)).total_seconds() for p in recent) / len(recent)
            a = self._flow_ema_alpha
            self._flow_avg_deadline = a * avg_dl + (1 - a) * self._flow_avg_deadline if self._flow_avg_deadline > 0 else avg_dl

    def invalidate_stale_targets(self):
        """Cancel in-flight moves whose target cells no longer match the intended zone."""
        expected_zone = {'import': ZONE_IMPORT, 'storage': ZONE_STORAGE, 'export': ZONE_EXPORT}
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
        from collections import deque
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
        from collections import deque
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
                        # Remove Package
                        if (removePackage == True):
                            if (packageAreaTarget == "import"):
                                self.packagesPlannedInImportCount -= 1
                            elif (packageAreaTarget == "storage"):
                                self.packagesPlannedInStorageCount -= 1
                            elif (packageAreaTarget == "export"):
                                self.packagesPlannedInExportCount -= 1
                            if packageAreaTarget in evicted:
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
        # Dynamic planning depth: only plan as many moves as robots can realistically
        # execute soon.  available_robots * 2 gives a one-task pipeline buffer.
        available_robots = sum(1 for r in self.robots
                               if r.status not in ('charging', 'move to charging station')
                               and r.batteryPercent >= 10)
        effective_move_cap = max(available_robots * 2, 4)  # floor of 4 to avoid stalling
        effective_move_cap = min(effective_move_cap, self.packagesMaxMoveQuantity)

        # Headroom = genuinely available destination slots this tick.
        # Cap new moveList entries to total_headroom so we don't queue more
        # packages than there are slots to assign targets to this tick.
        export_headroom  = max(0, self.numberOfExportSlots  - self.packagesInExportCount  - self.packagesPlannedInExportCount)
        storage_headroom = max(0, self.numberOfStorageSlots - self.packagesInStorageCount - self.packagesPlannedInStorageCount)
        total_headroom   = export_headroom + storage_headroom
        new_entries      = 0

        # Drain reorg list first — displaced packages get front-of-queue priority.
        # Reorg entries reclaim already-planned slot space so they don't consume
        # from new_entries (which caps against total_headroom for net-new work).
        for pkg_num in list(self.packagesReorgList):
            pkg_idx = next((i for i, p in enumerate(self.packages)
                            if p.packageNumber == pkg_num), None)
            if pkg_idx is not None:
                p = self.packages[pkg_idx]
                if (p.status == 'idle' and p.area != 'export'
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
                if (packageStatus == "idle"):
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
                    # Try to store package in Export (only if urgent)
                    if (movePackage == False) and (export_headroom > 0) and is_urgent:
                        if (packageArea != 'export') and (packageAreaTarget != 'export'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_EXPORT, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'export'
                                self.packagesPlannedInExportCount += 1
                                export_headroom -= 1
                    # Try to store package in Storage
                    if (movePackage == False) and (storage_headroom > 0):
                        if (packageArea != "storage") and (packageAreaTarget != 'storage'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_STORAGE, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'storage'
                                self.packagesPlannedInStorageCount += 1
                                storage_headroom -= 1
                    # Fallback: non-urgent package but storage full — try export anyway
                    if (movePackage == False) and (export_headroom > 0) and not is_urgent:
                        if (packageArea != 'export') and (packageAreaTarget != 'export'):
                            movePackage, xyLocation = Package_Functions.try_packageTargetLocation(self.zoneMap, ZONE_EXPORT, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.chargersInWarehouse)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'export'
                                self.packagesPlannedInExportCount += 1
                                export_headroom -= 1
                    # Move the package (or take no action)
                    if (movePackage == True):
                        self.packages[i].xyLocationTarget = xyLocation.copy()
                        self.packages[i].status = 'move planned'
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
                        self.packageTargetsInWarehouse, self.chargersInWarehouse)
                    if movePackage:
                        p.areaTarget = 'export'
                        p.xyLocationTarget = xyLocation.copy()
                        p.status = 'move planned'
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
                robotIndex = [y for y, x in enumerate(self.robots) if x.robotNumber == robotNumber][0]
                #print("- robotNumber: {}".format(robotNumber))
                #print("- robotIndex: {}".format(robotIndex))
                #print("- packageCarrier: {}".format(self.packages[i].carrier))
                #print("- robotLocation: {}".format(self.robots[robotIndex].xyLocation))
                #print("- packageLocation: {}".format(self.packages[i].xyLocation))
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
        for i in range(len(self.packages)):
            if (self.packages[i].deadline < datetimeNow) and (self.packages[i].packageNumber not in self.packagesMoveList):
                self.packages[i].colour = [80, 20, 255]
            elif (self.packages[i].deadline < datetimeNow) and (self.packages[i].packageNumber in self.packagesMoveList):
                self.packages[i].colour = [80, 120, 255]
            elif (self.packages[i].deadline >= datetimeNow) and (self.packages[i].packageNumber not in self.packagesMoveList):
                self.packages[i].colour = [150, 150, 150]
            elif (self.packages[i].deadline >= datetimeNow) and (self.packages[i].packageNumber in self.packagesMoveList):
                self.packages[i].colour = [120, 220, 120]
            if (self.packages[i].area == "export") and (timedelta(seconds=10) < self.packages[i].timeToDeadline < timedelta(seconds=60)) and (self.packages[i].status == "idle"):
                self.packages[i].colour = [240,120,120]
            elif (self.packages[i].area == "export") and (self.packages[i].timeToDeadline < timedelta(seconds=10)) and (self.packages[i].status == "idle"):
                self.packages[i].colour = [240,180,120]
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
            if canExport and (packageTimeToDeadline < timedelta(seconds=0)) and (packageStatus == "idle"):
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
        # Fleet-aware eco factor: when chargers are scarce, robots conserve energy
        # Pressure > 0.5 starts reducing drain; at pressure 1.0, drain cut by 30%
        _eco = 1.0 - 0.3 * max(0.0, self._charger_pressure - 0.5) * 2.0
        for i in range(len(self.robots)):
            r = self.robots[i]
            if r.status == 'charging':
                mult = DRAIN_CHARGING
            elif r.xyLocation != r.xyLocationTarget:
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
        # --- Adaptive power metrics ---
        _n_chargers = max(len(self.chargers), 1)
        _n_busy = sum(1 for c in self.chargers if c.status != 'idle')
        self._charger_pressure = _n_busy / _n_chargers
        self._avg_fleet_battery = (sum(r.batteryPercent for r in self.robots)
                                   / max(len(self.robots), 1))
        # Dynamic charge threshold: 25% base, scales up with charger pressure
        # At pressure 0: 25% | 0.5: 37.5% | 1.0: 50%
        self._charge_threshold = 25 + 25 * self._charger_pressure
        # --- End adaptive metrics ---
        for i in range(len(self.robots)):
            if self.robots[i].batteryPercent <= self._charge_threshold:
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
                    self.chargers[c].status = "charging planned"
                    self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False)
                    _log.debug('opportunistic top-off: robot#%d batt=%.1f%% pressure=%.2f -> charger %s',
                               self.robots[i].robotNumber, self.robots[i].batteryPercent,
                               self._charger_pressure, _csl)
            if self.robots[i].batteryPercent <= 5:
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
        for c in range(len(self.chargers)):
            loc = tuple(self.chargers[c].xyLocation)
            if self.chargers[c].status == "idle" and loc not in claimed and loc not in occupied:
                return True, c
        return False, -1

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
                            if (robotTaskQuantity < self.robotsTaskAssignmentMaxQuantity):
                                robotNumberInList = self.robotsTaskAssignmentList[0][0]
                                robotIndex = [y for y, x in enumerate(self.robots) if x.robotNumber == robotNumberInList][0]
                                # Battery saving mode: don't assign new work to robots below 10%
                                if self.robots[robotIndex].batteryPercent < 10:
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
                                if self.robots[robotIndex].actionQueue:
                                    robotQueuedCharging = [x for x in self.robots[robotIndex].actionQueue if "move to charging station" in x]
                                    if not robotQueuedCharging:
                                        self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
                                        self.packagesMovingList.append(packageNumber)
                                        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                        _log.debug('task assign: robot#%d -> pickup pkg#%d at %s',
                                                   self.robots[robotIndex].robotNumber, packageNumber, packagePosition)
                                else:
                                    self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
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
        for i in range(len(self.robots)):
            if self.robots[i].actionQueue:
                currentLocation = self.robots[i].xyLocation
                currentTargetLocation = self.robots[i].xyLocationTarget
                actualTargetLocation = self.robots[i].actionQueue[0][0]
                if currentTargetLocation != actualTargetLocation:
                    #print('- self.robots[{}].xyLocation: {}'.format(self.robots[i].robotNumber, self.robots[i].xyLocation))
                    #print('- self.robots[{}].xyLocationTarget: {}'.format(self.robots[i].robotNumber, self.robots[i].xyLocationTarget))
                    self.robots[i].xyLocationTarget = actualTargetLocation
                    #print('- self.robots[{}].xyLocationTarget: {}'.format(self.robots[i].robotNumber, self.robots[i].xyLocationTarget))
        return self.robots

    def update_robots_xyLocation(self):
        for i in range(len(self.robots)):
            currentLocation = self.robots[i].xyLocation
            targetLocation = self.robots[i].xyLocationTarget
            if (currentLocation != targetLocation):
                # Battery saving mode: skip every other movement tick at <10%
                if self.robots[i].batteryPercent < 10:
                    if not getattr(self.robots[i], '_lowbatt_skip', False):
                        self.robots[i]._lowbatt_skip = True
                        continue
                    else:
                        self.robots[i]._lowbatt_skip = False
                #print('- robot#{}: currentLoc: {}. targetLoc: {}'.format(i, currentLocation, targetLocation))
                diffX = self.robots[i].xyLocationTarget[0] - self.robots[i].xyLocation[0]
                diffY = self.robots[i].xyLocationTarget[1] - self.robots[i].xyLocation[1]
                self.robots[i].xyLocationDiff = [diffX, diffY]
                #print('- robot#{}: diffLocation: {}'.format(i, self.robots[i].xyLocationDiff))
                degrees = round(math.degrees(math.atan2(diffY, diffX)))
                #print('- degrees: {}'.format(degrees))
                self.robots[i].degrees = degrees
                cardinal = Functions.find_cardinal(degrees)
                self.robots[i].cardinal = cardinal
                #print('- cardinal: {}'.format(self.robots[i].cardinal))
                xyMovementTemp = Functions.find_location_from_cardinal(cardinal)
                self.robots[i].xyLocation[0] += xyMovementTemp[0]
                self.robots[i].xyLocation[1] += xyMovementTemp[1]
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
                            # Move robot to nearest neutral-zone cell (no zone) to idle
                            if not self.robots[i].actionQueue:
                                _idle_target = self._find_nearest_neutral_cell(
                                    self.robots[i].xyLocation[0], self.robots[i].xyLocation[1])
                                if _idle_target:
                                    self.robots[i].xyLocationTarget = list(_idle_target)
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
                    robotNumber = self.robots[i].robotNumber
                    if (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "pick up target package"):
                        packageIndex = [y for y, x in enumerate(self.packages) if x.xyLocation == currentLocation]
                        if packageIndex:
                            packageIndex = packageIndex[0]
                            packageNumber = self.packages[packageIndex].packageNumber
                            packageLocationTarget = self.packages[packageIndex].xyLocationTarget
                            if (self.robots[i].status != "carried") and (self.packages[packageIndex].status != "carried"):
                                # Block pickup if battery too low — robot can't reliably deliver
                                if self.robots[i].batteryPercent < 10:
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
