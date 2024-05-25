from datetime import date, datetime, timedelta
import math
import random
import numpy as np
import cv2 
import time

from data.functions import *
from data.functions_timeseries import *
from data.warehouse.warehouse_init import *
from data.warehouse.warehouse_functions import *
from data.warehouse.package import *
from data.warehouse.package_functions import *
from data.warehouse.charger import *
from data.warehouse.charger_functions import *
from data.warehouse.robot import *
from data.warehouse.robot_functions import *
from data.warehouse.warehouse_data import *

class Warehouse:
    def __init__(self,
                    # Window Resolution, Window Center
                    windowRes = [], windowBackgroundColour = [], windowCenter = [], windowArray = [], itemsList = [], addressesList = [],
                    # Warehouse
                    warehouseLoopCount = 0, datetimeNow = datetime.now(), timeStart = time.time(), timeElapsed = 0, warehouseWindowRes = [], warehouseBackgroundColour = [], warehouseWindowCenter = [], warehouseWindowArray = [], warehousePerimeterCoordinates = [],
                    heightOfImportFloor = 0, heightOfStorageFloor = 0, heightOfExportFloor = 0,
                    colourOfImportAreas = (10,30,10), colourOfStorageAreas = (30,10,10), colourOfExportAreas = (10,10,30),
                    numberOfImportAreas = [2,2], numberOfStorageAreas = [2,2], numberOfExportAreas = [2,2],
                    sizeOfImportAreas = [], sizeOfStorageAreas = [], sizeOfExportAreas = [],
                    numberOfImportSlots = 100, numberOfStorageSlots = 100, numberOfExportSlots = 100,
                    importAreas = [], storageAreas = [], exportAreas = [],
                    numberOfLanes = 2, colourOfLanes = (20,20,20), space = 0, demispace = 0,
                    logsMaxLength = 10000, dataMaxLength = 200, warehouse_data = [],
                    # Space Availability
                    packagesInImportCount = 0, packagesInStorageCount = 0, packagesInExportCount = 0, packagesInWarehouseCount = 0,
                    packagesPlannedInImportCount = 0, packagesPlannedInStorageCount = 0, packagesPlannedInExportCount = 0,
                    importSpaceAvailable = True, storageSpaceAvailable = True, exportSpaceAvailable = True,
                    importNumTries = 1, storageNumTries = 5, exportNumTries = 1,
                    packageExportCount = 0, packageExportRollingCount = 0,
                    # Packages
                    packageDimensionsLimit = 1, packagesActionsList = ["idle", "carried"],
                    packages = [], packagesLog = [], packagesMaxQuantity = 2000,
                    packagesMoveList = [], packagesMaxMoveQuantity = 40, packagesRollingCount = 0,
                    packagesInWarehouse = [], packagesInImportAreas = [], packagesInStorageAreas = [], packagesInExportAreas = [], packageTargetsInWarehouse = [],
                    # Robots
                    packagesMovingList = [], robotsActionsList = ["none", "idle", "charging", "move to charging station", "move to target pickup location", "move to target dropoff location", "pickup target package", "dropoff target package"],
                    robots = [], robotsLog = [], robotsInWarehouse = [], robotsMaxQuantity = 40, robotsRollingCount = 0, robotsInWarehouseCount = 0,
                    robotsTaskAssignmentList = [], robotsTaskAssignmentStyle = 0, robotsTaskAssignmentMaxQuantity = 3,
                    numRobotsIdle = 0, numRobotsMoving = 0, numRobotsCharging = 0,
                    # Charging station(s)
                    chargersActionsList = ["none", "idle", "charging planned", "charging"], 
                    chargers = [], chargersMaxQuantity = 120, chargersInWarehouse = [], chargersRollingCount = 0
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
        # Floor and Areas
        self.heightOfImportFloor = heightOfImportFloor
        self.heightOfStorageFloor = heightOfStorageFloor
        self.heightOfExportFloor = heightOfExportFloor
        self.colourOfImportAreas = colourOfImportAreas
        self.colourOfStorageAreas = colourOfStorageAreas
        self.colourOfExportAreas = colourOfExportAreas
        self.numberOfImportAreas = numberOfImportAreas
        self.numberOfStorageAreas = numberOfStorageAreas
        self.numberOfExportAreas = numberOfExportAreas
        self.sizeOfImportAreas = sizeOfImportAreas
        self.sizeOfStorageAreas = sizeOfStorageAreas
        self.sizeOfExportAreas = sizeOfExportAreas
        self.numberOfImportSlots = numberOfImportSlots
        self.numberOfStorageSlots = numberOfStorageSlots
        self.numberOfExportSlots = numberOfExportSlots
        self.importAreas = importAreas
        self.storageAreas = storageAreas
        self.exportAreas = exportAreas
        self.numberOfLanes = numberOfLanes
        self.colourOfLanes = colourOfLanes
        self.space = space
        self.demispace = demispace
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
        self.importNumTries = importNumTries
        self.storageNumTries = storageNumTries
        self.exportNumTries = exportNumTries
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
        self.packagesInImportAreas = packagesInImportAreas
        self.packagesInStorageAreas = packagesInStorageAreas
        self.packagesInExportAreas = packagesInExportAreas
        self.packageTargetsInWarehouse = packageTargetsInWarehouse
        # Robots
        self.packagesMovingList = packagesMovingList
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
        
        # Init Warehouse Screen
        self.windowCenter = Functions.get_screencenter(self.windowRes)
        self.warehouseWindowArray = self.windowArray
        self.warehousePerimeterCoordinates, self.warehousePerimeterCoordinatesMinusOne = Warehouse_Init.init_warehousePerimeter(self.windowRes)
        self.warehouseWindowRes, self.warehouseWindowCenter, self.packagesInWarehouse, self.robotsInWarehouse, self.chargersInWarehouse, self.idleAreasInWarehouse, self.packageTargetsInWarehouse = Warehouse_Init.init_warehouseWindow(self.windowRes, self.windowCenter)
        self.space, self.demispace = Warehouse_Init.init_areasSpace(self.numberOfLanes, self.packageDimensionsLimit)
        self.heightOfImportFloor, self.heightOfStorageFloor, self.heightOfExportFloor = Warehouse_Init.init_floorsHeight(self.warehouseWindowRes)
        #self.sizeOfImportAreas, self.sizeOfStorageAreas, self.sizeOfExportAreas = Warehouse_Init.init_areasSize()
        #self.numberOfImportAreas, self.numberOfStorageAreas, self.numberOfExportAreas = Warehouse_Init.init_areasNumber(Warehouse_Init, self.heightOfImportFloor, self.heightOfStorageFloor, self.heightOfExportFloor, self.sizeOfImportAreas, self.sizeOfStorageAreas, self.sizeOfExportAreas, self.demispace, self.warehouseWindowRes)
        self.sizeOfImportAreas, self.sizeOfStorageAreas, self.sizeOfExportAreas, self.importAreas, self.storageAreas, self.exportAreas = Warehouse_Init.init_areas(Warehouse_Init, self.numberOfImportAreas, self.numberOfStorageAreas, self.numberOfExportAreas, self.heightOfImportFloor, self.heightOfStorageFloor, self.heightOfExportFloor, self.space, self.warehouseWindowRes, self.warehouseWindowCenter)
        self.packagesInImportAreas, self.packagesInStorageAreas, self.packagesInExportAreas, self.numberOfImportSlots, self.numberOfStorageSlots, self.numberOfExportSlots = Warehouse_Init.init_packagesInAreas(Warehouse_Init, self.numberOfImportAreas, self.packagesInImportAreas, self.sizeOfImportAreas, self.numberOfStorageAreas, self.packagesInStorageAreas, self.sizeOfStorageAreas,self.numberOfExportAreas, self.packagesInExportAreas, self.sizeOfExportAreas)
        self.warehouse_data = Warehouse_Data(self.dataMaxLength)
    
    def update_warehouse(self):
        #self.printDebugInfo()
        self.datetimeNow = datetime.now() # Update datetime
        self.timeElapsed = time.time() - self.timeStart
        self.packages, self.packagesRollingCount, self.packagesInWarehouseCount, self.packagesLog, self.packageTargetsInWarehouse, self.packageExportCount, self.packageExportRollingCount = self.update_packages() # Update packages
        self.chargers, self.chargersRollingCount, self.chargersInWarehouse = self.update_chargers()
        self.packages, self.packagesMoveList, self.packagesMovingList, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount, self.robots, self.robotsRollingCount, self.robotsTaskAssignmentList, self.robotsLog, self.chargers = self.update_robots() # Update robots
        self.packages = self.update_carried_packages()
        self.packages, self.packagesInImportCount, self.packagesInStorageCount, self.packagesInExportCount = self.update_packages_areas()
        self.packagesInWarehouse, self.packagesInImportAreas, self.packagesInStorageAreas, self.packagesInExportAreas = self.update_packages_in_warehouse() # Update warehouse knowledge of packages
        self.chargersInWarehouse = self.update_chargers_in_warehouse()
        self.robotsInWarehouse = self.update_robots_in_warehouse()
        self.importSpaceAvailable, self.storageSpaceAvailable, self.exportSpaceAvailable = self.update_space_available()
        self.packagesLog, self.robotsLog = self.trim_logs()
        self.record_warehouse_data()
        self.printDebugInfo()
        self.warehouseLoopCount += 1
        return self

    def update_space_available(self):
        # Planned Spaces
        if (self.packagesPlannedInImportCount >= self.numberOfImportSlots):
            self.importSpaceAvailable = False
        else:
            self.importSpaceAvailable = True
        if (self.packagesPlannedInStorageCount >= self.numberOfStorageSlots):
            self.storageSpaceAvailable = False
        else:
            self.storageSpaceAvailable = True
        if (self.packagesPlannedInExportCount >= self.numberOfExportSlots):
            self.exportSpaceAvailable = False
        else:
            self.exportSpaceAvailable = True
        # Actual Spaces
        if (self.packagesInImportCount >= self.numberOfImportSlots):
            self.importSpaceAvailable = False
        else:
            self.importSpaceAvailable = True
        if (self.packagesInStorageCount >= self.numberOfStorageSlots):
            self.storageSpaceAvailable = False
        else:
            self.storageSpaceAvailable = True
        if (self.packagesInExportCount >= self.numberOfExportSlots):
            self.exportSpaceAvailable = False
        else:
            self.exportSpaceAvailable = True
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
            packagesInImportLen = np.count_nonzero(self.packagesInImportAreas)
            packagesInStorageLen = np.count_nonzero(self.packagesInStorageAreas)
            packagesInExportLen = np.count_nonzero(self.packagesInExportAreas)
            print("- pInWarehouseCount: {}, pInWarehouse: {}".format(self.packagesInWarehouseCount, packagesInWarehouseLen))
            print("- pInImport: {}&{}, pInStorage: {}&{}, pInExport: {}&{}".format(packagesInImportLen, self.packagesInImportCount, packagesInStorageLen, self.packagesInStorageCount, packagesInExportLen, self.packagesInExportCount))
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
        print("- importSpaceAvailable: {}".format(self.importSpaceAvailable))
        if self.importSpaceAvailable:
            self.packagesRollingCount, self.packagesInImportCount, self.packagesInWarehouseCount, self.packages, self.packagesLog = Package_Functions.import_package(Package_Functions, self.importNumTries, self.importAreas, self.numberOfImportAreas, self.sizeOfImportAreas, self.packagesRollingCount, self.packagesInImportCount, self.packagesPlannedInImportCount, self.packagesInWarehouseCount, self.packagesInImportAreas, self.packagesInWarehouse, self.packageTargetsInWarehouse, self.packagesMaxQuantity, self.packages, self.packagesLog, self.itemsList, self.addressesList, self.datetimeNow) # Import package
        self.packages = self.update_packages_timeToDeadline(self.datetimeNow) # Update package timeToDeadline
        self.packages.sort(key=lambda x: x.deadline, reverse=False) # Sort packages by deadline
        self.packagesMoveList = self.sort_packagesMoveList_by_deadline()
        self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount = self.remove_immovables_from_packagesMoveList()
        self.packagesMovelist, self.packages = self.append_to_packagesMoveList() # Add to packagesMoveList based on deadline
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

    def remove_immovables_from_packagesMoveList(self):
        if not self.packagesMoveList:
            return self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount
        # In order of least importance, For packages in PackageMoveList from last pass but still not in PackagesMovingList, (e.g., if they are blocked by space limitation) remove the package from packagesMoveList
        for packageNumber in reversed(self.packagesMoveList):
            if (packageNumber not in self.packagesMovingList):
                packageIndex = [y for y, x in enumerate(self.packages) if x.packageNumber == packageNumber]
                if packageIndex:
                    packageIndex = packageIndex[0]
                    packageStatus = self.packages[packageIndex].status
                    packageArea = self.packages[packageIndex].area
                    packageAreaTarget = self.packages[packageIndex].areaTarget
                    packageLocation = self.packages[packageIndex].xyLocation
                    packageTargetLocation = self.packages[packageIndex].xyLocationTarget
                    removePackage = False
                    if (packageStatus == "idle"):
                        # Determine If removePackage
                        if (packageAreaTarget == "none"):
                            removePackage = True
                        elif (packageAreaTarget == "import") and (self.importSpaceAvailable == False):
                            removePackage = True
                        elif (packageAreaTarget == "storage") and (self.storageSpaceAvailable == False):
                            removePackage = True
                        elif (packageAreaTarget == "export") and (self.exportSpaceAvailable == False):
                            removePackage = True
                        # Remove Package
                        if (removePackage == True):
                            if (packageAreaTarget == "import"):
                                self.packagesPlannedInImportCount -= 1
                            elif (packageAreaTarget == "storage"):
                                self.packagesPlannedInStorageCount -= 1
                            elif (packageAreaTarget == "export"):
                                self.packagesPlannedInExportCount -= 1
                            self.packages[packageIndex].status = "idle"
                            self.packages[packageIndex].areaTarget = "none"
                            self.packagesMoveList.remove(packageNumber)
        return self.packagesMoveList, self.packages, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def append_to_packagesMoveList(self):
        # Add package to packagesMoveList if there is enough bandwidth
        if (len(self.packagesMoveList) < self.packagesMaxMoveQuantity):
            for i in range(len(self.packages)):
                packageNumber = self.packages[i].packageNumber
                packageStatus = self.packages[i].status
                packageArea = self.packages[i].area
                packageAreaTarget = self.packages[i].areaTarget
                # Add to packagesMoveList Accordingly
                addPackage = False
                if (packageStatus == "idle"):
                    if (packageNumber not in self.packagesMoveList) and (packageNumber not in self.packagesMovingList) and (packageArea != "export"):
                        if ((packageArea == "import") and ((self.storageSpaceAvailable == True) or (self.exportSpaceAvailable == True))):
                            addPackage = True
                        elif ((packageArea == "storage") and (self.exportSpaceAvailable == True)):
                            addPackage = True
                    if (addPackage == True):
                        self.packagesMoveList.append(packageNumber)
                # Cut packagesMoveList Loop if exceeded packagesMaxMoveQuantity
                if (len(self.packagesMoveList) >= self.packagesMaxMoveQuantity):
                    break
        return self.packagesMoveList, self.packages
    
    def update_packages_targetLocation(self):
        if not self.packagesMoveList:
            return self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount
        for i in range(len(self.packages)):
            packageNumber = self.packages[i].packageNumber
            packageStatus = self.packages[i].status
            packageArea = self.packages[i].area
            packageAreaTarget = self.packages[i].areaTarget
            packageAreaLocationTarget = self.packages[i].areaLocationTarget
            packageXYLocationTarget = self.packages[i].xyLocationTarget
            if (packageNumber in self.packagesMoveList):
                if (packageStatus == 'idle'):
                    movePackage = False
                    # Try to store package in Export
                    if (movePackage == False) and (self.exportSpaceAvailable == True):
                        if (packageArea != 'export') and (packageAreaTarget != 'export'):
                            movePackage, areaLocation, xyLocation = Package_Functions.try_packageTargetLocation(self.exportNumTries, self.exportAreas, self.numberOfExportAreas, self.sizeOfExportAreas, self.packagesInExportAreas, self.packagesInWarehouse, self.packageTargetsInWarehouse)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'export'
                                self.packagesPlannedInExportCount += 1
                    # Try to store package in Storage
                    if (movePackage == False) and (self.storageSpaceAvailable == True):
                        if (packageArea != "storage") and (packageAreaTarget != 'storage'):
                            movePackage, areaLocation, xyLocation = Package_Functions.try_packageTargetLocation(self.storageNumTries, self.storageAreas, self.numberOfStorageAreas, self.sizeOfStorageAreas, self.packagesInStorageAreas, self.packagesInWarehouse, self.packageTargetsInWarehouse)
                            if (movePackage == True):
                                self.packages[i].areaTarget = 'storage'
                                self.packagesPlannedInStorageCount += 1
                    # Move the package (or take no action)
                    if (movePackage == True):
                        self.packages[i].areaLocationTarget = areaLocation.copy()
                        self.packages[i].xyLocationTarget = xyLocation.copy()
                        self.packages[i].status = 'move planned'
                        self.packageTargetsInWarehouse[xyLocation[1]][xyLocation[0]] = 1
                        #print('-----------------------')
                        #print(self.packages[packageIndex].areaLocation)
                        #print(self.packages[packageIndex].areaLocationTarget)
                        #print(self.packages[packageIndex].xyLocation)
                        #print(self.packages[packageIndex].xyLocationTarget)
            #if (packageNumber not in self.packagesMoveList) and (movePackage == False):
            #    self.packages[i].areaTarget = packageArea
            #    self.packages[i].areaLocationTarget = packageAreaLocationTarget
            #    self.packages[i].xyLocationTarget = packageXYLocationTarget
            #    self.packages[i].status = "idle"
        return self.packages, self.packageTargetsInWarehouse, self.packagesPlannedInImportCount, self.packagesPlannedInStorageCount, self.packagesPlannedInExportCount

    def update_packageTargetsInWarehouse(self):
        if not self.packagesMoveList:
            return self.packageTargetsInWarehouse
        else:
            self.packageTargetsInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
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
        if not self.packages:
            return self.packages
        for i in range(len(self.packages)):
            if (0 <= self.packages[i].xyLocation[1] < self.heightOfImportFloor):
                if self.packages[i].area != "import":
                    if self.packages[i].area == "storage":
                        self.packagesInStorageCount -= 1
                        #self.packagesPlannedInStorageCount -= 1
                    elif self.packages[i].area == "export":
                        self.packagesInExportCount -= 1
                        #self.packagesPlannedInExportCount -= 1
                    self.packages[i].area = "import"
                    self.packagesInImportCount += 1
                    #self.packagesPlannedInImportCount += 1
                    #self.packages[i].colour[0] = 100
            elif (self.heightOfImportFloor <= self.packages[i].xyLocation[1] < (self.heightOfImportFloor + self.heightOfStorageFloor)):
                if self.packages[i].area != "storage":
                    if self.packages[i].area == "import":
                        self.packagesInImportCount -= 1
                        #self.packagesPlannedInImportCount -= 1
                    elif self.packages[i].area == "export":
                        self.packagesInExportCount -= 1
                        #self.packagesPlannedInExportCount -= 1
                    self.packages[i].area = "storage"
                    self.packagesInStorageCount += 1
                    #self.packagesPlannedInStorageCount += 1
                    #self.packages[i].colour[0] = 150
            elif ((self.heightOfImportFloor + self.heightOfStorageFloor) <= self.packages[i].xyLocation[1] < (self.heightOfImportFloor + self.heightOfStorageFloor + self.heightOfExportFloor)):
                if self.packages[i].area != "export":
                    if self.packages[i].area == "import":
                        self.packagesInImportCount -= 1
                        #self.packagesPlannedInImportCount -= 1
                    elif self.packages[i].area == "storage":
                        self.packagesInStorageCount -= 1
                        #self.packagesPlannedInStorageCount -= 1
                    self.packages[i].area = "export"
                    self.packagesInExportCount += 1
                    #self.packagesPlannedInExportCount += 1
                    #self.packages[i].colour[0] = 200
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
        i = 0
        while 1:
            if (not self.packages):
                break
            try:
                packageStatus = self.packages[i].status
            except:
                break
            packageTimeToDeadline = self.packages[i].timeToDeadline
            packageArea = self.packages[i].area
            packageAreaTarget = self.packages[i].areaTarget
            packageLocation = self.packages[i].xyLocation
            packageTargetLocation = self.packages[i].xyLocationTarget
            if (packageArea == "export") and (packageTimeToDeadline < timedelta(seconds=0)) and (packageStatus == "idle"):
                #print("- packageTimeToDeadline: {}".format(packageTimeToDeadline))
                self.packages.pop(i)
                self.packageExportCount += 1
                self.packageExportRollingCount += 1
                self.packagesInWarehouseCount -= 1
                self.packagesInExportCount -= 1
                i -= 1
            i += 1
        return self.packages, self.packageExportCount, self.packageExportRollingCount

    # Update Warehouse
    def update_packages_in_warehouse(self):
        self.packagesInWarehouse = np.zeros((self.warehouseWindowRes[1], self.warehouseWindowRes[0]), dtype = 'uint8')
        self.packagesInImportAreas = np.zeros((self.numberOfImportAreas[0], self.numberOfImportAreas[1], self.sizeOfImportAreas[0]*self.sizeOfImportAreas[1]), dtype = 'uint8')
        self.packagesInStorageAreas = np.zeros((self.numberOfStorageAreas[0], self.numberOfStorageAreas[1], self.sizeOfStorageAreas[0]*self.sizeOfStorageAreas[1]), dtype = 'uint8')
        self.packagesInExportAreas = np.zeros((self.numberOfExportAreas[0], self.numberOfExportAreas[1], self.sizeOfExportAreas[0]*self.sizeOfExportAreas[1]), dtype = 'uint8')
        for i in range(len(self.packages)):
            packageXY = self.packages[i].xyLocation
            self.packagesInWarehouse[packageXY[1]][packageXY[0]] = 1
            if (self.packages[i].area == "import"):
                xyLocationWithinAreaBool, areasRowCol, areaIndex = Warehouse_Functions.convert_xyWarehouse_to_xyArea(Warehouse_Functions, self.importAreas, self.numberOfImportAreas, self.packages[i].xyLocation)
                if (xyLocationWithinAreaBool == True):
                    self.packagesInImportAreas[areasRowCol[0]][areasRowCol[1]][areaIndex] = 1
                    #print("packagesInImportAreas: {}".format(self.packagesInImportAreas[areasRowCol[0]][areasRowCol[1]]))
            elif (self.packages[i].area == "storage"):
                xyLocationWithinAreaBool, areasRowCol, areaIndex = Warehouse_Functions.convert_xyWarehouse_to_xyArea(Warehouse_Functions, self.storageAreas, self.numberOfStorageAreas, self.packages[i].xyLocation)
                if (xyLocationWithinAreaBool == True):
                    self.packagesInStorageAreas[areasRowCol[0]][areasRowCol[1]][areaIndex] = 1
                    #print("packagesInStorageAreas: {}".format(self.packagesInStorageAreas[areasRowCol[0]][areasRowCol[1]]))
            elif (self.packages[i].area == "export"):
                xyLocationWithinAreaBool, areasRowCol, areaIndex = Warehouse_Functions.convert_xyWarehouse_to_xyArea(Warehouse_Functions, self.exportAreas, self.numberOfExportAreas, self.packages[i].xyLocation)
                if (xyLocationWithinAreaBool == True):
                    self.packagesInExportAreas[areasRowCol[0]][areasRowCol[1]][areaIndex] = 1
                    #print("packagesInExportAreas: {}".format(self.packagesInExportAreas[areasRowCol[0]][areasRowCol[1]]))
        return self.packagesInWarehouse, self.packagesInImportAreas, self.packagesInStorageAreas, self.packagesInExportAreas
    
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
        self.robotsRollingCount, self.robotsInWarehouseCount, self.robots, self.robotsTaskAssignmentList = Robot_Functions.import_robot(Robot_Functions, self.warehousePerimeterCoordinates, self.robotsRollingCount, self.robotsInWarehouseCount, self.robotsMaxQuantity, self.robotsInWarehouse, self.robots, self.robotsTaskAssignmentList, self.chargersInWarehouse) # Import Robot
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
        self.chargersRollingCount, self.chargers = Charger_Functions.import_charger(Charger_Functions, self.warehousePerimeterCoordinatesMinusOne, self.chargersRollingCount, self.chargersMaxQuantity, self.chargersInWarehouse, self.chargers)
        return self.chargers, self.chargersRollingCount, self.chargersInWarehouse

    def update_robots_batteryPercent(self):
        for i in range(len(self.robots)):
            self.robots[i].batteryPercent -= self.robots[i].batteryDepletingRate
            self.robots[i].batteryPercent = np.round(self.robots[i].batteryPercent,2)
        return self.robots
    
    def update_robots_checkBattery(self):
        for i in range(len(self.robots)):
            if self.robots[i].batteryPercent <= 20:
                # ************* Make smarter logic for deciding to recharge or other
                if self.robots[i].status != "charging":
                    if self.robots[i].actionQueue:
                        robotQueuedCharging = [x for x in self.robots[i].actionQueue if "move to charging station" in x]
                        if not robotQueuedCharging:
                            chargerAvailable, c = self.find_available_charger()
                            if (chargerAvailable == True):
                                chargingStationLocation = self.chargers[c].xyLocation
                                #print('- chargingStationLocation: {}'.format(chargingStationLocation))
                                if ("dropoff" not in self.robots[i].status):
                                    self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, chargingStationLocation, "move to charging station")
                                    self.chargers[c].status = "charging planned"
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
                        chargerAvailable, c = self.find_available_charger()
                        if (chargerAvailable == True):
                            chargingStationLocation = self.chargers[c].xyLocation
                            if ("dropoff" not in self.robots[i].status):
                                self.robots[i], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(i, 0, chargingStationLocation, "move to charging station")
                                self.chargers[c].status = "charging planned"
                                self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                            #print('- Robot #{}: Battery low'.format(i))
                            #print('- robotTaskAssignmentIndex: {}'.format(robotTaskAssignmentIndex))
            if self.robots[i].batteryPercent <= 5:
                # ************* ENTER SLEEP MODE, NEED TO BE PICKED UP BY BOT **********
                pass
            self.robots[i].batteryPercent = Functions.ensure_limit_1d(self.robots[i].batteryPercent, 0, 100)
        
        #print(self.robotsTaskAssignmentList[0][0])
        #print(i)
        return self.robots, self.robotsTaskAssignmentList, self.robotsLog, self.chargers, self.packagesMoveList
    
    def find_available_charger(self):
        for c in range(len(self.chargers)):
            if (self.chargers[c].status == "idle"):
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
                                #print('- self.robotsTaskAssignmentList: {}'.format(self.robotsTaskAssignmentList))
                                #print('- self.robots[{}].actionQueue: {}'.format(robotNumber, self.robots[robotNumber].actionQueue))
                                if self.robots[robotIndex].actionQueue:
                                    robotQueuedCharging = [x for x in self.robots[robotIndex].actionQueue if "move to charging station" in x]
                                    if not robotQueuedCharging:
                                        self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
                                        self.packagesMovingList.append(packageNumber)
                                        self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks
                                else:
                                    self.robots[robotIndex], self.robotsTaskAssignmentList, self.robotsLog = self.robot_insert_task(robotIndex, robotTaskQuantity, packagePosition, "move to target pickup location")
                                    self.packagesMovingList.append(packageNumber)
                                    self.robotsTaskAssignmentList.sort(key=lambda y: y[1], reverse=False) # Sort robotTasksAssignmentList by #tasks

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
                    chargingStation = [x for x in self.chargers if x.xyLocation == robotTargetLocation][0]
                    chargingStationNumber = chargingStation.chargerNumber
                    chargingStationLocation = chargingStation.xyLocation
                    chargingStationIndex = [i for i, x in enumerate(self.chargers) if x.chargerNumber == chargingStationNumber][0]
                    #print('- chargingStationLocation: {}'.format(chargingStationLocation))
                    if (self.robots[i].xyLocation == chargingStationLocation): # Just arriving at charging station
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
                    chargingStation = [x for x in self.chargers if x.xyLocation == robotTargetLocation][0]
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
                                #print('- self.packagesMoveList: {}'.format(str(self.packagesMoveList)))
                                #print('- self.packagesMovingList: {}'.format(str(self.packagesMovingList)))
                                #print('- self.robotsTaskAssignmentList: {}'.format(str(self.robotsTaskAssignmentList)))
                                #print('- packageNumber: {}'.format(str(packageNumber)))
                                #print('- robotNumber: {}'.format(str(self.robots[i].robotNumber)))
                                #print('- actionQueue: {}'.format(str(self.robots[i].actionQueue)))

                    elif (self.robots[i].actionQueue[0][0] == currentLocation) and (self.robots[i].actionQueue[0][1] == "drop off target package"):
                        packageIndex = [y for y, x in enumerate(self.packages) if x.xyLocationTarget == currentLocation]
                        if packageIndex:
                            packageIndex = packageIndex[0]
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

    def find_robots_at_this_location(self, xyLocation):
        
        return
    
    def find_packages_at_this_location(self, xyLocation):
        
        return

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
        except:
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
            time.sleep(10)
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
        robotActionIndex = [i2 for i2, x in enumerate(self.robots[robotIndex].actionQueue) if robotActionToPop in x][0]
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
        robotActionIndex = [i2 for i2, x in enumerate(self.robots[robotIndex].actionQueue) if robotActionToPop in x][0]
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
