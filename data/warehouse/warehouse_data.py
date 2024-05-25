import numpy as np

class Warehouse_Data:
    def __init__(self,  
                        dataMaxLength, # Input variable
                        # Time
                        datetimeStamps = [],
                        timeElapseds = [],
                        # numPackages
                        numPackagesIdle = [], 
                        numPackagesMoving = [],
                        numPackagesImported = [], 
                        numPackagesInWarehouse = [], 
                        numPackagesInImport = [],
                        numPackagesInStorage = [],
                        numPackagesInExport = [],
                        numPackagesExported = [],
                        numPackagesInMoveList = [], 
                        numPackagesInMovingList = [], 
                        numPackagesMovedToStorage = [], 
                        numPackagesMovedToExport = [],
                        # listPackages
                        listPackagesImportedNames = [], 
                        # numChargers
                        numChargersIdle = [], 
                        numChargersCharging = [],
                        # numRobots
                        numRobotsImported = [], 
                        numRobotsInWarehouse = [], 
                        numRobotsExported = [],
                        numRobotsInImportArea = [], 
                        numRobotsInStorageArea = [], 
                        numRobotsInExportArea = [],
                        numRobotsIdle = [], 
                        numRobotsMoving = [], 
                        numRobotsCharging = [], 
                        # meanRobots
                        meanRobotsBatteryPercent = [], 
                        meanRobotsLenActionQueue = []
                ) -> None:
        self.dataMaxLength = dataMaxLength
        # Time
        self.datetimeStamps = datetimeStamps
        self.timeElapseds = timeElapseds
        # numPackages
        self.numPackagesIdle = numPackagesIdle
        self.numPackagesMoving = numPackagesMoving
        self.numPackagesImported = numPackagesImported
        self.numPackagesInWarehouse = numPackagesInWarehouse
        self.numPackagesInImport = numPackagesInImport
        self.numPackagesInStorage = numPackagesInStorage
        self.numPackagesInExport = numPackagesInExport
        self.numPackagesExported = numPackagesExported
        self.numPackagesInMoveList = numPackagesInMoveList
        self.numPackagesInMovingList = numPackagesInMovingList
        self.numPackagesMovedToStorage = numPackagesMovedToStorage
        self.numPackagesMovedToExport = numPackagesMovedToExport
        # listPackages
        self.listPackagesImportedNames = listPackagesImportedNames
        # numChargers
        self.numChargersIdle = numChargersIdle
        self.numChargersCharging = numChargersCharging
        # nuMRobots
        self.numRobotsImported = numRobotsImported
        self.numRobotsInWarehouse = numRobotsInWarehouse
        self.numRobotsExported = numRobotsExported
        self.numRobotsInImportArea = numRobotsInImportArea
        self.numRobotsInStorageArea = numRobotsInStorageArea
        self.numRobotsInExportArea = numRobotsInExportArea
        self.numRobotsIdle = numRobotsIdle
        self.numRobotsMoving = numRobotsMoving
        self.numRobotsCharging = numRobotsCharging
        # meanRobots
        self.meanRobotsBatteryPercent = meanRobotsBatteryPercent
        self.meanRobotsLenActionQueue = meanRobotsLenActionQueue

        # Init Warehouse_Data
        # Time
        self.datetimeStamps = [0] * self.dataMaxLength
        self.timeElapseds = np.zeros(self.dataMaxLength, dtype = "int")
        # numPackages
        self.numPackagesIdle = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesMoving = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesImported = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInWarehouse = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInImport = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInStorage = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInExport = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesExported = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInMoveList = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesInMovingList = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesMovedToStorage = np.zeros(self.dataMaxLength, dtype = "int")
        self.numPackagesMovedToExport = np.zeros(self.dataMaxLength, dtype = "int")
        # listPackages
        self.listPackagesImportedNames = np.zeros(self.dataMaxLength, dtype = "int")
        # numChargers
        self.numChargersIdle = np.zeros(self.dataMaxLength, dtype = "int")
        self.numChargersCharging = np.zeros(self.dataMaxLength, dtype = "int")
        # nuMRobots
        self.numRobotsImported = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsInWarehouse = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsExported = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsInImportArea = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsInStorageArea = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsInExportArea = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsIdle = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsMoving = np.zeros(self.dataMaxLength, dtype = "int")
        self.numRobotsCharging = np.zeros(self.dataMaxLength, dtype = "int")
        # meanRobots
        self.meanRobotsBatteryPercent = np.zeros(self.dataMaxLength, dtype = "int")
        self.meanRobotsLenActionQueue = np.zeros(self.dataMaxLength, dtype = "int")