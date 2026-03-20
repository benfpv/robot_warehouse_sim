import random
from datetime import datetime, timedelta
import numpy as np

from data.warehouse.warehouse_log import *
from data.warehouse.package import *

# Zone constants (must match warehouse.py)
ZONE_NONE = 0
ZONE_IMPORT = 1
ZONE_STORAGE = 2
ZONE_EXPORT = 3


class Package_Functions:
    """Helpers for spawning and positioning packages."""

    def import_package(self, zoneMap, chargersInWarehouse, packagesRollingCount, packagesInImportCount, packagesInWarehouseCount, packagesInWarehouse, packageTargetsInWarehouse, packagesMaxQuantity, packages, packagesLog, itemsList, addressesList, datetimeNow, spawnZoneId=ZONE_IMPORT):
        """Attempt to spawn one new package in the given zone if capacity allows."""
        if packagesInWarehouseCount < packagesMaxQuantity:
            movePackage, xyLocation = self.try_packageTargetLocation(zoneMap, spawnZoneId, packagesInWarehouse, packageTargetsInWarehouse, chargersInWarehouse)
            if (movePackage == True):
                zoneNames = {ZONE_IMPORT: 'import', ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}
                spawnArea = zoneNames.get(spawnZoneId, 'neutral')
                package = self.generate_package(xyLocation, itemsList, addressesList, packagesRollingCount, area=spawnArea)
                packages.append(package)
                packagesLog.append(Packages_Log(packagesRollingCount, package, 'import', datetimeNow))
                packagesRollingCount += 1
                packagesInImportCount += 1
                packagesInWarehouseCount += 1
        return packagesRollingCount, packagesInImportCount, packagesInWarehouseCount, packages, packagesLog

    @staticmethod
    def try_packageTargetLocation(zoneMap, zoneId, packagesInWarehouse, packageTargetsInWarehouse, chargersInWarehouse):
        """Find a free cell in *zoneId* that has no package, planned target, or charger.

        Returns (True, [x, y]) on success or (False, []) if the zone is full.
        """
        candidates = np.argwhere(
            (zoneMap == zoneId) &
            (packagesInWarehouse == 0) &
            (packageTargetsInWarehouse == 0) &
            (chargersInWarehouse == 0)
        )
        if len(candidates) == 0:
            return False, []
        idx = random.randint(0, len(candidates) - 1)
        y, x = candidates[idx]
        return True, [int(x), int(y)]

    @staticmethod
    def generate_package(xyLocation, itemsList, addressesList, packageRollingCount, area='import'):
        """Construct a new Package with randomised item, addresses, and deadline."""
        itemName = random.sample(sorted(itemsList), 1)[0]
        packageLog = []
        #print('- itemName: ' + str(itemName))
        itemValues = itemsList[itemName]
        #print('- itemValues: ' + str(itemValues))
        addressFrom = random.sample(sorted(addressesList), 1)[0]
        addressTo = random.sample(sorted(addressesList), 1)[0]
        while addressTo == addressFrom:
            addressTo = random.sample(sorted(addressesList), 1)[0]
        todaysDate = datetime.now()
        # generate deadline
        deltaDays = timedelta(days=random.randint(0,1))
        deltaHours = timedelta(hours=random.randint(0,1))
        deltaMinutes = timedelta(minutes=random.randint(0,10))
        deltaSeconds = timedelta(seconds=random.randint(0,120))
        #deltaSeconds = timedelta(seconds=random.randint(0,1))
        deadline = todaysDate + deltaDays + deltaHours + deltaMinutes + deltaSeconds
        deadline = todaysDate + deltaMinutes + deltaSeconds
        # calculate timeToDeadline
        timeToDeadline = deadline - todaysDate
        timeToDeadline = timeToDeadline/60
        #g = random.randint(1,254)
        #r = random.randint(1,254)
        #b = random.randint(1,254)
        #colour = [b, g, r]
        colour = [120, 120, 120]
        areaTarget = "none"
        xyLocationTarget = xyLocation.copy()
        status = "idle"
        carrier = -1
        package = Package(packageRollingCount, packageLog, itemValues, addressFrom, addressTo, deadline, timeToDeadline, colour, area, xyLocation, areaTarget, xyLocationTarget, status, carrier)
        return package