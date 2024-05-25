import random
import math
from datetime import date, datetime, timedelta

from data.warehouse.warehouse_log import *
from data.warehouse.package import *

class Package_Functions:
    def import_package(self, numTries, importAreas, numberOfImportAreas, sizeOfImportAreas, packagesRollingCount, packagesInImportCount, packagesPlannedInImportCount, packagesInWarehouseCount, packagesInImportAreas, packagesInWarehouse, packageTargetsInWarehouse, packagesMaxQuantity, packages, packagesLog, itemsList, addressesList, datetimeNow):
        if packagesInWarehouseCount < packagesMaxQuantity:
            # Try to find space for import
            movePackage, areaLocation, xyLocation = self.try_packageTargetLocation(numTries, importAreas, numberOfImportAreas, sizeOfImportAreas, packagesInImportAreas, packagesInWarehouse, packageTargetsInWarehouse)
            # If there is space, generate + import package
            if (movePackage == True):
                package = self.generate_package(areaLocation, xyLocation, itemsList, addressesList, packagesRollingCount)
                packages.append(package)
                packagesLog.append(Packages_Log(packagesRollingCount, package, 'import', datetimeNow))
                packagesRollingCount += 1
                packagesInImportCount += 1
                packagesInWarehouseCount += 1
                #print('- Len packages: {}'.format(len(packages)))
        return packagesRollingCount, packagesInImportCount, packagesInWarehouseCount, packages, packagesLog

    def try_packageTargetLocation(numTries, areas, numberOfAreas, sizeOfAreas, packagesInAreas, packagesInWarehouse, packageTargetsInWarehouse):
        loc_count = 0
        # Generate candidate location (n tries)
        for i in range(0, numTries):
            #print("- numberOfAreas: {}".format(numberOfAreas))
            #print("- sizeOfAreas: {}".format(sizeOfAreas))
            areaRowIndex = random.randint(0, numberOfAreas[0]-1)
            areaColIndex = random.randint(0, numberOfAreas[1]-1)
            areaIndicesIndex = random.randint(0, (sizeOfAreas[0] * sizeOfAreas[1])-1)
            #print("- [areaRowIndex]{}, [areaColIndex]{}, [areaIndicesIndex]{}".format(areaRowIndex, areaColIndex, areaIndicesIndex))
            if (packagesInAreas[areaRowIndex][areaColIndex][areaIndicesIndex] == 0):
                # Convert to "areas" format
                areaIndex = (areaRowIndex*numberOfAreas[0]) + areaColIndex
                #print("- [areaIndex]{} /{}".format(areaIndex, len(areas)))
                #print("- area: {}".format(areas[areaIndex]))
                areaY = math.floor(areaIndicesIndex / sizeOfAreas[0])
                areaX = areaIndicesIndex % sizeOfAreas[0]
                areaXY = [areaX, areaY]
                x = areas[areaIndex][0][0] + areaXY[0]
                y = areas[areaIndex][0][1] + areaXY[1]
                #print("- [areaX]{}, [areaY]{}, [areaXY]{}".format(areaX, areaY, areaXY))
                if (packagesInWarehouse[y][x] == 0) and (packageTargetsInWarehouse[y][x] == 0):
                    areaLocation = [areaRowIndex, areaColIndex, areaIndicesIndex]
                    xyLocation = [x, y]
                    return True, areaLocation, xyLocation
            loc_count += 1
        return False, [], []

    def generate_package(areaLocation, xyLocation, itemsList, addressesList, packageRollingCount):
        itemName = random.sample(sorted(itemsList), 1)[0]
        packageLog = []
        #print('- itemName: ' + str(itemName))
        itemValues = itemsList[itemName]
        #print('- itemValues: ' + str(itemValues))
        addressFrom = random.sample(sorted(addressesList), 1)
        addressTo = random.sample(sorted(addressesList), 1)
        while addressTo == addressFrom:
            addressTo = random.sample(sorted(addressesList), 1)
        todaysDate = datetime.now()
        # generate deadline
        deltaDays = timedelta(days=random.randint(0,30))
        deltaDays = timedelta(days=random.randint(0,1))
        deltaHours = timedelta(hours=random.randint(0,24))
        deltaHours = timedelta(hours=random.randint(0,1))
        deltaMinutes = timedelta(minutes=random.randint(0,60))
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
        area = 'import'
        areaTarget = "none"
        areaLocationTarget = areaLocation.copy()
        xyLocationTarget = xyLocation.copy()
        status = "idle"
        carrier = -1
        package = Package(packageRollingCount, packageLog, itemValues, addressFrom, addressTo, deadline, timeToDeadline, colour, area, areaLocation, xyLocation, areaTarget, areaLocationTarget, xyLocationTarget, status, carrier)
        return package
    
    def export_package(self, packagesInWarehouseCount, packagesInWarehouseAreas, packages, packagesLog, datetimeNow):
        if packages:
            i_count = 0
            packageIndicesToPop = []
            for i in packages:
                if (i.area == 'export') and (i.status == 'idle'):
                    if i.timeToDeadline.total_seconds() <= 0:
                        #print('- PastDeadline: package#: ' + str(i.packageNumber) + ', total_seconds: ' + str(i.timeToDeadline.total_seconds()))
                        packageIndicesToPop.append(i_count)
                i_count += 1
            if packageIndicesToPop:
                # Sort packageIndicesToPop
                packageIndicesToPop.sort(reverse=True)
                #print('- All Package Indices to Pop: ' + str(packageIndicesToPop))
                # Pop indices
                for ip in packageIndicesToPop:
                    packages, packagesLog, packagesInWarehouseAreas = self.remove_package_by_package_index(packages, ip, packagesLog, packagesInWarehouseAreas, 'export', datetimeNow)
        return packagesInWarehouseCount, packagesInWarehouseAreas, packages, packagesLog
    
    def remove_package_by_package_index(packages, packageIndex, packagesLog, packagesInWarehouseAreas, packageReasonForRemoval, datetimeNow):
        package = packages[packageIndex]
        packageNumber = package.packageNumber
        packageArea = package.area
        packageXYLocation = package.xyLocation
        print('- Popping: index: ' + str(packageIndex) + ', package#: ' + str(packageNumber) + ', area: ' + str(packageArea) + ', xyLocation: ' + str(packageXYLocation))
        packagesLog.append(PackagesLog(packageNumber, package, packageReasonForRemoval, datetimeNow))
        if packagesInWarehouseAreas[packageXYLocation[1]][packageXYLocation[0]] == 1:
            packagesInWarehouseAreas[packageXYLocation[1]][packageXYLocation[0]] = 0
        else:
            print('- ERROR: No package exists at that xyLocation. Cannot remove package from packagesInWarehouseAreas.')
        if package:
            packages.pop(packageIndex)
        else:
            print('- ERROR: No package exists at that packageIndex. Cannot remove package from packages.')
        return packages, packagesLog, packagesInWarehouseAreas