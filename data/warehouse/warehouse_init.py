import random
import math
from datetime import date, datetime, timedelta
import numpy as np
import cv2

from data.functions import *

class Warehouse_Init:
    def init_warehousePerimeter(warehouseWindowRes):
        warehousePerimeterCoordinates = Functions.find_perimeter_coordinates(warehouseWindowRes, 0)
        warehousePerimeterCoordinatesMinusOne = Functions.find_perimeter_coordinates(warehouseWindowRes, 1)
        return warehousePerimeterCoordinates, warehousePerimeterCoordinatesMinusOne

    def init_warehouseWindow(warehouseWindowRes, warehouseWindowCenter):
        packagesInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype = 'uint8')
        robotsInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype = 'uint8')
        chargersInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype = 'uint8')
        idleAreasInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype = 'uint8')
        packageTargetsInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype = 'uint8')
        return warehouseWindowRes, warehouseWindowCenter, packagesInWarehouse, robotsInWarehouse, chargersInWarehouse, idleAreasInWarehouse, packageTargetsInWarehouse
    
    def init_createAreasXY(warehouseWindowRes):
        importAreasXY = cv2.rectangle(mapBinary1, [2,2], [6,6], color = 1, thickness = -1)
        storageAreasXY = cv2.rectangle(mapBinary2, [12,12], [14,14], color = 1, thickness = -1)
        exportAreasXY = cv2.rectangle(mapBinary3, [4,6], [10,10], color = 1, thickness = -1)
        return importAreasXY, storageAreasXY,exportAreasXY

    def init_areasSpace(numberOfLanes, packageDimensionsLimit):
        # Space
        space = numberOfLanes * packageDimensionsLimit
        demispace = space * .5
        print('- Space, demispace of areas: {}, {}'.format(space, demispace))
        return space, demispace

    def init_floorsHeight(warehouseWindowRes):
        # Height of Floors
        windowHeightThird = math.floor(warehouseWindowRes[1]/3)-1
        heightOfImportFloor, heightOfStorageFloor, heightOfExportFloor = windowHeightThird, windowHeightThird, windowHeightThird
        print('- Height of floors: {}, {}, {}'.format(heightOfImportFloor, heightOfStorageFloor, heightOfExportFloor))
        return heightOfImportFloor, heightOfStorageFloor, heightOfExportFloor
    
    def init_areasSize():
        # sizeOfImportAreas
        sizeOfImportAreas = [10,10]
        sizeOfStorageAreas = [10,10]
        sizeOfExportAreas = [10,10]
        print('- Size of areas: {}, {}, {}'.format(sizeOfImportAreas, sizeOfStorageAreas, sizeOfExportAreas))
        return sizeOfImportAreas, sizeOfStorageAreas, sizeOfExportAreas
    
    def init_areasNumber(self, heightOfImportFloor, heightOfStorageFloor, heightOfExportFloor, sizeOfImportAreas, sizeOfStorageAreas, sizeOfExportAreas, demispace, warehouseWindowRes):
        # numberOfAreas AND convert to RowsCols
        numberOfImportAreas = self.init_areaNumber(heightOfImportFloor, sizeOfImportAreas, demispace, warehouseWindowRes)
        numberOfStorageAreas = self.init_areaNumber(heightOfStorageFloor, sizeOfStorageAreas, demispace, warehouseWindowRes)
        numberOfExportAreas = self.init_areaNumber(heightOfExportFloor, sizeOfExportAreas, demispace, warehouseWindowRes)
        print('- Number of areas: {}, {}, {}'.format(numberOfImportAreas, numberOfStorageAreas, numberOfExportAreas))
        return numberOfImportAreas, numberOfStorageAreas, numberOfExportAreas
    
    def init_areaNumber(heightOfFloor, sizeOfAreas, demispace, warehouseWindowRes):
        numRows = (heightOfFloor) / (sizeOfAreas[1] + demispace + demispace)
        numCols = (warehouseWindowRes[0]) / (sizeOfAreas[0] + demispace + demispace)
        numberOfAreas = [int(numRows), int(numCols)]
        return numberOfAreas

    def init_areas(self, numberOfImportAreas, numberOfStorageAreas, numberOfExportAreas, heightOfImportFloor, heightOfStorageFloor, heightOfExportFloor, space, warehouseWindowRes, warehouseWindowCenter):
        # Determine xy Area Start/Ends
        xyStartImport = [space, space*2]
        xyEndImport = [int(warehouseWindowRes[0] - space*2), heightOfImportFloor]
        xyStartStorage = [space, space*2 + heightOfImportFloor]
        xyEndStorage = [int(warehouseWindowRes[0] - space*2), (heightOfImportFloor + heightOfStorageFloor)]
        xyStartExport = [space, space*2 + heightOfImportFloor + heightOfStorageFloor]
        xyEndExport = [int(warehouseWindowRes[0] - space*2), (heightOfImportFloor + heightOfStorageFloor + heightOfExportFloor)]
        # Determine and Draw Areas
        sizeOfImportAreas, importAreas = self.init_area(xyStartImport, xyEndImport, numberOfImportAreas, space, warehouseWindowRes, warehouseWindowCenter)
        sizeOfStorageAreas, storageAreas = self.init_area(xyStartStorage, xyEndStorage, numberOfStorageAreas, space, warehouseWindowRes, warehouseWindowCenter)
        sizeOfExportAreas, exportAreas = self.init_area(xyStartExport, xyEndExport, numberOfExportAreas, space, warehouseWindowRes, warehouseWindowCenter)
        lenImportAreas = len(importAreas)
        lenStorageAreas = len(storageAreas)
        lenExportAreas = len(exportAreas)
        print("- Size of Areas: {}, {}, {}".format(sizeOfImportAreas, sizeOfStorageAreas, sizeOfExportAreas))
        print('- Len Areas: {}, {}, {}'.format(lenImportAreas, lenStorageAreas, lenExportAreas))
        return sizeOfImportAreas, sizeOfStorageAreas, sizeOfExportAreas, importAreas, storageAreas, exportAreas
    
    def init_area(xyAreaStart, xyAreaEnd, numberOfAreas, space, warehouseWindowRes, warehouseWindowCenter):
        numRows = numberOfAreas[0]
        numCols = numberOfAreas[1]
        
        areasXY = np.zeros((numRows * numCols, 2, 2), dtype = 'int')
        count = 0

        yTotalLen = int(xyAreaEnd[1] - xyAreaStart[1])
        yTotalSize = int(yTotalLen / numRows)
        
        xTotalLen = int(xyAreaEnd[0] - xyAreaStart[0])
        xTotalSize = int(xTotalLen / numCols)

        for row in range(numRows):
            colSwitch = 1
            colSwitchCount = 1
            yStart = xyAreaStart[1] + (yTotalSize * row) + space
            yEnd = yStart + yTotalSize - space
            for col in range(numCols):
                xStart = xyAreaStart[0] + (xTotalSize * col) + space
                xEnd = xStart + xTotalSize - space
                areasXY[count][0][0], areasXY[count][0][1] = xStart, yStart
                areasXY[count][1][0], areasXY[count][1][1] = xEnd, yEnd
                count += 1
        #print('- Len Area: {}'.format(len(areasXY)))
        #print(areasXY)
        return [xTotalSize-space, yTotalSize-space], areasXY
    
    def init_packagesInAreas(self, numberOfImportAreas, packagesInImportAreas, sizeOfImportAreas, numberOfStorageAreas, packagesInStorageAreas, sizeOfStorageAreas, numberOfExportAreas, packagesInExportAreas, sizeOfExportAreas):
        packagesInImportAreas, numberOfImportSlots = self.init_packagesInArea(numberOfImportAreas, packagesInImportAreas, sizeOfImportAreas)
        packagesInStorageAreas, numberOfStorageSlots = self.init_packagesInArea(numberOfStorageAreas, packagesInStorageAreas, sizeOfStorageAreas)
        packagesInExportAreas, numberOfExportSlots = self.init_packagesInArea(numberOfExportAreas, packagesInExportAreas, sizeOfExportAreas)
        print('- Packages in areas: {}, {}, {}'.format(len(packagesInImportAreas), len(packagesInStorageAreas[0]), len(packagesInExportAreas[0][0])))
        return packagesInImportAreas, packagesInStorageAreas, packagesInExportAreas, numberOfImportSlots, numberOfStorageSlots, numberOfExportSlots
    
    def init_packagesInArea(numberOfAreas, packagesInAreas, sizeOfAreas):
        numberOfSlots = (sizeOfAreas[0]*sizeOfAreas[1]) * (numberOfAreas[0] * numberOfAreas[1])
        packagesInAreas = np.zeros((numberOfAreas[0], numberOfAreas[1], sizeOfAreas[0]*sizeOfAreas[1]), dtype = 'uint8')
        return packagesInAreas, numberOfSlots