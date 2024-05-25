from datetime import date, datetime, timedelta
import math
import random
import numpy as np
import cv2 
import time

class Warehouse_Functions:
    def convert_xyWarehouse_to_xyArea(self, areas, numberOfAreas, xyLocation):
        len_areas = len(areas)
        xyLocationWithinAreaBool = False
        for i in range(0, len_areas):
            this_area = areas[i]
            xyLocationWithinAreaBool = self.check_if_within_box(xyLocation, this_area)
            if (xyLocationWithinAreaBool == True):
                break
        # print("- numberOfAreas: {}".format(numberOfAreas))
        # print("- i: {}".format(i))
        # print("- xyLocationWithinAreaBool: {}".format(xyLocationWithinAreaBool))
        if (xyLocationWithinAreaBool == False):
            return False, [], []
        areasRow = int(i / numberOfAreas[1])
        areasCol = i % numberOfAreas[1]
        areasRowCol = [areasRow, areasCol]
        # print("- areasRowCol: {}".format(areasRowCol))
        areaIndex = self.convert_xyWarehouse_to_areaIndex(xyLocation, this_area)
        return xyLocationWithinAreaBool, areasRowCol, areaIndex
    
    def check_if_within_box(xyLocation, thisArea):
        xWithin = False
        yWithin = False
        if (thisArea[0][0] <= xyLocation[0] < thisArea[1][0]):
            xWithin = True
        if (thisArea[0][1] <= xyLocation[1] < thisArea[1][1]):
            yWithin = True
        if (xWithin == True) and (yWithin == True):
            return True
        else:
            return False
    
    def convert_xyWarehouse_to_areaIndex(xyLocation, thisArea):
        #print("- xyLocation: {}".format(xyLocation))
        #print("- thisArea: {}, {}".format(thisArea[0], thisArea[1]))
        rowSize = thisArea[1][0]-thisArea[0][0]
        colSize = thisArea[1][1]-thisArea[0][1]
        xIn = (xyLocation[0] - thisArea[0][0])
        yIn = (xyLocation[1] - thisArea[0][1])
        areaIndex = (yIn*rowSize) + xIn
        #print("- rowSize: {}".format(rowSize))
        #print("- colSize: {}".format(colSize))
        #print("- xIn: {}".format(xIn))
        #print("- yIn: {}".format(yIn))
        #print("- areaIndex: {}".format(areaIndex))
        return areaIndex