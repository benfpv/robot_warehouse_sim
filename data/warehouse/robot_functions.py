from datetime import date, datetime
import math
import random
import time

from data.warehouse.robot import *

class Robot_Functions:
    def import_robot(self, robotSpawnLocationsAvailableMap, robotsRollingCount, robotsInWarehouseCount, robotsMaxQuantity, robotsInWarehouse, robots, robotsTaskAssignmentList, chargersInWarehouse):
        if robotsRollingCount < robotsMaxQuantity:
            #print('- Robot_Functions.import_robot()')
            # Search for adequate spawn area
            xyLocation = self.try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse)
            # Generate robot
            if (xyLocation):
                robot = self.generate_robot(xyLocation, robotsRollingCount)
                robots.append(robot)
                robotsTaskAssignmentList.append([robotsRollingCount, 0])
                robotsRollingCount += 1
                robotsInWarehouseCount += 1
                #print('- Len robots: {}'.format(len(self.robots)))
        return robotsRollingCount, robotsInWarehouseCount, robots, robotsTaskAssignmentList

    def try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse):
        #print('- try robotTargetLocation')
        loc_count = 0
        # Generate candidate coordinate (n tries)
        while loc_count < 3:
            randIndex = random.randint(0, len(robotSpawnLocationsAvailableMap)-1)
            candidateCoordinate = robotSpawnLocationsAvailableMap[randIndex]
            x = candidateCoordinate[0]
            y = candidateCoordinate[1]
            if (robotsInWarehouse[candidateCoordinate[1]][candidateCoordinate[0]] == 0):
                # If charger is free
                xyLocation = [x, y]
                break
            loc_count += 1
        if loc_count == 3:
            return []
        return xyLocation

    def generate_robot(xyLocationSpawn, robotsRollingCount):
        #print('- generate_robot')
        robotNumber = robotsRollingCount
        robotLog = []
        timerCheckBattery = time.time()
        batteryPercent = random.randint(80,100)
        batteryChargingRate = random.randint(10,15) * .1
        batteryDepletingRate = random.randint(1,8) * .01 #.0001 debug
        actionQueue = []
        colour = [255, 255, 255]
        area = 'import'
        xyLocation = xyLocationSpawn.copy()
        areaTarget = 'import'
        xyLocationTarget = xyLocation.copy()
        xyLocationDiff = [0,0]
        direction = 0
        cardinal = ''
        velocity = 0
        status = 'idle' # idle, moving to package, moving package to target,
        carrying = -1
        carrier = -1
        robot = Robot(robotNumber, robotLog, timerCheckBattery, batteryPercent, batteryChargingRate, batteryDepletingRate, actionQueue, colour, area, xyLocation, areaTarget, xyLocationTarget, xyLocationDiff, direction, cardinal, velocity, status, carrying, carrier)
        return robot