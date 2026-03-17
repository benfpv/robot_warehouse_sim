import random
import time

from data.warehouse.robot import *


class Robot_Functions:
    """Helpers for spawning robots onto the warehouse perimeter."""

    def import_robot(self, robotSpawnLocationsAvailableMap, robotsRollingCount, robotsInWarehouseCount, robotsMaxQuantity, robotsInWarehouse, robots, robotsTaskAssignmentList, chargersInWarehouse):
        """Attempt to spawn one new robot on the outer perimeter if capacity allows."""
        if robotsRollingCount < robotsMaxQuantity:
            #print('- Robot_Functions.import_robot()')
            # Search for adequate spawn area
            xyLocation = self.try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse, chargersInWarehouse)
            # Generate robot
            if (xyLocation):
                robot = self.generate_robot(xyLocation, robotsRollingCount)
                robots.append(robot)
                robotsTaskAssignmentList.append([robotsRollingCount, 0])
                robotsRollingCount += 1
                robotsInWarehouseCount += 1
                #print('- Len robots: {}'.format(len(self.robots)))
        return robotsRollingCount, robotsInWarehouseCount, robots, robotsTaskAssignmentList

    @staticmethod
    def try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse, chargersInWarehouse):
        """Pick a random free outer-perimeter cell (max 3 attempts).

        Returns [x, y] on success or [] if no free cell was found.
        """
        loc_count = 0
        # Generate candidate coordinate (n tries)
        while loc_count < 3:
            randIndex = random.randint(0, len(robotSpawnLocationsAvailableMap)-1)
            candidateCoordinate = robotSpawnLocationsAvailableMap[randIndex]
            x = candidateCoordinate[0]
            y = candidateCoordinate[1]
            if (robotsInWarehouse[y][x] == 0 and chargersInWarehouse[y][x] == 0):
                # Cell free of other robots and chargers
                xyLocation = [x, y]
                break
            loc_count += 1
        if loc_count == 3:
            return []
        return xyLocation

    @staticmethod
    def generate_robot(xyLocationSpawn, robotsRollingCount):
        #print('- generate_robot')
        robotNumber = robotsRollingCount
        robotLog = []
        timerCheckBattery = time.time()
        batteryPercent = random.randint(80,100)
        batteryChargingRate = random.randint(10,15) * .1
        batteryDepletingRate = random.randint(1,8) * .01
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