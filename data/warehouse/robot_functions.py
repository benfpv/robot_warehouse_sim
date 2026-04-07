import random

from data.warehouse.robot import Robot


def try_spawn_location(available_map, grid_a, grid_b, max_attempts=3):
    """Pick a random free cell from *available_map* (up to *max_attempts* tries).

    Returns [x, y] on success or [] if no free cell was found.
    A cell is free when both *grid_a* and *grid_b* are 0 at that position.
    """
    for _ in range(max_attempts):
        x, y = available_map[random.randint(0, len(available_map) - 1)]
        if grid_a[y][x] == 0 and grid_b[y][x] == 0:
            return [x, y]
    return []


class Robot_Functions:
    """Helpers for spawning robots onto the warehouse perimeter."""

    def import_robot(self, robotSpawnLocationsAvailableMap, robotsRollingCount, robotsInWarehouseCount, robotsMaxQuantity, robotsInWarehouse, robots, robotsTaskAssignmentList, chargersInWarehouse):
        """Attempt to spawn one new robot on the outer perimeter if capacity allows."""
        if robotsRollingCount < robotsMaxQuantity:
            # Search for adequate spawn area
            xyLocation = self.try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse, chargersInWarehouse)
            # Generate robot
            if (xyLocation):
                robot = self.generate_robot(xyLocation, robotsRollingCount)
                robots.append(robot)
                robotsTaskAssignmentList.append([robotsRollingCount, 0])
                robotsRollingCount += 1
                robotsInWarehouseCount += 1
        return robotsRollingCount, robotsInWarehouseCount, robots, robotsTaskAssignmentList

    @staticmethod
    def try_robotTargetLocation(robotSpawnLocationsAvailableMap, robotsInWarehouse, chargersInWarehouse):
        """Pick a random free outer-perimeter cell (max 3 attempts).

        Returns [x, y] on success or [] if no free cell was found.
        """
        return try_spawn_location(robotSpawnLocationsAvailableMap, robotsInWarehouse, chargersInWarehouse)

    @staticmethod
    def generate_robot(xyLocationSpawn, robotsRollingCount):
        """Construct a new Robot with randomised battery and default motion model."""
        robotNumber = robotsRollingCount
        robotLog = []
        timerCheckBattery = 0.0
        batteryPercent = random.randint(80,100)
        batteryChargingRate = random.randint(10,15) * .1
        # 25% more battery capacity: same charge rate, 20% lower drain.
        batteryDepletingRate = batteryChargingRate * 0.1
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
        status = 'idle'
        carrying = -1
        carrier = -1
        robot = Robot(robotNumber, robotLog, timerCheckBattery, batteryPercent, batteryChargingRate, batteryDepletingRate, actionQueue, colour, area, xyLocation, areaTarget, xyLocationTarget, xyLocationDiff, direction, cardinal, velocity, status, carrying, carrier)
        robot.birthLocation = xyLocationSpawn.copy()
        return robot