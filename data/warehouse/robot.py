from datetime import date, datetime
import math
import random
import time

class Robot:
    def __init__(self, robotNumber, robotLog, timerCheckBattery, batteryPercent, batteryChargingRate, batteryDepletingRate, actionQueue, colour, area, xyLocation, areaTarget, xyLocationTarget, xyLocationDiff, direction, cardinal, velocity, status, carrying, carrier) -> None:
        self.robotNumber = robotNumber
        self.robotLog = robotLog
        self.timerCheckBattery = timerCheckBattery
        self.batteryPercent = batteryPercent
        self.batteryChargingRate = batteryChargingRate
        self.batteryDepletingRate = batteryDepletingRate
        self.actionQueue = actionQueue
        self.colour = colour
        self.area = area
        self.xyLocation = xyLocation
        self.areaTarget = areaTarget
        self.xyLocationTarget = xyLocationTarget
        self.xyLocationDiff = xyLocationDiff
        self.direction = direction
        self.cardinal = cardinal
        self.velocity = velocity
        self.status = status
        self.carrying = carrying
        self.carrier = carrier

class Robot_Log:
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow