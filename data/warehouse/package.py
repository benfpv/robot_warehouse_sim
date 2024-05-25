import random
import math
from datetime import date, datetime, timedelta

class Package:
    def __init__(self, packageNumber, packageLog, itemValues, addressFrom, addressTo, deadline, timeToDeadline, colour, area, areaLocation, xyLocation, areaTarget, areaLocationTarget, xyLocationTarget, status, carrier) -> None:
        self.packageNumber = packageNumber
        self.packageLog = packageLog
        self.itemValues = itemValues
        self.addressFrom = addressFrom
        self.addressTo = addressTo
        self.deadline = deadline
        self.timeToDeadline = timeToDeadline
        self.colour = colour
        self.area = area
        self.areaLocation = areaLocation
        self.xyLocation = xyLocation
        self.areaTarget = areaTarget
        self.areaLocationTarget = areaLocationTarget
        self.xyLocationTarget = xyLocationTarget
        self.status = status
        self.carrier = carrier

class Package_Log:
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow