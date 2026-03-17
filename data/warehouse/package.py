from datetime import datetime, timedelta
import time

class Package:
    def __init__(self, packageNumber, packageLog, itemValues, addressFrom, addressTo, deadline, timeToDeadline, colour, area, xyLocation, areaTarget, xyLocationTarget, status, carrier) -> None:
        self.packageNumber = packageNumber
        self.packageLog = packageLog
        self.itemValues = itemValues
        self.addressFrom = addressFrom
        self.addressTo = addressTo
        self.deadline = deadline
        self.timeToDeadline = timeToDeadline
        self.colour = colour
        self.area = area
        self.xyLocation = xyLocation
        self.areaTarget = areaTarget
        self.xyLocationTarget = xyLocationTarget
        self.status = status
        self.carrier = carrier
        self.createdAt = time.time()