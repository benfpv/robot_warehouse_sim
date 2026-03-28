from datetime import date, datetime
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
        self.targetDirection = direction
        self.cardinal = cardinal
        self.velocity = velocity
        self.status = status
        self.carrying = carrying
        self.carrier = carrier
        self.createdAt = time.time()
        self.batteryDrainMultiplier = 1.0
        # Phase 1 motion model: velocity converges toward desiredVelocity each tick.
        self.desiredVelocity = 0.0
        self.maxVelocity = 0.6
        self.accelerationRate = 0.008
        self.decelerationRate = 0.008
        self.minMovingVelocity = 0.06
        # Directionality model: turning has non-zero time cost.
        self.turnRateDegPerTick = 6.0
        self.rotateMoveThresholdDeg = 18.0
        self.rotateBrakeThresholdDeg = 45.0
        self.lastHeadingNormalizeLogT = 0.0
        self.lastCardinal = cardinal
        # Accumulates fractional movement in limp mode until a full grid step is earned.
        self.movementProgress = 0.0
        # Fine-grained tick state for movement and deadlock recovery.
        self.blockedTicks = 0
        self.degrees = round(float(direction))

class Robot_Log:
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow