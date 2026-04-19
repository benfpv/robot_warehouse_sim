"""Robot data class with motion model and path-planning state."""
from datetime import datetime

class Robot:
    """Autonomous warehouse robot.

    Attributes:
        robotNumber:       Unique ID (sequential from 0).
        robotLog:          List of Robot_Log entries.
        timerCheckBattery: Cooldown timer for battery-check scheduling.
        batteryPercent:    Current charge level (0–100).
        batteryChargingRate:   Charge gained per tick while docked.
        batteryDepletingRate:  Base drain per tick while moving/idle.
        batteryDrainMultiplier: Dynamic drain multiplier (1.0 normal, higher when carrying).
        actionQueue:       List of (target_xy, action_string) tuples — FIFO task queue.
        colour:            BGR display colour.
        area:              Current zone name ('import', 'storage', 'export', 'none').
        xyLocation:        Current [x, y] grid position.
        areaTarget:        Target zone for current task.
        xyLocationTarget:  Next [x, y] waypoint the robot is stepping toward.
        xyLocationDiff:    Signed step direction [dx, dy] computed from path.
        direction:         Display-only heading angle (degrees); snap-set by movement controller.
        cardinal:          Display-only cardinal direction string.
        velocity:          Current movement speed (cells/tick).
        status:            Current action string from robotsActionsList.
        carrying:          Package number being carried, or -1.
        carrier:           (Unused on Robot; mirrors Package.carrier for symmetry.)
        createdAt:         Sim-time stamp of robot creation.

    Motion model:
        desiredVelocity:     Target speed for this tick (set by speed profile).
        maxVelocity:         Absolute speed cap (0.50 cells/tick).
        accelerationRate:    Speed increase per tick (0.05).
        decelerationRate:    Speed decrease per tick (0.05).
        minMovingVelocity:   Below this the robot snaps to zero (0.10).
        movementProgress:    Fractional grid-step accumulator for sub-cell movement.
        blockedTicks:        Consecutive ticks the robot couldn't move (deadlock detection).

    Path planning (A*):
        path:              Dense list of [x, y] waypoints from A*.
        pathPlan:          Per-waypoint metadata dicts (speed, turn_deg).
        pathTarget:        Final [x, y] goal of the current path.
        pathAge:           Ticks since last replan.
        recentLocations:   Ring buffer of last positions (stuck detection).
        replanCooldownTicks: Ticks remaining before replan is allowed.

    Fleet management:
        decommissioning:   True when robot is flagged for retirement.
        birthLocation:     [x, y] grid cell where the robot was originally spawned.
    """
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
        self.xyLocation = list(xyLocation)
        self.areaTarget = areaTarget
        self.xyLocationTarget = xyLocationTarget
        self.xyLocationDiff = xyLocationDiff
        self.direction = direction      # display-only: snap-set by movement controller
        self.targetDirection = direction
        self.cardinal = cardinal          # display-only
        self.velocity = velocity
        self.status = status
        self.carrying = carrying
        self.carrier = carrier
        self.createdAt = 0.0   # sim-time stamp, set by warehouse at spawn
        self.batteryDrainMultiplier = 1.0
        # Phase 1 motion model: velocity converges toward desiredVelocity each tick.
        self.desiredVelocity = 0.0
        self.maxVelocity = 0.50
        self.accelerationRate = 0.05
        self.decelerationRate = 0.05
        self.minMovingVelocity = 0.10
        self.lastCardinal = cardinal
        # Accumulates fractional movement in limp mode until a full grid step is earned.
        self.movementProgress = 0.0
        # Fine-grained tick state for movement and deadlock recovery.
        self.blockedTicks = 0
        self.degrees = round(float(direction))
        # Route planning state (A* waypoints).
        self.path = []
        self.pathPlan = []
        self.pathTarget = None
        self.pathAge = 0
        self.recentLocations = []
        self.replanCooldownTicks = 0
        self.stallTicks = 0
        self.hasCharged = False
        self.hasCarried = False
        self.decommissioning = False
        self._pending_removal = False
        self.birthLocation = xyLocation.copy()

class Robot_Log:
    """Immutable log entry recording a robot action at a point in time."""
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow