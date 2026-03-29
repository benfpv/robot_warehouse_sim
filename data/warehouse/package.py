"""Package data class for the warehouse lifecycle."""
from datetime import datetime, timedelta
import time

class Package:
    """A package in the warehouse lifecycle (import → storage → export).

    Attributes:
        packageNumber:    Unique ID (sequential from 0).
        packageLog:       List of log entries tracking status changes.
        itemValues:       Item object (name, weight, dimensions).
        addressFrom:      Origin Address object.
        addressTo:        Destination Address object.
        deadline:         datetime when the package must be exported.
        timeToDeadline:   timedelta remaining until deadline (updated each tick).
        colour:           BGR display colour (set by update_packages_colours).
        area:             Current zone name ('import', 'storage', 'export').
        xyLocation:       Current [x, y] grid position.
        areaTarget:       Target zone for next planned move ('import', 'storage', 'export', 'none').
        xyLocationTarget: Destination [x, y] within the target zone.
        status:           'idle', 'move planned', 'carried', or 'error'.
        carrier:          Robot number carrying this package, or -1.
        createdAt:        Wall-clock timestamp of package spawn.
        deliveredAt:      Wall-clock timestamp of last drop-off (delivery cooldown).
        plannedAt:        Wall-clock timestamp when status became 'move planned' (stale-plan eviction).
    """
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
        self.deliveredAt = None  # stamped when dropped off; used for export cooldown
        self.plannedAt = None     # stamped when status -> 'move planned'; stale-plan eviction