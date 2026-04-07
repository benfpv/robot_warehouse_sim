"""Charger data class for stationary charging stations."""
from datetime import datetime

class Charger:
    """A stationary charging station.

    Attributes:
        chargerNumber: Unique ID (sequential from 0).
        colour:        BGR display colour.
        area:          Zone name where the charger is placed.
        xyLocation:    [x, y] grid position.
        status:        'idle', 'charging planned', or 'charging'.
    """
    def __init__(self, chargerNumber, colour, area, xyLocation, status) -> None:
        self.chargerNumber = chargerNumber
        self.colour = colour
        self.area = area
        self.xyLocation = xyLocation
        self.status = status

class Charger_Log:
    """Immutable log entry recording a charger event."""
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow