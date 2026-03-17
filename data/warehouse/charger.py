from datetime import date, datetime

class Charger:
    def __init__(self, chargerNumber, colour, area, xyLocation, status) -> None:
        self.chargerNumber = chargerNumber
        self.colour = colour
        self.area = area
        self.xyLocation = xyLocation
        self.status = status

class Charger_Log:
    def __init__(self, action, datetimeNow):
        self.action = action
        self.datetimeNow = datetimeNow