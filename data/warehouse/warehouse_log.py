"""Immutable log-entry classes for package and robot event histories."""
from datetime import datetime


class Packages_Log:
    """Records a single package lifecycle event (import, move, export, etc.)."""
    def __init__(self, packageNumber, package, action, datetimeNow) -> None:
        self.packageNumber = packageNumber
        self.package = package
        self.action = action
        self.datetimeNow = datetimeNow

class Robots_Log:
    """Records a single robot lifecycle event (spawn, task assignment, etc.)."""
    def __init__(self, robotNumber, robot, action, datetimeNow):
        self.robotNumber = robotNumber
        self.robot = robot
        self.action = action
        self.datetimeNow = datetimeNow