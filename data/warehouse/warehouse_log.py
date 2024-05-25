import random
import math
from datetime import date, datetime, timedelta
import numpy as np

from data.functions import *

class Packages_Log:
    def __init__(self, packageNumber, package, action, datetimeNow) -> None:
        self.packageNumber = packageNumber
        self.package = package
        self.action = action
        self.datetimeNow = datetimeNow

class Robots_Log:
    def __init__(self, robotNumber, robot, action, datetimeNow):
        self.robotNumber = robotNumber
        self.robot = robot
        self.action = action
        self.datetimeNow = datetimeNow