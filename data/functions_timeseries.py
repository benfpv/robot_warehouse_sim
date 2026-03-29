"""Rolling-array helpers for timeseries data recording."""
import numpy as np
import random
import math


class Timeseries_Functions:
    """Static methods for rolling NumPy array updates."""
    @staticmethod
    def rollUpdate(thisArray, numberOfRolls, newValue):
        thisArray = np.roll(thisArray, -numberOfRolls)
        thisArray[-1] = newValue
        return thisArray