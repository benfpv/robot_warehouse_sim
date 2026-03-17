import numpy as np
import random
import math

class Timeseries_Functions:
    @staticmethod
    def rollUpdate(thisArray, numberOfRolls, newValue):
        thisArray = np.roll(thisArray, -numberOfRolls)
        thisArray[-1] = newValue
        return thisArray