"""General-purpose grid and geometry utility functions."""
import random
import numpy as np


class Functions:
    """Static helpers for screen geometry, perimeter coordinates, and cardinal directions."""
    @staticmethod
    def get_screensize(screenshot):
        if not screenshot.any():
            return []
        screenshape = screenshot.shape
        screensize = [screenshape[1], screenshape[0]]
        return screensize

    @staticmethod
    def get_screencenter(screensize):
        if not screensize:
            return []
        screencenter = (int(screensize[0]*0.5), int(screensize[1]*0.5))
        return screencenter

    @staticmethod
    def get_screenarray_colour(screensize, backgroundColour):
        """Create a BGR uint8 array filled with *backgroundColour*."""
        if (not screensize) or (not backgroundColour):
            return []
        screenArray = np.zeros((screensize[1], screensize[0], 3), dtype = 'uint8')
        screenArray[:][:] = backgroundColour
        return screenArray
    
    @staticmethod
    def get_screenarray_gray(screensize):
        """Create a single-channel uint8 zero-filled array."""
        if (not screensize):
            return []
        screenArray = np.zeros((screensize[1], screensize[0]), dtype = 'uint8')
        screenArray[:][:] = 0
        return screenArray

    @staticmethod
    def find_perimeter_coordinates(windowRes, pixelsFromEdge):
        perimeterCoordinates = []
        # First Row
        for r in range(pixelsFromEdge, windowRes[0]-1-(1 * pixelsFromEdge)):
            c = pixelsFromEdge
            coordinates = [r,c]
            if coordinates not in perimeterCoordinates:
                perimeterCoordinates.append(coordinates)
        # Last Row
        for r in range(pixelsFromEdge, windowRes[0]-1-(1 * pixelsFromEdge)):
            c = windowRes[1]-1-(1 * pixelsFromEdge)
            coordinates = [r,c]
            if coordinates not in perimeterCoordinates:
                perimeterCoordinates.append(coordinates)
        # First Column
        for c in range(pixelsFromEdge, windowRes[1]-1-(1 * pixelsFromEdge)):
            r = pixelsFromEdge
            coordinates = [r,c]
            if coordinates not in perimeterCoordinates:
                perimeterCoordinates.append(coordinates)
        # Last Column
        for c in range(pixelsFromEdge, windowRes[1]-1-(1 * pixelsFromEdge)):
            r = windowRes[0]-1-(1 * pixelsFromEdge)
            coordinates = [r,c]
            if coordinates not in perimeterCoordinates:
                perimeterCoordinates.append(coordinates)
        return perimeterCoordinates

    @staticmethod
    def find_cardinal(degree):
        # Safe default and normalized heading prevent branch gaps from crashing callers.
        cardinal = 'E'
        if degree is None:
            return cardinal
        try:
            degree = float(degree)
        except (TypeError, ValueError):
            return cardinal
        degree = ((degree + 180.0) % 360.0) - 180.0
        # degree of 360
        if degree == 0:
            cardinal = 'E'
        elif degree > 0:
            if 0 <= degree < 30:
                cardinal = 'E'
            elif 30 <= degree < 60:
                cardinal = 'NE'
            elif 60 <= degree < 120:
                cardinal = 'N'
            elif 120 <= degree < 150:
                cardinal = 'NW'
            elif 150 <= degree < 210:
                cardinal = 'W'
            elif 210 <= degree < 240:
                cardinal = 'SW'
            elif 240 <= degree < 300:
                cardinal = 'S'
            elif 300 <= degree < 330:
                cardinal = 'SE'
            elif 330 <= degree <= 360:
                cardinal = 'E'
        elif degree < 0:
            if -30 < degree <= 0:
                cardinal = 'E'
            elif -60 < degree <= -30:
                cardinal = 'SE'
            elif -120 < degree <= -60:
                cardinal = 'S'
            elif -150 < degree <= -120:
                cardinal = 'SW'
            elif -210 < degree <= -150:
                cardinal = 'W'
            elif -240 < degree <= -210:
                cardinal = 'NW'
            elif -300 < degree <= -240:
                cardinal = 'N'
            elif -330 < degree <= -300:
                cardinal = 'NE'
            elif -360 <= degree <= -330:
                cardinal = 'E'
        return cardinal

    @staticmethod
    def find_location_from_cardinal(cardinal):
        xyMove = [0,0]
        if cardinal == 'E':
            xyMove[0] += 1
        elif cardinal == 'NE':
            xyMove[0] += 1
            xyMove[1] += 1
        elif cardinal == 'N':
            xyMove[1] += 1
        elif cardinal == 'NW':
            xyMove[0] -= 1
            xyMove[1] += 1
        elif cardinal == 'W':
            xyMove[0] -= 1
        elif cardinal == 'SW':
            xyMove[0] -= 1
            xyMove[1] -= 1
        elif cardinal == 'S':
            xyMove[1] -= 1
        elif cardinal == 'SE':
            xyMove[0] += 1
            xyMove[1] -= 1
        return xyMove

    @staticmethod
    def find_angle_from_cardinal(cardinal):
        if cardinal == 'E':
            coinflip = random.randint(0,1)
            if coinflip == 0:
                angle = random.randint(0,30)
            else:
                angle = random.randint(330,360)
        elif cardinal == 'SE':
            angle = random.randint(30,60)
        elif cardinal == 'S':
            angle = random.randint(60,120)
        elif cardinal == 'SW':
            angle = random.randint(120,150)
        elif cardinal == 'W':
            angle = random.randint(150,210)
        elif cardinal == 'NW':
            angle = random.randint(210,240)
        elif cardinal == 'N':
            angle = random.randint(240,300)
        elif cardinal == 'NE':
            angle = random.randint(300,330)
        return angle
    
    @staticmethod
    def zerofy_1d(array):
        for i in range(0, len(array)):
            if array[i] != 0:
                array[i] = 0
        return array

    @staticmethod
    def zerofy_2d(array):
        for row in range(0, len(array)):
            for col in range(0, len(array[row])):
                if array[row][col] != 0:
                    array[row][col] = 0
        return array

    @staticmethod
    def ensure_limit_1d(val, min_val, max_val):
        if (val < min_val):
            val = min_val
        elif (val > max_val):
            val = max_val
        return val