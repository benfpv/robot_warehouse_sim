import random
import numpy as np

class Functions:
    def get_screensize(screenshot):
        #print("Functions.get_screensize()")
        if not screenshot.any():
            return []
        screenshape = screenshot.shape
        screensize = [screenshape[1], screenshape[0]]
        #print("- screensize: {}".format(screensize))
        return screensize

    def get_screencenter(screensize):
        #print("Functions.get_screencenter()")
        if not screensize:
            return []
        screencenter = (int(screensize[0]*0.5), int(screensize[1]*0.5))
        return screencenter

    def get_screenarray_colour(screensize, backgroundColour):
        print("Functions.get_screenarray_colour()")
        if (not screensize) or (not backgroundColour):
            return []
        screenArray = np.zeros((screensize[1], screensize[0], 3), dtype = 'uint8')
        screenArray[:][:] = backgroundColour
        return screenArray
    
    def get_screenarray_gray(screensize):
        print("Functions.get_screenarray_gray()")
        if (not screensize):
            return []
        screenArray = np.zeros((screensize[1], screensize[0]), dtype = 'uint8')
        screenArray[:][:] = 0
        return screenArray

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
        #print('- perimeterCoordinates: ')
        #for i in perimeterCoordinates:
        #    print('- ' + str(i))
        return perimeterCoordinates

    def find_cardinal(degree):
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
    
    def zerofy_1d(array):
        for i in range(0, len(array)):
            if array[i] != 0:
                array[i] = 0
        return array

    def zerofy_2d(array):
        for row in range(0, len(array)):
            for col in range(0, len(array[row])):
                if array[row][col] != 0:
                    array[row][col] = 0
        return array

    def ensure_limit_1d(val, min, max):
        if (val < min):
            val = min
        elif (val > max):
            val = max
        return val