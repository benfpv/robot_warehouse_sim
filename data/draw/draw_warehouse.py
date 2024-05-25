import numpy as np
import cv2

class Draw_Warehouse:
    def draw_warehouseAreas(mainWindow, numberOfAreas, warehouseAreas, colour):
        numRows = numberOfAreas[0]
        numCols = numberOfAreas[1]
        count = 0
        for row in range(numRows):
            for col in range(numCols):
                xStart, yStart = warehouseAreas[count][0][0], warehouseAreas[count][0][1]
                xEnd, yEnd = warehouseAreas[count][1][0], warehouseAreas[count][1][1]
                
                cv2.rectangle(mainWindow, (xStart, yStart), (xEnd-1, yEnd-1), colour, -1)
                count += 1
        return mainWindow
    
    def draw_warehouseLanes(mainWindow, importAreas, storageAreas, exportAreas, colourOfLanes):
        # Determine and Draw Lanes around Areas ***WIP
        #print('- Determine and Draw Lanes around Areas')
        laneSpaces = [0]
        for i in importAreas:
            yStart, xStart = i[0][1], i[0][0]
            yEnd, xEnd = i[1][1]-1, i[1][0]-1
            for ls in laneSpaces:
                cv2.rectangle(mainWindow, (xStart-ls, yStart-ls), (xEnd+ls, yEnd+ls), colourOfLanes, 1)
        for i in storageAreas:
            yStart, xStart = i[0][1], i[0][0]
            yEnd, xEnd = i[1][1]-1, i[1][0]-1
            for ls in laneSpaces:
                cv2.rectangle(mainWindow, (xStart-ls, yStart-ls), (xEnd+ls, yEnd+ls), colourOfLanes, 1)
        for i in exportAreas:
            yStart, xStart = i[0][1], i[0][0]
            yEnd, xEnd = i[1][1]-1, i[1][0]-1
            for ls in laneSpaces:
                cv2.rectangle(mainWindow, (xStart-ls, yStart-ls), (xEnd+ls, yEnd+ls), colourOfLanes, 1)
        #print('- Lanes: {}'.format('laneshere *wip*'))
        return mainWindow

    def draw_chargers(mainWindow, warehouse):
        if warehouse.chargers:
            for i in warehouse.chargers:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow

    def draw_packages(mainWindow, warehouse):
        if warehouse.packages:
            for i in warehouse.packages:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow

    def draw_robots(mainWindow, warehouse):
        if warehouse.robots:
            for i in warehouse.robots:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow