from datetime import date, datetime
import math
import random
import time

from data.warehouse.charger import *

class Charger_Functions:
    def import_charger(self, chargerLocationsAvailableMap, chargersRollingCount, chargersMaxQuantity, chargersInWarehouse, chargers):
        if chargersRollingCount < chargersMaxQuantity:
            # Search for adequate spawn area
            xyLocation = self.try_chargerTargetLocation(chargerLocationsAvailableMap, chargersInWarehouse)
            # Generate charger
            if (xyLocation):
                charger = self.generate_charger(xyLocation, chargersRollingCount)
                chargers.append(charger)
                chargersRollingCount += 1
        return chargersRollingCount, chargers
    
    def try_chargerTargetLocation(chargerLocationsAvailableMap, chargersInWarehouse):
        #print('- try chargerTargetLocation')
        loc_count = 0
        # Generate candidate coordinate (n tries)
        while loc_count < 3:
            randIndex = random.randint(0, len(chargerLocationsAvailableMap)-1)
            candidateCoordinate = chargerLocationsAvailableMap[randIndex]
            x = candidateCoordinate[0]
            y = candidateCoordinate[1]
            if (chargersInWarehouse[candidateCoordinate[1]][candidateCoordinate[0]] == 0):
                xyLocation = [x, y]
                break
            loc_count += 1
        if loc_count >= 3:
            return []
        return xyLocation
    
    def generate_charger(xyLocationSpawn, chargersRollingCount):
        #print('- generate_robot')
        chargerNumber = chargersRollingCount
        colour = (50, 160, 160)
        area = 'import'
        xyLocation = xyLocationSpawn
        status = 'idle' # idle, charging
        charger = Charger(chargerNumber, colour, area, xyLocation, status)
        return charger
    