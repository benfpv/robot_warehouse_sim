import logging
import random
import time

from data.warehouse.charger import *

_log = logging.getLogger(__name__)


class Charger_Functions:
    """Helpers for spawning charging stations onto the warehouse perimeter."""

    def import_charger(self, chargerLocationsAvailableMap, chargersRollingCount, chargersMaxQuantity, chargersInWarehouse, chargers, packagesInWarehouse):
        """Attempt to spawn one new charger on the inner perimeter if capacity allows."""
        if chargersRollingCount < chargersMaxQuantity:
            # Search for adequate spawn area
            xyLocation = self.try_chargerTargetLocation(chargerLocationsAvailableMap, chargersInWarehouse, packagesInWarehouse)
            # Generate charger
            if (xyLocation):
                charger = self.generate_charger(xyLocation, chargersRollingCount)
                chargers.append(charger)
                chargersRollingCount += 1
            else:
                _log.warning('charger #%d spawn failed: no free cell after 3 attempts (%d/%d placed)',
                             chargersRollingCount, len(chargers), chargersMaxQuantity)
        return chargersRollingCount, chargers
    
    @staticmethod
    def try_chargerTargetLocation(chargerLocationsAvailableMap, chargersInWarehouse, packagesInWarehouse):
        """Pick a random free inner-perimeter cell (max 3 attempts).

        Returns [x, y] on success or [] if no free cell was found.
        """
        loc_count = 0
        # Generate candidate coordinate (n tries)
        while loc_count < 3:
            randIndex = random.randint(0, len(chargerLocationsAvailableMap)-1)
            candidateCoordinate = chargerLocationsAvailableMap[randIndex]
            x = candidateCoordinate[0]
            y = candidateCoordinate[1]
            if (chargersInWarehouse[y][x] == 0 and packagesInWarehouse[y][x] == 0):
                xyLocation = [x, y]
                break
            loc_count += 1
        if loc_count >= 3:
            return []
        return xyLocation
    
    @staticmethod
    def generate_charger(xyLocationSpawn, chargersRollingCount):
        """Construct a new Charger at the given location."""
        chargerNumber = chargersRollingCount
        colour = (50, 190, 230)
        area = 'neutral'
        xyLocation = xyLocationSpawn
        status = 'idle'
        charger = Charger(chargerNumber, colour, area, xyLocation, status)
        return charger
    