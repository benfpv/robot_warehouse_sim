import logging
import random

from data.warehouse.charger import Charger
from data.warehouse.robot_functions import try_spawn_location

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
        return try_spawn_location(chargerLocationsAvailableMap, chargersInWarehouse, packagesInWarehouse)
    
    @staticmethod
    def generate_charger(xyLocationSpawn, chargersRollingCount):
        """Construct a new Charger at the given location."""
        chargerNumber = chargersRollingCount
        colour = (50, 190, 230)
        area = 'neutral'
        # Copy to avoid aliasing the caller's spawn-location list (matches Robot_Functions.generate_robot).
        xyLocation = list(xyLocationSpawn)
        status = 'idle'
        charger = Charger(chargerNumber, colour, area, xyLocation, status)
        return charger
    