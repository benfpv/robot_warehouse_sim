import random
from datetime import datetime, timedelta
import numpy as np

from data.warehouse.warehouse_log import *
from data.warehouse.package import *

# Zone constants (must match warehouse.py)
ZONE_NONE = 0
ZONE_IMPORT = 1
ZONE_STORAGE = 2
ZONE_EXPORT = 3


class Package_Functions:
    """Helpers for spawning and positioning packages."""

    def import_package(self, zoneMap, chargersInWarehouse, packagesRollingCount, packagesInImportCount, packagesInWarehouseCount, packagesInWarehouse, packageTargetsInWarehouse, packagesMaxQuantity, packages, packagesLog, itemsList, addressesList, datetimeNow, spawnZoneId=ZONE_IMPORT):
        """Attempt to spawn one new package in the given zone if capacity allows."""
        if packagesInWarehouseCount < packagesMaxQuantity:
            movePackage, xyLocation = self.try_packageTargetLocation(zoneMap, spawnZoneId, packagesInWarehouse, packageTargetsInWarehouse, chargersInWarehouse)
            if (movePackage == True):
                zoneNames = {ZONE_IMPORT: 'import', ZONE_STORAGE: 'storage', ZONE_EXPORT: 'export'}
                spawnArea = zoneNames.get(spawnZoneId, 'neutral')
                package = self.generate_package(xyLocation, itemsList, addressesList, packagesRollingCount, area=spawnArea)
                packages.append(package)
                packagesLog.append(Packages_Log(packagesRollingCount, package, 'import', datetimeNow))
                packagesRollingCount += 1
                packagesInImportCount += 1
                packagesInWarehouseCount += 1
        return packagesRollingCount, packagesInImportCount, packagesInWarehouseCount, packages, packagesLog

    @staticmethod
    def try_packageTargetLocation(zoneMap, zoneId, packagesInWarehouse, packageTargetsInWarehouse, chargersInWarehouse, reference_xy=None, mode='random'):
        """Find a free cell in *zoneId* that has no package, planned target, or charger.

        Modes:
            'random'    — pick a random free cell (even spread, avoids clustering)
            'nearest'   — pick the closest free cell to reference_xy (min travel)
            'zone_edge' — pick the closest free cell adjacent to a different zone
                          (pre-stages for next pipeline move); falls back to nearest

        Returns (True, [x, y]) on success or (False, []) if the zone is full.
        """
        mode = str(mode).strip().lower()
        if mode not in ('random', 'nearest', 'zone_edge'):
            mode = 'random'

        candidates = np.argwhere(
            (zoneMap == zoneId) &
            (packagesInWarehouse == 0) &
            (packageTargetsInWarehouse == 0) &
            (chargersInWarehouse == 0)
        )
        if len(candidates) == 0:
            return False, []

        if mode == 'nearest' and reference_xy is not None:
            rx, ry = reference_xy
            dists = (candidates[:, 1] - rx) ** 2 + (candidates[:, 0] - ry) ** 2
            idx = int(np.argmin(dists))
        elif mode == 'zone_edge' and reference_xy is not None:
            # Filter to candidates 4-adjacent to a cell of a different zone
            h, w = zoneMap.shape
            edge_mask = np.zeros(len(candidates), dtype=bool)
            for ci in range(len(candidates)):
                cy, cx = candidates[ci]
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and zoneMap[ny, nx] != zoneId and zoneMap[ny, nx] != 0:
                        edge_mask[ci] = True
                        break
            edge_cands = candidates[edge_mask]
            if len(edge_cands) > 0:
                rx, ry = reference_xy
                dists = (edge_cands[:, 1] - rx) ** 2 + (edge_cands[:, 0] - ry) ** 2
                idx_e = int(np.argmin(dists))
                y, x = edge_cands[idx_e]
                return True, [int(x), int(y)]
            # Fallback to nearest
            rx, ry = reference_xy
            dists = (candidates[:, 1] - rx) ** 2 + (candidates[:, 0] - ry) ** 2
            idx = int(np.argmin(dists))
        else:
            idx = random.randint(0, len(candidates) - 1)

        y, x = candidates[idx]
        return True, [int(x), int(y)]

    @staticmethod
    def generate_package(xyLocation, itemsList, addressesList, packageRollingCount, area='import'):
        """Construct a new Package with randomised item, addresses, and deadline."""
        itemName = random.sample(sorted(itemsList), 1)[0]
        packageLog = []
        #print('- itemName: ' + str(itemName))
        itemValues = itemsList[itemName]
        #print('- itemValues: ' + str(itemValues))
        addressFrom = random.sample(sorted(addressesList), 1)[0]
        addressTo = random.sample(sorted(addressesList), 1)[0]
        while addressTo == addressFrom:
            addressTo = random.sample(sorted(addressesList), 1)[0]
        todaysDate = datetime.now()
        # generate deadline (tight windows for debugging)
        deltaSeconds = timedelta(seconds=random.randint(10, 90))
        deadline = todaysDate + deltaSeconds
        # timeToDeadline is recomputed from deadline each sim tick;
        # initialise to the full span so the Package constructor has a valid value.
        timeToDeadline = deadline - todaysDate
        #g = random.randint(1,254)
        #r = random.randint(1,254)
        #b = random.randint(1,254)
        #colour = [b, g, r]
        colour = [120, 120, 120]
        areaTarget = "none"
        xyLocationTarget = xyLocation.copy()
        status = "idle"
        carrier = -1
        package = Package(packageRollingCount, packageLog, itemValues, addressFrom, addressTo, deadline, timeToDeadline, colour, area, xyLocation, areaTarget, xyLocationTarget, status, carrier)
        return package