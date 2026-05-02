import numpy as np

from data.functions import Functions


class Warehouse_Init:
    """One-time initialisation helpers called during Warehouse construction."""

    @staticmethod
    def init_warehousePerimeter(warehouseWindowRes):
        """Return (perimeterCoords, perimeterCoordsMinusOne) for the grid boundary.

        The outer perimeter is used for charger/robot spawn locations.
        The inner perimeter (1 cell in) is used for charger placement to leave
        a navigable outer ring for robots.
        """
        warehousePerimeterCoordinates = Functions.find_perimeter_coordinates(warehouseWindowRes, 0)
        warehousePerimeterCoordinatesMinusOne = Functions.find_perimeter_coordinates(warehouseWindowRes, 1)
        return warehousePerimeterCoordinates, warehousePerimeterCoordinatesMinusOne

    @staticmethod
    def init_warehouseWindow(warehouseWindowRes, warehouseWindowCenter):
        """Allocate all per-cell occupancy grid arrays (uint8, shape H×W).

        Returns the resolved resolution, center, and five zero-filled arrays:
        packagesInWarehouse, robotsInWarehouse, chargersInWarehouse,
        idleAreasInWarehouse, packageTargetsInWarehouse.
        """
        packagesInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype='uint8')
        robotsInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype='uint8')
        chargersInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype='uint8')
        idleAreasInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype='uint8')
        packageTargetsInWarehouse = np.zeros((warehouseWindowRes[1], warehouseWindowRes[0]), dtype='uint8')
        return warehouseWindowRes, warehouseWindowCenter, packagesInWarehouse, robotsInWarehouse, chargersInWarehouse, idleAreasInWarehouse, packageTargetsInWarehouse

    @staticmethod
    def init_zoneMap(warehouseWindowRes):
        """Build the zone map (uint8 H×W array) with a *pad*-cell border gap.

        Zone IDs: 0 = neutral, 1 = import, 2 = storage, 3 = export.
        The three zones are horizontally equal thirds of the usable interior.
        """
        W, H = warehouseWindowRes
        pad   = 3      # minimum gap from all edges
        scale = 0.775  # ~60% of area (0.775² ≈ 0.60)
        zoneMap = np.zeros((H, W), dtype='uint8')
        inner_w  = W - pad * 2
        inner_h  = H - pad * 2
        zone_w   = int(inner_w * scale)
        zone_h   = int((inner_h // 3) * scale)
        # Centre the zone block within the inner area
        x0     = pad + (inner_w - zone_w) // 2
        y_base = pad + (inner_h - zone_h * 3) // 2
        zoneMap[y_base            : y_base + zone_h,     x0:x0 + zone_w] = 1  # ZONE_IMPORT
        zoneMap[y_base + zone_h   : y_base + zone_h * 2, x0:x0 + zone_w] = 2  # ZONE_STORAGE
        zoneMap[y_base + zone_h*2 : y_base + zone_h * 3, x0:x0 + zone_w] = 3  # ZONE_EXPORT
        print('- Zone map: shape {}, cells per zone: {}'.format(
            zoneMap.shape,
            {v: int(np.count_nonzero(zoneMap == v)) for v in [1, 2, 3]}
        ))
        return zoneMap