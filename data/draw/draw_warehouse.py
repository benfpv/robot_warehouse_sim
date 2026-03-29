import numpy as np
import cv2


class Draw_Warehouse:
    """Static helpers that paint warehouse entities onto NumPy image arrays."""

    @staticmethod
    def draw_warehouseZones(mainWindow, zoneMap, zoneColours):
        """Colour-fill each zone in *mainWindow* using the provided zone→colour mapping."""
        for zoneId, colour in zoneColours.items():
            mask = zoneMap == zoneId
            mainWindow[mask] = colour
        return mainWindow

    @staticmethod
    def draw_zoneMap_display(zoneMap, zoneColours, bgColour=(20, 20, 20)):
        """Return a BGR image of the zone map suitable for display as a sub-view."""
        h, w = zoneMap.shape
        img = np.full((h, w, 3), bgColour, dtype=np.uint8)
        for zoneId, colour in zoneColours.items():
            img[zoneMap == zoneId] = colour
        return img
    @staticmethod
    def draw_chargers(mainWindow, warehouse):
        """Paint each charger as a single pixel using its own colour."""
        if warehouse.chargers:
            for i in warehouse.chargers:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow

    @staticmethod
    def draw_packages(mainWindow, warehouse):
        """Paint each package as a single pixel using its deadline-coded colour."""
        if warehouse.packages:
            for i in warehouse.packages:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow

    @staticmethod
    def draw_robots(mainWindow, warehouse):
        """Paint each robot as a single pixel (white by default)."""
        if warehouse.robots:
            for i in warehouse.robots:
                mainWindow[i.xyLocation[1]][i.xyLocation[0]] = [i.colour[0], i.colour[1], i.colour[2]]
        return mainWindow

    @staticmethod
    def draw_package_arrows(img, packages, scale):
        """Two-pass render onto *img* for packages that are planned or being carried.

        Pass 1: dark-grey arrowedLine from current location to target.
        Pass 2: grey 2×2 dot at current location (drawn on top of arrows).
        """
        half = max(scale // 2, 1)
        arrow_col = (70, 70, 70)
        active = []
        for p in packages:
            if p.status not in ("move planned", "carried"):
                continue
            sx = p.xyLocation[0] * scale + half
            sy = p.xyLocation[1] * scale + half
            ex = p.xyLocationTarget[0] * scale + half
            ey = p.xyLocationTarget[1] * scale + half
            active.append((sx, sy, ex, ey))
            if (sx, sy) != (ex, ey):
                cv2.arrowedLine(img, (sx, sy), (ex, ey), arrow_col, 1, tipLength=0.05)
        # Draw dots after all arrows so they sit on top
        for (sx, sy, ex, ey) in active:
            img[max(sy-1, 0):sy+1, max(sx-1, 0):sx+1] = (160, 160, 160)
        return img