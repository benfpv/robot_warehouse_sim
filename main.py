import numpy as np
import math
import random
import time
import cv2

from data.item import *
from data.importer import *
from data.functions import *

from data.warehouse.warehouse_init import *
from data.warehouse.warehouse import *
from data.warehouse.package import *
from data.warehouse.robot import *
from data.draw.draw_warehouse import *
from data.display.display_functions import *

class MainGame:
    def __init__(self):
        print("--- MainGame Init ---")
        self.timeStart = int(time.time())
        self.exit = False
        # User Parameters
        self.warehouse_res = (80,80)
        self.warehouse_windowBackgroundColour = [50,50,50]
        self.warehouse_windowRes = (366, 366) # upsized resolution (width, height)

        # Init Windows
        self = self.initWarehouseWindow()
        self = self.initSecondaryWindows()
        # Init Imports
        self.itemsList = Importer.init_import_csv_as_list('resources/list_items.csv')
        self.itemsList = Importer.init_objectify_items_list(self.itemsList)
        self.addressesList = Importer.init_import_csv_as_list('resources/list_addresses.csv')
        self.addressesList = Importer.init_objectify_addresses_list(self.addressesList)
        # Init Warehouse
        self.warehouse = Warehouse(self.warehouse_res, self.warehouse_windowBackgroundColour, self.warehouse_windowCenter, self.warehouse_windowArray, self.itemsList, self.addressesList)

        # Temporary draw
        self.warehouseWindow = self.warehouse_windowArray.copy()

    def initWarehouseWindow(self):
        self.warehouse_windowCenter = Functions.get_screencenter(self.warehouse_res)
        self.warehouse_windowArray = Functions.get_screenarray_colour(self.warehouse_res, self.warehouse_windowBackgroundColour)
        self.warehouse_windowWindow = self.warehouse_windowArray.copy()
        print('- warehouse_res: {}, center: {}, arrayShape: {}, windowRes: {}'.format(self.warehouse_res, self.warehouse_windowCenter, 
                                                                                        np.shape(self.warehouse_windowArray), self.warehouse_windowRes))
        return self
    
    def initSecondaryWindows(self):
        self.warehouse_chargersInWarehouse_windowRes = (int(self.warehouse_windowRes[0]*.5),int(self.warehouse_windowRes[1]*.5))
        self.warehouse_packagesInWarehouse_windowRes = (int(self.warehouse_windowRes[0]*.5),int(self.warehouse_windowRes[1]*.5))
        self.warehouse_robotsInWarehouse_windowRes = (int(self.warehouse_windowRes[0]*.5),int(self.warehouse_windowRes[1]*.5))
        self.warehouse_packageTargetsInWarehouse_windowRes = (int(self.warehouse_windowRes[0]*.5),int(self.warehouse_windowRes[1]*.5))
        print('- windowRes: chargers: {}, packages: {}, robots: {}, packageTargets: {}'.format(self.warehouse_chargersInWarehouse_windowRes, self.warehouse_packagesInWarehouse_windowRes,
                                                                                                self.warehouse_robotsInWarehouse_windowRes, self.warehouse_packageTargetsInWarehouse_windowRes))
        self.warehouse_chargersInWarehouse_windowPosition = [self.warehouse_windowRes[0], 0]
        self.warehouse_packagesInWarehouse_windowPosition = [self.warehouse_windowRes[0] + self.warehouse_chargersInWarehouse_windowRes[0], 0]
        self.warehouse_robotsInWarehouse_windowPosition = [self.warehouse_windowRes[0], self.warehouse_chargersInWarehouse_windowRes[1]]
        self.warehouse_packageTargetsInWarehouse_windowPosition = [self.warehouse_windowRes[0] + self.warehouse_robotsInWarehouse_windowRes[0], self.warehouse_chargersInWarehouse_windowRes[1]]
        print('- windowPosition: chargers: {}, packages: {}, robots: {}, packageTargets: {}'.format(self.warehouse_chargersInWarehouse_windowPosition, self.warehouse_packagesInWarehouse_windowPosition,
                                                                                                self.warehouse_robotsInWarehouse_windowPosition, self.warehouse_packageTargetsInWarehouse_windowPosition))
        return self

    def gameLoop(self, loop_count):
        timeLoopStart = time.time()

        self.timeElapsed = int(time.time() - self.timeStart)
        #print('--- New loop --- #{}, timeElapsed: {}'.format(loop_count, self.timeElapsed))

        if self.timeElapsed > 24000:
            self.gameEnd()
            return self

        # Update Existing
        self.warehouse = self.warehouse.update_warehouse()

        # Draw
        self.gameDraw()

        frameTime = round(time.time() - timeLoopStart, 3)
        if loop_count % 1 == 0:
            pass
            #print('- L#{}, t: {}, pIn: {}, pToMove: {}, rIn: {}, pMoving: {}, avg_frameTime: {}'.format(loop_count, self.timeElapsed, self.warehouse.packagesInWarehouseCount, len(self.warehouse.packagesMoveList), self.warehouse.robotsInWarehouseCount, len(self.warehouse.packagesMovingList), frameTime))

        return self, frameTime

    def gameDraw(self):
        # Draw Warehouse
        self.warehouseWindow = self.warehouse_windowArray.copy()
        self.warehouseWindow = Draw_Warehouse.draw_warehouseAreas(self.warehouseWindow, self.warehouse.numberOfImportAreas, self.warehouse.importAreas, self.warehouse.colourOfImportAreas)
        self.warehouseWindow = Draw_Warehouse.draw_warehouseAreas(self.warehouseWindow, self.warehouse.numberOfStorageAreas, self.warehouse.storageAreas, self.warehouse.colourOfStorageAreas)
        self.warehouseWindow = Draw_Warehouse.draw_warehouseAreas(self.warehouseWindow, self.warehouse.numberOfExportAreas, self.warehouse.exportAreas, self.warehouse.colourOfExportAreas)
        self.warehouseWindow = Draw_Warehouse.draw_warehouseLanes(self.warehouseWindow, self.warehouse.importAreas, self.warehouse.storageAreas, self.warehouse.exportAreas, self.warehouse.colourOfLanes)
        self.warehouseWindow = Draw_Warehouse.draw_chargers(self.warehouseWindow, self.warehouse)
        self.warehouseWindow = Draw_Warehouse.draw_packages(self.warehouseWindow, self.warehouse)
        self.warehouseWindow = Draw_Warehouse.draw_robots(self.warehouseWindow, self.warehouse)

        # Display Windows
        self.show_warehouseWindow = Display_Functions.display_image("show_warehouseWindow", self.warehouseWindow.copy(), False, self.warehouse_windowRes, [0,0])
        self.show_warehouse_chargersInWarehouseWindow = Display_Functions.display_image("chargersInWarehouse", self.warehouse.chargersInWarehouse.copy(), True, self.warehouse_chargersInWarehouse_windowRes, self.warehouse_chargersInWarehouse_windowPosition)
        self.show_warehouse_packagesInWarehouseWindow = Display_Functions.display_image("packagesInWarehouse", self.warehouse.packagesInWarehouse.copy(), True, self.warehouse_packagesInWarehouse_windowRes, self.warehouse_packagesInWarehouse_windowPosition)
        self.show_warehouse_robotsInWarehouseWindow = Display_Functions.display_image("robotsInWarehouse", self.warehouse.robotsInWarehouse.copy(), True, self.warehouse_robotsInWarehouse_windowRes, self.warehouse_robotsInWarehouse_windowPosition)
        self.show_warehouse_packageTargetsInWarehouseWindow = Display_Functions.display_image("packageTargetsInWarehouse", self.warehouse.packageTargetsInWarehouse.copy(), True, self.warehouse_packageTargetsInWarehouse_windowRes, self.warehouse_packageTargetsInWarehouse_windowPosition)

        cv2.waitKey(1)
        return self

    def gameEnd(self):
        self.exit = True

if __name__ == '__main__':

    mainGame = MainGame()
    loop_count = 0

    print('-- Game Loop Start --')
    while not mainGame.exit:
        mainGame.gameLoop(loop_count)
        loop_count += 1
    print('-- Game Loop End --')

    time.sleep(10000)
    cv2.destroyAllWindows()