import csv
from data.item import *
from data.address import *

class Importer:
    def init_import_csv_as_list(csvPath):
        print('Importer.init_import_csv_as_list() ')
        # import list_items.csv as a list of items; row = item, col = item characteristics
        print('- csvPath: {}'.format(csvPath))
        itemsList = [] 
        with open(csvPath, newline='') as csvfile:
            itemsList_csv = csv.reader(csvfile, delimiter=',') 
            next(itemsList_csv)
            row_count = 0
            for row in itemsList_csv:
                itemsList.append([]) 
                for i in row:
                    itemsList[row_count].append(i) 
                row_count += 1 
        return itemsList

    def init_objectify_items_list(itemsList):
        print('Importer.init_objectify_items_list()')
        newItemsList = {} 
        for row in itemsList:
            newItemsList[row[0]]=(Item(row[0], row[1], row[2], row[3], row[4], row[5])) 
        #print('- Example from Objectified Items List: ' + str(newItemsList['apple'].name))
        return newItemsList
    
    def init_objectify_addresses_list(itemsList):
        print('Importer.init_objectify_addresses_list()')
        newAddressesList = {} 
        for row in itemsList:
            newAddressesList[row[0]]=(Address(row[0], row[1], row[2], row[3]))
        #print('- Example from Objectified Addresses List: ' + str(newAddressesList['10 Rainy Drive'].city))
        return newAddressesList