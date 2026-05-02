"""CSV data loaders for item catalogues and address lists."""
import csv
from data.item import Item
from data.address import Address


class Importer:
    """Reads CSV files and converts rows into Item / Address objects."""
    @staticmethod
    def init_import_csv_as_list(csvPath):
        """Read a CSV file and return a list-of-lists (one per row, skipping the header)."""
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

    @staticmethod
    def init_objectify_items_list(itemsList):
        """Convert raw CSV rows into a dict of {name: Item} objects."""
        newItemsList = {} 
        for row in itemsList:
            newItemsList[row[0]]=(Item(row[0], row[1], row[2], row[3], row[4], row[5])) 
        return newItemsList
    
    @staticmethod
    def init_objectify_addresses_list(itemsList):
        """Convert raw CSV rows into a dict of {street: Address} objects."""
        newAddressesList = {} 
        for row in itemsList:
            newAddressesList[row[0]]=(Address(row[0], row[1], row[2], row[3]))
        return newAddressesList