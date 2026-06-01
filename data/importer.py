"""CSV data loaders for item catalogues and address lists."""
import csv
from data.item import Item
from data.address import Address

# Expected column counts (used for malformed-row validation).
_ITEM_COLUMNS = 6
_ADDRESS_COLUMNS = 4


class Importer:
    """Reads CSV files and converts rows into Item / Address objects."""
    @staticmethod
    def init_import_csv_as_list(csvPath):
        """Read a CSV file and return a list-of-lists (one per row, skipping the header).

        Raises:
            FileNotFoundError: if ``csvPath`` does not exist.
            ValueError: if the file is empty (no header row).
        """
        rows = []
        try:
            with open(csvPath, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                try:
                    next(reader)  # discard header
                except StopIteration:
                    raise ValueError(f"CSV file is empty (no header): {csvPath}")
                for row in reader:
                    rows.append(list(row))
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csvPath}")
        return rows

    @staticmethod
    def init_objectify_items_list(itemsList):
        """Convert raw CSV rows into a dict of {name: Item} objects.

        Raises:
            ValueError: if any row does not have exactly ``_ITEM_COLUMNS`` fields.
        """
        newItemsList = {}
        for rowNumber, row in enumerate(itemsList, start=1):
            if len(row) != _ITEM_COLUMNS:
                raise ValueError(
                    f"Malformed item row {rowNumber}: expected {_ITEM_COLUMNS} "
                    f"columns, got {len(row)}: {row!r}"
                )
            newItemsList[row[0]] = Item(row[0], row[1], row[2], row[3], row[4], row[5])
        return newItemsList

    @staticmethod
    def init_objectify_addresses_list(addressesList):
        """Convert raw CSV rows into a dict of {street: Address} objects.

        Raises:
            ValueError: if any row does not have exactly ``_ADDRESS_COLUMNS`` fields.
        """
        newAddressesList = {}
        for rowNumber, row in enumerate(addressesList, start=1):
            if len(row) != _ADDRESS_COLUMNS:
                raise ValueError(
                    f"Malformed address row {rowNumber}: expected {_ADDRESS_COLUMNS} "
                    f"columns, got {len(row)}: {row!r}"
                )
            newAddressesList[row[0]] = Address(row[0], row[1], row[2], row[3])
        return newAddressesList
