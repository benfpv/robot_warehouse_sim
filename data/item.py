"""Item catalogue data class."""


class Item:
    """A physical item contained in a package (name, labels, weight, dimensions)."""
    def __init__(self, name, labels, description, dateOfManufacture, dimensions, weight):
        self.name = name
        self.labels = labels
        self.description = description
        self.dateOfManufacture = dateOfManufacture
        self.dimensions = dimensions
        self.weight = weight