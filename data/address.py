"""Shipping address data class."""


class Address:
    """Postal address attached to a package origin or destination."""
    def __init__(self, street_address, city, province, zipcode):
        self.street_address = street_address
        self.city = city
        self.province = province
        self.zipcode = zipcode