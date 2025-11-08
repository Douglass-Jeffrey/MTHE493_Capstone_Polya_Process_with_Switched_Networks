import random

class Urn:
    def __init__(self):
        self._contents = {}

    # --------------------
    # Property definitions
    # --------------------
    @property
    def contents(self):
        """Return a copy of the urn's contents to prevent accidental external modification."""
        return dict(self._contents)

    @contents.setter
    def contents(self, value):
        """Allow replacing the entire contents dictionary, with type checking."""
        if not isinstance(value, dict):
            raise TypeError("contents must be a dictionary mapping item_id -> quantity")
        for qty in value.values():
            if not isinstance(qty, int) or qty < 0:
                raise ValueError("All quantities must be non-negative integers")
        self._contents = value

    # --------------------
    # Urn operations
    # --------------------
    def add_item(self, item_id, quantity):
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("quantity must be a non-negative integer")
        if item_id in self._contents:
            self._contents[item_id] += quantity
        else:
            self._contents[item_id] = quantity

    def remove_item(self, item_id, quantity):
        if item_id not in self._contents:
            raise KeyError(f"Item '{item_id}' not found in urn")
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("quantity must be a non-negative integer")
        if self._contents[item_id] < quantity:
            raise ValueError("Not enough quantity to remove")
        self._contents[item_id] -= quantity
        if self._contents[item_id] == 0:
            del self._contents[item_id]

    def get_item(self, item_id):
        if item_id not in self._contents:
            raise KeyError(f"Item '{item_id}' not found in urn")
        return self._contents[item_id]

    def choose_random_item(self):
        if not self._contents:
            raise ValueError("Urn is empty")
        items = list(self._contents.keys())
        weights = list(self._contents.values())
        return random.choices(items, weights=weights, k=1)[0]
