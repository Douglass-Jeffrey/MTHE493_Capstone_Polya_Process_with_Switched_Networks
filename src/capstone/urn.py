import random
from collections import Counter

class Urn:
    def __init__(self):
        self._contents = Counter()
        self._last_drawn_item = None

    @property
    def contents(self):
        return dict(self._contents)

    @property
    def last_drawn_item(self):
        return self._last_drawn_item
    
    @last_drawn_item.setter
    def last_drawn_item(self, value):
        self._last_drawn_item = value


    @contents.setter
    def contents(self, value):
        if not isinstance(value, dict):
            raise TypeError("contents must be a dictionary mapping item_id -> quantity")
        for qty in value.values():
            if not isinstance(qty, int) or qty < 0:
                raise ValueError("All quantities must be non-negative integers")
        self._contents = value

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