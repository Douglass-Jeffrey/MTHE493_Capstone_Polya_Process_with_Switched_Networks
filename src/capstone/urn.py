import random

class urn:
    def __init__ (self):
        self.contents = {}
    
    def add_item(self, item_id, quantity):
        if item_id in self.contents:
            self.contents[item_id] += quantity
        else:
            self.contents[item_id] = quantity
    
    def remove_item(self, item_id, quantity):
        if item_id in self.contents:
            if self.contents[item_id] >= quantity:
                self.contents[item_id] -= quantity
                if self.contents[item_id] == 0:
                    del self.contents[item_id]
            else:
                raise ValueError("Not enough quantity to remove")
        else:
            raise KeyError("Item not found in urn")
    
    def get_contents(self, item_id=None):
        return (self.contents[item_id] if (item_id != None) else self.contents)

    def draw_item (self):
        total_items = sum(self.contents.values())
        if total_items == 0:
            raise ValueError("Urn is empty")
        
        rand_choice = random.randint(1, total_items)
        cumulative = 0
        for item_id, quantity in self.contents.items():
            cumulative += quantity
            if rand_choice <= cumulative:
                return item_id