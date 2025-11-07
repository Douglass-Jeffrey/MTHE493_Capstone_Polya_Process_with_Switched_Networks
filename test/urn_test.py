import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.urn import urn

if __name__ == "__main__":
    
    urn = urn()
    urn.add_item("red", 5)
    urn.add_item("blue", 3)
    urn.add_item("green", 2)

    print ("Initial urn contents:", urn.get_contents().keys())