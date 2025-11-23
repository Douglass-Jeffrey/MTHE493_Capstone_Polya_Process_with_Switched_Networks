# src/capstone_optimized/core/__init__.py

from .graph_x import Graph
from .polya_x import Polya_Process
from .switched_growth import Switch_Network_Growth
from .sg_opt import Dim_Optimized_Switch_Network_Growth
from .sn_opt_graph import Switched_Network_Optimized_Graph
from ..cupy_fallback import cp, CUPY_AVAILABLE

# Optional: define __all__ for cleaner import *
__all__ = ["Graph", "Polya_Process", "Switch_Network_Growth", "Dim_Optimized_Switch_Network_Growth", "Switched_Network_Optimized_Graph", "cp", "CUPY_AVAILABLE"]
