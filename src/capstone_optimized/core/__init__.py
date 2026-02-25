# src/capstone_optimized/core/__init__.py

from .graph_x import Graph
from .polya_x import Polya_Process
from .switched_growth import Switch_Network_Growth
from .sg_opt import Dim_Optimized_Switch_Network_Growth
from .sn_opt_graph import Switched_Network_Optimized_Graph
from .B_A_growth import Barabasi_Albert_Growth
from .sn_node_growth_intervention import Switch_Network_Node_Growth_Intervention
from .sn_injection_intervention import Switch_Network_Injection_Intervention
from .polya_animator import Polya_Process_Animator
from ..cupy_fallback import cp, CUPY_AVAILABLE

# Optional: define __all__ for cleaner import *
__all__ = ["Graph", "Polya_Process", "Switch_Network_Growth", "Dim_Optimized_Switch_Network_Growth", "Switched_Network_Optimized_Graph", "Barabasi_Albert_Growth", "Polya_Process_Animator", "Switch_Network_Node_Growth_Intervention", "Switch_Network_Injection_Intervention", "cp", "CUPY_AVAILABLE"]
