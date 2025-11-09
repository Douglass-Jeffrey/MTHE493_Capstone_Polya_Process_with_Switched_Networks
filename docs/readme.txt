MTHE 494 Capstone Project

This repository is a framework to generate, modify and visualize graphs while running processes on them.
In particular, the group is concerned with the "Switched Network" process, and the "Polya" process.

Dev Log: 
    2025-11-07: Finished implementing all core features, both polya and switched network processes can be simulated and visualized.

Future Considerations:
    Create additional variants of polya process (memory vs memoryless)
    Create tests on different graphs, new graph growth or creation processes other than switched network

    Allow graphs to be saved and loaded as data structures so that we can run different processes on the same graphs
    Seriously consider graph efficiency updates, at high node and edge counts (nodes>1000, edges> 10000) in graphs polya can take hours or days to complete. Consider existing python graphing libraries.
        Maybe running steps in batches? Maybe only running full polya process every x steps? Maybe sorting graph's nodes based on edge counts?
        For polya there may be ways we can speed up the single step on the graph by approximating the process to some reasonable degree, like if there is a 1:10000 ratio of blue to red automatically add red instead of doing process.
        For polya we might save a ton of time by optimizing the whole mega-urn (super urn?) process. Constructing mega urns for every node and running the picking process on them by FAR takes the bulk of the time, for fully connected nodes its O(n)^2, rest is just getting and updating the nodes which is O(1)
        Look in to increasing efficiency with matrix multiplications using gpu. 
