Experimental, trying to optimize graph and processes for high node counts, extremely large graphs.
At this level we need everything fully vectorized so we want to use Sparse matrix architectures
Also want to replace slow python data structures with numpy, scipy.
proposed structure:

sparse matrices to represent graph adjacencies of n urns  (nxn)
vectorized urns (1xn)
sparse matrix for super urns

consider graph matrix structure with switched network growth
                            # did node (row) hit switch cond?
[1, 0, 1, 0, 0, 0, 1, 0;    # generated as 0
 0, 1, 1, 0, 0, 0, 1, 0;    # generated as 0
 1, 1, 1, 0, 0, 0, 1, 0;    # connects to all prev
 0, 0, 0, 1, 0, 0, 1, 0;    # generated as 0
 0, 0, 0, 0, 1, 0, 1, 0;    # generated as 0
 0, 0, 0, 0, 0, 1, 1, 0;    # generated as 0
 1, 1, 1, 1, 1, 1, 1, 0;    # connects to all prev
 0, 0, 0, 0, 0, 0, 0, 1;]   # generated as 0

diagonal all 1 since self connections hold for mega
urn calcs consider that once a sw node n spawns, all
previous nodes (rows) need their column index
updated at column n. Other than CSR is there a
way to improve this process or compress it to 
make it faster?

List of improvements:
    - GPU support with cuda and cupy with scipy fallback
    - Fully vectorized representations of urns, graphs, processes
    - Implemented batch growth
    - Impremented CSR graph class, which is generally pretty good for directed graphs
    - Implemented switched network graph class which is hyper optimized
      This is because we dont have to store a NxN adjacency for the switched 
      network growth, we just need a Nx1 array of bits where the 'n'th bit equalling 0
      means that the bernoulli trial succeeded, all prev nodes should be connected to node n
      (Really proud of this one, it was a crazy epiphany)

Next steps:
    - Create new visualizer for processes
    - Add support for other random trials other than bernoulli in growth

    - Look into ways to further reduce memory for the graph, one might be 
      not using int32 for the edges in the CSR since technically we just need
      one bit (or bool?) to represent that there is an edge there.
      There are some caveats here though i think cp.csr cant be a bit or bool
      matrix for some reason need to look further into it.

    - Hybridize GPU and CPU operations and storage for the graph.
      Ie since the vram on my 1660TI is too small (6GB) we slow down
      massively on large graphs with lots of edges and run into
      occasional malloc errors when we try to construct CSR.
    
    - We want the best of both worlds, current workload is good when
      the total memory of the graph is less than 6GB of vram,
      but when it is greater, the GPU ends up wasting all its time
      writing to unified memory in RAM so we slow way down (like 100x).
      The CPU actually outperforms the GPU in graph growth in these cases,
      and the GPU crashes before we can even run the polya (which i still have to fix).
    
    - GPU is exceptional at polya compared to cpu for cases where were
      just under 6gb of vram, so we want to leverage that. I want to
      find a way where i can use the GPU for the stuff that it's good at,
      and leave the other stuff to the CPU.

