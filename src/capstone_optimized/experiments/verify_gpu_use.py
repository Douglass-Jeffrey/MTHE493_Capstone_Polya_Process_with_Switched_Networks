import sys
import os
import time

# Ensure repo 'src' is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

print('Running verify_gpu_use.py')

from capstone_optimized.core import Graph

try:
    import cupy as cp
except Exception as e:
    print('CuPy import failed:', e)
    cp = None

if cp is None:
    print('CuPy not available — cannot verify GPU use')
    raise SystemExit(1)

print('cupy version:', cp.__version__)
print('cuda available:', cp.cuda.is_available())
try:
    dev_count = cp.cuda.runtime.getDeviceCount()
except Exception:
    dev_count = 0
print('device_count =', dev_count)
if dev_count:
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print('device[0].name =', props.get('name', props.get(b'name', b'')).decode() if isinstance(props.get('name'), (bytes, bytearray)) else props.get('name'))
    except Exception:
        pass

# Create a tiny Graph requesting GPU and inspect types
g = Graph(num_nodes=10, use_gpu=True)
print('Graph.use_gpu =', g.use_gpu)
node_urns_type = type(g.node_urns)
print('node_urns type =', node_urns_type)
print('node_urns module =', node_urns_type.__module__)

# Perform a timed GPU operation (sum of elementwise product) using cupy events
N = 5_000_000
print(f'Running GPU op with N={N} (this allocates ~{N*4/1024/1024:.1f} MB with float32)')
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
# elementwise multiply and sum
res = cp.sum(a * b)
end.record()
end.synchronize()
elapsed_ms = cp.cuda.get_elapsed_time(start, end)
print(f'GPU op elapsed (ms): {elapsed_ms:.3f}')
print('sample result (first 3 digits):', float(res) if hasattr(res, 'item') else res)

# Also show that arrays live on GPU by checking attribute
try:
    print('a device:', a.device.id)
except Exception:
    pass

print('GPU verification complete')
