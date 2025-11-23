import cupy as cp
import sys

print(f"Python Executable: {sys.executable}")

# 1. Check CuPy Package Version
print(f"CuPy Version: {cp.__version__}")

# 2. Check CUDA Runtime Version (Correct function for CuPy 13+)
# This returns an integer (e.g., 12000 means 12.0)
try:
    runtime_ver = cp.cuda.runtime.runtimeGetVersion()
    print(f"CUDA Runtime Version: {runtime_ver}")
except Exception as e:
    print(f"Error getting runtime version: {e}")

# 3. Check GPU Compute Capability
try:
    device = cp.cuda.Device(0)
    print(f"Device Name: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
    print(f"Compute Capability: {device.compute_capability}")
    
    # 4. Create an array on the GPU
    x = cp.array([1.0, 2.0, 3.0])
    print(f"Success! Array created on: {x.device}")
    print(f"Result of x * 2: {x * 2}")

except Exception as e:
    print(f"\nGPU Access Failed: {e}")
    print("Double check your CUDA_PATH environment variable if this fails.")