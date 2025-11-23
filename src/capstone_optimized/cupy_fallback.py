try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    # Check whether CuPy actually has access to CUDA devices
    try:
        has_cuda = cp.cuda.is_available()
        dev_count = cp.cuda.runtime.getDeviceCount() if has_cuda else 0
    except Exception:
        has_cuda = False
        dev_count = 0

    if has_cuda and dev_count > 0:
        CUPY_AVAILABLE = True
        print(f"using CuPy (devices={dev_count})")
    else:
        # Fall back to NumPy if CuPy can't access a CUDA device
        import numpy as cp
        from scipy.sparse import csr_matrix as cpu_csr_matrix
        CUPY_AVAILABLE = False
        print("CuPy present but no CUDA device available; falling back to NumPy")
except ImportError:
    import numpy as cp
    from scipy.sparse import csr_matrix as cpu_csr_matrix
    CUPY_AVAILABLE = False
    print("using NumPy (CuPy not installed)")
