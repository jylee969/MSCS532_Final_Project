import time
import numpy as np

def run_benchmark():
    # Experimental Setup: Define dataset parameters
    N = 10_000  # Number of spatial coordinates (simulating N elements/mesh nodes)
    K = 100     # Dimension of the invariant transformation matrix (K x K)
    
    # Seed the random number generator for reproducible benchmark results
    np.random.seed(42)
    
    # Generate mock spatial dataset arrays (x, y coordinates) and a scaling matrix
    data_x = np.random.rand(N)
    data_y = np.random.rand(N)
    scaling_matrix = np.random.rand(K, K)

    print(f"==================================================")
    print(f"HPC Optimization Benchmark: N={N}, Matrix={K}x{K}")
    print(f"==================================================\n")

    # -------------------------------------------------------------
    # 1. Unoptimized: Redundant invariant Computation inside Loop
    # -------------------------------------------------------------
    def compute_unoptimized(x, y, matrix):
        results = []
        # Traversing coordinates element-by-element using a standard Python loop
        for i in range(len(x)):
            # REDUNDANT COMPUTATION:
            # The matrix transpose and product (@) yield the exact same value on 
            # every iteration, but are re-calculated N times inside the loop
            # Time complexity per iteration: O(K^3), scaling overall execution to O(N * K^3)
            transposed_norm = np.linalg.norm(matrix.T @ matrix)
            
            # Compute point transformation with Python interpreter overhead
            val = (x[i] ** 2 + y[i] ** 2) * transposed_norm
            results.append(val)
            
        return np.array(results)

    # Time the unoptimized implementation
    start_time = time.perf_counter()
    res_unoptimized = compute_unoptimized(data_x, data_y, scaling_matrix)
    time_unoptimized = time.perf_counter() - start_time
    print(f"[Unoptimized] Execution Time: {time_unoptimized:.4f} seconds")

    # -------------------------------------------------------------
    # 2. Optimized: Hoisted Invariant + Vectorized Execution
    # -------------------------------------------------------------
    def compute_optimized(x, y, matrix):
        # HOISTED INVARIANT (Optimization via Loop-Invariant Code Motion):
        # Calculate the invariant matrix norm ONCE outside the processing loop
        # This reduces the matrix calculation cost from O(N * K^3) down to O(K^3)
        transposed_norm = np.linalg.norm(matrix.T @ matrix)
        
        # VECTORIZED COMPUTATION:
        # Bypasses Python interpreter loops entirely by using contiguous memory 
        # operations in compiled NumPy routines (SIMD execution)
        # Time complexity for array traversal: O(N), yielding total complexity O(K^3 + N)
        results = (x**2 + y**2) * transposed_norm
        return results

    # Time the optimized implementation
    start_time = time.perf_counter()
    res_optimized = compute_optimized(data_x, data_y, scaling_matrix)
    time_optimized = time.perf_counter() - start_time
    print(f"[Optimized]   Execution Time: {time_optimized:.4f} seconds")

    # Verification: Ensure numerical output matches between both implementations
    assert np.allclose(res_unoptimized, res_optimized), "Validation Failure!"
    
    # Calculate performance gains
    speedup = time_unoptimized / time_optimized
    print(f"\n=> Results Verified Correct.")
    print(f"=> Overall Speedup Factor: {speedup:.2f}x\n")

if __name__ == "__main__":
    run_benchmark()