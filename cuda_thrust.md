# Thrust Library — CUDA STL‑Style Parallel Algorithms
```cpp
+-------------------------------------------------------------+
|                     Application Code                        |
|  - Your C++/CUDA program                                    |
|  - Calls high-level algorithms (Thrust, custom kernels)     |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|                         Thrust                              |
|  - STL-style C++ template library                           |
|  - High-level parallel algorithms:                          |
|      * sort, reduce, transform, scan                        |
|  - Containers: host_vector, device_vector                   |
|  - Fancy iterators: counting, transform, zip, permutation   |
|  - Can target CUDA, TBB, OpenMP backends                    |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|                     Other CUDA Libraries                    |
|  - CUB: low-level, highly tuned primitives                  |
|      * block/warp-level scan, reduce, radix sort            |
|  - cuBLAS, cuDNN, cuSPARSE, etc.                            |
|      * domain-specific math / ML / linear algebra           |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|                     CUDA Runtime API                        |
|  - Kernel launch syntax <<<grid, block>>>                   |
|  - Memory management (cudaMalloc, cudaMemcpy, cudaFree)     |
|  - Streams, events, error handling                          |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|                     CUDA Driver API                         |
|  - Low-level control of contexts, modules, functions        |
|  - JIT compilation, explicit device management              |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|                        GPU Hardware                         |
|  - SMs, warps, threads                                      |
|  - Registers, shared memory, global memory                  |
|  - Execution of compiled kernels                            |
+-------------------------------------------------------------+

```

Thrust is CUDA’s high‑level, STL‑style parallel algorithms library — designed to let you write GPU code using familiar C++ patterns like `sort`, `reduce`, `transform`, and `scan`, without manually managing kernels or thread blocks.

---

## What Thrust Is (Core Idea)

Thrust is a **C++ template library** that provides **parallel algorithms and data structures** for CUDA.  
It looks and feels like the C++ STL, but executes operations on the GPU with highly optimized CUDA kernels.

It dramatically reduces boilerplate: instead of writing custom kernels for sorting, scanning, or reductions, you call a single function like:

```cpp
thrust::sort();
```
## 2. Key Components of Thrust
### 2.1 Device & Host Vectors

- Thrust provides two STL‑like containers:

```cpp
thrust::host_vector<T>    // stored in CPU memory
thrust::device_vector<T>  // stored in GPU memory
```
They behave like std::vector, support dynamic resizing, and can be copied between host and device using assignment:

```cpp
thrust::device_vector<int> d = h;
```
### 2.2 Parallel Algorithms

- Thrust includes GPU‑accelerated versions of common STL algorithms:
```cpp
thrust::sort      // massively parallel sort
thrust::reduce    // sum or combine values
thrust::transform // element‑wise operations
thrust::scan      // prefix sums
thrust::copy
thrust::fill
thrust::generate
```
- These algorithms automatically dispatch optimized CUDA kernels.
### 2.3 Fancy Iterators

- Thrust provides powerful iterator types that let you build complex operations without writing kernels:
```cpp
thrust::counting_iterator
thrust::transform_iterator
thrust::zip_iterator
thrust::permutation_iterator
```
- These allow creation of virtual sequences, combining multiple vectors, or applying transformations on‑the‑fly.
### 2.4 Backend Flexibility

- Thrust can run on multiple backends:
```
CUDA   — GPU execution
TBB    — Intel Threading Building Blocks
OpenMP — CPU parallelism
```
## 3. Minimal Example: Sorting on GPU
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
    thrust::host_vector<int> h = {4, 2, 7, 1};
    thrust::device_vector<int> d = h;

    thrust::sort(d.begin(), d.end());

    thrust::copy(d.begin(), d.end(), h.begin());
}
```
- It dramatically simplifies GPU programming by removing the need to write custom kernels for common operations.

