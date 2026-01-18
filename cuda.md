# CUDA – Basic Study Notes  
*A beginner-friendly overview of CUDA concepts for parallel programming*

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform that allows developers to write programs that run on the GPU. It enables massive parallelism using thousands of lightweight threads.

---

# 1. GPU vs CPU: The Core Idea

## CPU
- Few powerful cores  
- Optimized for sequential tasks  
- Large caches, complex control logic  

## GPU
- Hundreds to thousands of simple cores  
- Optimized for parallel workloads  
- High memory bandwidth  
- Executes many threads simultaneously  

**CUDA lets you write code that uses this parallel hardware.**

---

# 2. CUDA Programming Model

CUDA organizes work into a hierarchy:

```
Grid → Blocks → Threads
```

### Thread
- Smallest execution unit  
- Runs a single instance of the kernel  

### Block
- Group of threads  
- Can synchronize using `__syncthreads()`  
- Shares fast on-chip **shared memory**  

### Grid
- Collection of blocks  
- Represents the entire kernel launch  

---

# 3. Kernels

A CUDA kernel is a function that runs on the GPU.

```c
__global__ void add(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```
- Launched from the CPU (host):
- add<<<gridSize, blockSize>>>(a, b, c);

# 4. Thread Indexing
- CUDA provides built-in variables:
    - threadIdx.x
    - blockIdx.x
    - blockDim.x
    - gridDim.x
- Global index:
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
- Always guard against out-of-range access:
- if (idx < N) { ... }

# 5. Memory Types in CUDA
### 5.1 Global Memory
    - Large but slow
    - Accessible by all threads
    - Must be accessed carefully (coalescing)

### 5.2 Shared Memory
    - Fast on-chip memory
    - Shared within a block
    - Great for tiling and caching
### 5.3 Registers
    - Fastest memory
    - Private to each thread
### 5.4 Constant & Texture Memory
    - Read-only cached memory
    - Useful for lookup tables and images
# 6. Memory Coalescing
- GPUs are fastest when threads in a warp access contiguous memory.
- Example of coalesced access:
- a[idx], a[idx+1], a[idx+2] ...
- Uncoalesced access causes:
    - Many small memory transactions
    - Lower bandwidth
    - Slow kernels
# 7. Warp and SIMT Execution
    - A warp = 32 threads executing in lockstep
    - GPU uses SIMT (Single Instruction, Multiple Threads)
    - Divergent branches inside a warp cause serialization
- Avoid:
    - if (threadIdx.x % 2 == 0) { ... }
    - else { ... }
# 8. Synchronization
- Within a block:
  - __syncthreads();
  - Across blocks:
    - No direct synchronization
    - Must use multiple kernel launches or atomics
# 9. Atomics
- Used for safe updates to shared/global memory:
  - atomicAdd(&counter, 1);
  - Useful for:
      - Reductions
      - Histograms
      - Counters
# 10. Performance Tips (Beginner Level)
    - Use SoA instead of AoS
    - Ensure coalesced memory access
    - Avoid warp divergence
    - Use shared memory for reuse
    - Keep kernels simple and parallel
    - Use grid-stride loops for large arrays

# 11. Typical CUDA Workflow
    - Allocate memory on GPU (cudaMalloc)
    - Copy data from CPU → GPU (cudaMemcpy)
    - Launch kernel
    - Copy results GPU → CPU
    - Free GPU memory

# 12. Example: Vector Addition
```
__global__ void vecAdd(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}
Launch:
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;
vecAdd<<<gridSize, blockSize>>>(a, b, c, N);

```

# Summary
### CUDA enables:
    - Massive parallelism
    - Thousands of threads
    - High memory bandwidth
    - Efficient data-parallel programming
### Key concepts:
    - Kernels
    - Threads, blocks, grids
    - Memory hierarchy
    - Coalescing
    - Warps and SIMT
    - Synchronization and atomics
- Mastering these basics sets the foundation for advanced GPU programming.


# CUDA Cheat Sheet  
*A fast reference for essential CUDA concepts, syntax, and performance rules*

---

# 1. CUDA Execution Model

## Thread hierarchy
- Grid → Blocks → Threads

### Built‑in variables
- `threadIdx.x`, `.y`, `.z`
- `blockIdx.x`, `.y`, `.z`
- `blockDim.x`, `.y`, `.z`
- `gridDim.x`, `.y`, `.z`

### Global thread index (1D)
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
Guarding out‑of‑range

- if (idx >= N) return;
# 2. Kernel Basics

- Defining a kernel
```
__global__ void kernel(args...) {
    // GPU code
}


- Launching a kernel
  - kernel<<<gridSize, blockSize>>>(args...);
- Grid‑stride loop (best practice)
  ```
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    ...
}
```
## 3. Memory Types

| Memory Type     | Scope        | Speed     | Notes               |
|-----------------|--------------|-----------|---------------------|
| Registers       | Thread       | Fastest   | Limited per thread  |
| Shared Memory   | Block        | Very fast | Manual caching      |
| Global Memory   | All threads  | Slow      | Must coalesce       |
| Constant Memory | All threads  | Cached    | Read‑only           |
| Texture Memory  | All threads  | Cached    | Good for images     |
```

# 4. Memory Coalescing Rules

- ✔ Threads in a warp should access contiguous addresses
- ✔ Use Structure of Arrays (SoA)  
- ✔ Align data (4, 8, 16 bytes)
- ✔ Avoid strided or random access
- ✔ Prefer float4, int4 vector loads

# 5. Warp & SIMT Execution
  - Warp = 32 threads
  - Execute in lockstep
  - Divergence = slow
# Avoid warp divergence
```
if (threadIdx.x % 2 == 0) { ... }  // BAD

Warp sync
__syncwarp(); // GOOD 
```


# 6. Synchronization
- Block-level sync
- __syncthreads();
- Warp-level sync
- __syncwarp();

- No grid-wide sync inside a kernel
- Use multiple kernel launches.

# 7. Atomics
- Useful for counters, reductions, histograms.
- atomicAdd(&x, 1);
- atomicCAS(&ptr, old, new);

# 8. Shared Memory
- Declaring shared memory
- __shared__ float tile[256];
- Dynamic shared memory
- extern __shared__ float buf[];

## Use shared memory for:
    - Tiling
    - Reuse
    - Avoiding repeated global loads

# 9. Performance Checklist
- ✔ Memory
    - Use SoA
    - Coalesce global memory
    - Use shared memory for reuse
    - Avoid bank conflicts

- ✔ Threads & Blocks
    - Use 128–1024 threads per block
    - Keep occupancy high
    - Use grid‑stride loops
- ✔ Control Flow
    - Avoid divergence
    - Use warp‑level primitives
- ✔ Math
    - Prefer fused operations (fmaf)
    - Use fast math (__sinf, __expf) when acceptable
# 10. Common Kernel Patterns
- Vector add
  - c[idx] = a[idx] + b[idx];
- Reduction (warp-level)
  - val += __shfl_down_sync(0xffffffff, val, 16);
- Tiled matrix multiply (shared memory)
  - __shared__ float As[T][T], Bs[T][T];
# 11. Error Checking
```
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

```
# 12. Host–Device Memory Management
```
cudaMalloc(&d_ptr, size);
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
cudaFree(d_ptr);

```
- Summary
# CUDA fundamentals:
    - Threads → Blocks → Grids
    - Coalesced memory = fast
    - Shared memory = manual cache
    - Warps execute in lockstep
    - Avoid divergence
    - Use atomics sparingly
    - Optimize for bandwidth and parallelism





# Lock‑Free Concepts — Detailed Study Notes  
*A deep dive into non‑blocking concurrency for high‑performance systems*

Lock‑free programming avoids traditional locking mechanisms (mutexes, critical sections, semaphores) and instead uses atomic operations and memory‑ordering guarantees to ensure progress without blocking.  
This is essential for real‑time systems, multi‑core CPUs, and GPUs.

---

# 1. What “Lock‑Free” Means

A system is **lock‑free** if:

> At least one thread always makes progress, even if others are paused or delayed.

### Levels of non‑blocking progress

| Type              | Guarantee |
|-------------------|-----------|
| **Wait‑free**     | Every thread makes progress in bounded time |
| **Lock‑free**     | At least one thread always makes progress |
| **Obstruction‑free** | A thread makes progress if it runs alone |

Lock‑free is the practical middle ground for high‑performance systems.

---

# 2. Atomic Operations — The Foundation

Lock‑free algorithms rely on **atomic read‑modify‑write** instructions.

### Most important: Compare‑and‑Swap (CAS)

```c
bool atomicCAS(addr, expected, newValue);
```
- Meaning:
    - If *addr == expected, replace it with newValue
    - Otherwise, do nothing
    - Returns success/failure
- Other atomic operations
    - atomicAdd
    - atomicExch
    - atomicMin
    - atomicMax
    - atomicInc
    - atomicDec
- Atomics allow safe concurrent updates without locks.
- 
