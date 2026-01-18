# Memory Layout  
*A foundational concept for performance, vectorization, cache behavior, and GPU coalescing*

Memory layout refers to **how data is physically arranged in memory**.  
It directly affects:

- Cache efficiency  
- SIMD/SIMT performance  
- Memory bandwidth usage  
- Pointer aliasing behavior  
- CPU/GPU interoperability  
- Predictability of access patterns  

On Jetson, memory layout is one of the biggest levers for real performance gains.

---

# 1. Two Fundamental Layout Styles

## 1.1 Array of Structures (AoS)

```c
struct Point { float x, y, z; };
Point pts[N];
```
- Characteristics
    - Each element is a full object
    - Fields of the same type are far apart in memory
    - Great for object‑oriented code
    - Poor for vectorization and GPU access
- Problems
    - CPU SIMD loads cannot grab x values contiguously
    - GPU threads accessing pts[i].x cause uncoalesced loads
    - Cache lines contain unused fields
## 1.2 Structure of Arrays (SoA)
float x[N];
float y[N];
float z[N];
- Characteristics
    - Each field is stored in its own contiguous array
    - Perfect for SIMD and SIMT
    - Cache lines contain only relevant data
    - Enables coalesced GPU memory access
# Why Memory Layout Matters for Performance

## 2.1 Cache Locality
- CPUs fetch memory in cache lines (typically 64 bytes).
- A good layout ensures:
    - Spatial locality
    - Fewer cache misses
    - Better prefetching
- AoS often wastes cache space.
- SoA packs useful data tightly.
## 2.2 SIMD Vectorization (CPU)
- NEON loads 128 bits (4 floats) at once.
- AOS
x0 y0 z0 | x1 y1 z1 | x2 y2 z2 ...
NEON cannot load x0 x1 x2 x3 in one instruction.

SoA
x0 x1 x2 x3 x4 x5 ...
Perfect for vector loads.

## 2.3 SIMT Coalescing (GPU)
A warp of 32 threads wants to load:x[i], x[i+1], x[i+2], ...
With SoA → one 128‑byte transaction  
With AoS → 32 scattered loads → bandwidth collapse

## AoSoA (Array of Structures of Arrays)
 - Used in HPC and graphics.
 ```
struct Block {
    float x[16];
    float y[16];
    float z[16];
};
Block blocks[N/16];
```
- Benefits
    - Vectorizable
    - Cache-friendly
### One-Sentence Summary
- Memory layout determines how efficiently CPUs and GPUs can access data — SoA enables vectorization, coalescing, and cache-friendly behavior, while AoS is easier to code but often much slower.


- GPUs handle “wrapping” in thread indexing very differently from CPUs — and the key point is this:
- GPUs do NOT automatically wrap thread indices.
- If an index goes out of range, it’s the programmer’s responsibility to clamp, wrap, or guard it.

