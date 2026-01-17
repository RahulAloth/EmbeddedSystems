# SIMD vs SIMT Execution  
*A CPU vs GPU execution model comparison*

Modern processors use two major parallel execution models: **SIMD** (used by CPUs) and **SIMT** (used by GPUs).  
Both apply “one instruction to many data,” but they differ dramatically in how they scale and how programmers interact with them.

---

# 1. SIMD — Single Instruction, Multiple Data  
**Used by:** CPUs (ARM NEON), DSPs, vector processors

SIMD executes **one instruction** across **multiple data elements** simultaneously using **vector registers**.

## 1.1 How SIMD Works
- CPU loads a vector (e.g., 4 integers)
- Executes one instruction on all lanes at once
- All lanes must follow the same control flow

### Example
```
[1,2,3,4] + [5,6,7,8] → [6,8,10,12]
```

## 1.2 Characteristics
- Fixed-width vector lanes (e.g., 128-bit NEON)
- No divergence allowed
- Great for:
  - Image filtering
  - DSP operations
  - Robotics math (quaternions, transforms)
  - Small-scale parallelism

## 1.3 Jetson Context
ARM CPUs in Jetson (A57, A72, Carmel, A78AE) use **NEON SIMD** for:
- Preprocessing camera frames
- Matrix operations
- Filtering and transforms
- CPU-side robotics workloads

---

# 2. SIMT — Single Instruction, Multiple Threads  
**Used by:** NVIDIA GPUs (CUDA)

SIMT executes **one instruction stream** across **many threads**, grouped into **warps** (32 threads on NVIDIA GPUs).

## 2.1 How SIMT Works
- Each thread has its own registers and program state
- Threads run in groups (warps) that execute in lockstep
- Divergence is allowed but reduces efficiency

### Example

```
thread0: a[0] + b[0]
thread1: a[1] + b[1]
...
thread31: a[31] + b[31]

```

## 2.2 Characteristics
- Thread-based programming model
- Each thread behaves like a tiny independent CPU
- Hardware executes warps in SIMD fashion underneath
- Scales to thousands of threads

## 2.3 Jetson Context
All Jetson GPUs (Nano → Orin) use SIMT for:
- Deep learning inference
- Stereo vision
- Dense optical flow
- Point cloud processing
- Massive pixel/voxel parallelism

---

# 3. SIMD vs SIMT — Key Differences

| Feature | SIMD (CPU NEON) | SIMT (GPU CUDA) |
|--------|------------------|------------------|
| Parallel unit | Vector lanes | Threads in a warp |
| Programming model | Vector operations | Thread-per-element |
| Divergence | Not allowed | Allowed but costly |
| Scale | 4–16 lanes | Thousands of threads |
| Best for | Small parallel tasks | Massive parallel tasks |
| Jetson usage | CPU preprocessing | GPU-heavy workloads |

---

# 4. Why This Matters on Jetson

## CPU (SIMD)
- Low-latency, predictable execution
- Great for robotics control loops
- Efficient for small matrices and filters
- Ideal for preprocessing before GPU kernels

## GPU (SIMT)
- Massive throughput
- Ideal for deep learning and vision
- Hides latency with warp scheduling
- Handles huge workloads efficiently

### Typical Jetson Workflow
1. CPU uses **SIMD NEON** for preprocessing  
2. GPU uses **SIMT CUDA** for inference or vision kernels  
3. CPU uses SIMD again for post-processing or control  

---

# 5. One-Sentence Summary

**SIMD = vector lanes on a CPU.  
SIMT = thousands of threads on a GPU.  
Both apply one instruction to many data, but SIMT is massively scalable and thread-based.**


# SIMD vs SIMT — Mermaid Comparison Diagram

```mermaid
flowchart LR

%% SIMD Side
subgraph A[SIMD (CPU NEON)]
    A1[Single Instruction]
    A2[Vector Register<br>(e.g., 128-bit NEON)]
    A3[Multiple Data Elements<br>(Lanes)]
    A4[Lockstep Execution<br>(No Divergence)]
end

A1 --> A2 --> A3 --> A4

%% SIMT Side
subgraph B[SIMT (GPU CUDA)]
    B1[Single Instruction Stream]
    B2[Warp of Threads<br>(32 threads)]
    B3[Each Thread Has<br>Own Registers & State]
    B4[Divergence Allowed<br>(But Slows Warp)]
end

B1 --> B2 --> B3 --> B4

%% Comparison Arrows
A ---|Vector Lanes| C((Parallelism))
B ---|Many Threads| C

C:::compare

%% Styles
classDef compare fill:#ffe9b3,stroke:#d19a00,stroke-width:2px;
```
# Jetson-Specific SIMD/SIMT Optimization Guide  
*Using CPU NEON (SIMD) and GPU CUDA (SIMT) together effectively*

This guide focuses on how to **combine ARM NEON SIMD on the CPU** and **CUDA SIMT on the GPU** to get the best out of Jetson platforms (Nano, TX2, Xavier, Orin).

---

# 1. When to Use SIMD (CPU NEON) vs SIMT (GPU CUDA)

| Workload Type            | Prefer SIMD (CPU NEON)                      | Prefer SIMT (GPU CUDA)                          |
|--------------------------|---------------------------------------------|-------------------------------------------------|
| Small matrices/vectors   | Yes                                         | Overkill                                       |
| Control loops            | Yes (low latency, predictable)              | No (GPU latency, launch overhead)              |
| Image preprocessing      | Yes (resize, normalize, simple filters)     | Yes, if batched or heavy                       |
| Deep learning inference  | Only for tiny models                        | Yes (TensorRT, large models)                   |
| Stereo, optical flow     | Sometimes (small ROIs)                      | Yes (full-frame, dense)                        |
| Point clouds / voxels    | Limited                                     | Yes (massive parallelism)                      |

---

# 2. SIMD (CPU NEON) Optimization on Jetson

## 2.1 Good Targets for NEON
- **Per-pixel operations** on small images or ROIs  
- **Vector math**: transforms, quaternions, Jacobians  
- **Filtering**: 1D/2D small kernels (e.g., 3×3, 5×5)  
- **Preprocessing** before sending data to GPU  

## 2.2 Practical Guidelines
- **Use SoA (Structure of Arrays)** where possible  
  - Easier to vectorize than AoS  
- **Align data** to 16 bytes for NEON  
- Use compiler auto-vectorization:
  - `-O3 -mfpu=neon -ftree-vectorize` (GCC/Clang flags vary by toolchain)  
- For hot loops:
  - Consider **intrinsics** (`vaddq_f32`, `vmulq_f32`, etc.)  
- Avoid:
  - Branch-heavy code inside NEON loops  
  - Misaligned loads/stores  

---

# 3. SIMT (GPU CUDA) Optimization on Jetson

## 3.1 Good Targets for CUDA SIMT
- Full-frame image processing (720p, 1080p, 4K)  
- Deep learning inference (TensorRT)  
- Stereo disparity, optical flow, feature extraction  
- Point cloud processing, voxel grids, occupancy maps  

## 3.2 Practical Guidelines
- **Thread-per-element** mapping:
  - One thread per pixel, feature, point, etc.  
- Ensure **coalesced memory access**:
  - Adjacent threads → adjacent memory addresses  
- Use **shared memory** for:
  - Tiles, windows, convolution blocks  
- Avoid **warp divergence**:
  - Minimize `if/else` inside warps  
- Use **streams** to overlap:
  - Transfers  
  - Kernels  
  - Pre/post-processing  

---

# 4. Combining SIMD + SIMT in Real Pipelines

## 4.1 Typical Jetson Vision Pipeline

1. **Capture**
   - Camera → NVMM (zero-copy)  
2. **CPU NEON Preprocessing (SIMD)**
   - Light transforms, normalization, small filters  
   - Only if it’s cheaper than sending to GPU  
3. **GPU CUDA / TensorRT (SIMT)**
   - Heavy lifting: inference, dense vision, stereo, flow  
4. **CPU NEON Post-processing**
   - Small reductions, formatting, control decisions  

### Rule of Thumb
- **CPU NEON**: low-latency, small/medium work  
- **GPU CUDA**: high-throughput, large/batched work  

---

# 5. Data Layout and Movement

## 5.1 For SIMD (NEON)
- Prefer **contiguous arrays** (SoA)  
- Align to 16 bytes  
- Avoid frequent heap allocations in hot paths  

## 5.2 For SIMT (CUDA)
- Keep data **resident on GPU** as long as possible  
- Avoid CPU↔GPU ping-pong  
- Use:
  - Pinned memory for transfers  
  - Zero-copy for camera frames (NVMM)  
  - TensorRT I/O directly in GPU memory  

---

# 6. Latency vs Throughput Considerations

- **SIMD (CPU)**:
  - Lower latency, predictable timing  
  - Ideal for control loops and tight real-time constraints  
- **SIMT (GPU)**:
  - Higher throughput, higher startup latency  
  - Ideal for batch processing and heavy compute  

For **real-time robotics**:
- Keep **control loops** on CPU (with NEON where useful)  
- Use GPU for **perception** and **heavy inference**  
- Avoid blocking CPU on GPU results unless necessary  

---

# 7. Quick Jetson SIMD/SIMT Checklist

## SIMD (CPU NEON)
- [ ] Data aligned and contiguous  
- [ ] Loops vectorized (check compiler reports)  
- [ ] Intrinsics used for hot paths (if needed)  
- [ ] No heavy branching inside vector loops  

## SIMT (GPU CUDA)
- [ ] Coalesced global memory access  
- [ ] Shared memory used for reuse  
- [ ] Warp divergence minimized  
- [ ] Streams used for overlap  
- [ ] Data kept on GPU as long as possible  

## Pipeline-Level
- [ ] CPU does only what it’s best at (control, light math)  
- [ ] GPU handles heavy parallel work (vision, DL, point clouds)  
- [ ] Minimal CPU↔GPU transfers  
- [ ] Zero-copy used where appropriate (camera, NVMM)  

---

# 8. One-Sentence Strategy

> Use **SIMD on Jetson’s CPUs** for low-latency, small/medium parallel work, and **SIMT on the GPU** for massive, throughput-heavy workloads—then stitch them together with smart data movement and clear ownership of each stage.
