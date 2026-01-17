# Memory Hierarchy  
## Latency • Bandwidth • NUMA  
*A computer‑architecture perspective*

Modern processors (CPUs, GPUs, and SoCs like NVIDIA Jetson) use a **memory hierarchy** to balance speed, capacity, and power.  
No single memory technology can provide *both* high speed and large capacity, so systems layer multiple memory types.

This document explains the hierarchy in terms of:
- **Latency** (how long it takes to access data)
- **Bandwidth** (how much data can be moved per second)
- **NUMA** (Non‑Uniform Memory Access)

---

# 1. Why Memory Hierarchy Exists

Memory technologies trade off:
- **Speed** (SRAM is fast, DRAM is slow)
- **Capacity** (DRAM is large, SRAM is small)
- **Cost** (SRAM is expensive, DRAM is cheap)
- **Power** (SRAM consumes more static power)

To optimize performance, systems organize memory into layers:
- Small, fast memory close to the core
- Larger, slower memory farther away

---

# 2. Memory Hierarchy Levels

```
Registers → L1 Cache → L2 Cache → L3 Cache → DRAM → Storage
(Fastest)                                               (Slowest)
```

Each level increases:
- **Latency**  
- **Capacity**  
- **Distance from the core**  

Each level decreases:
- **Bandwidth**  
- **Cost per byte**  

---

# 3. Latency

## 3.1 What Is Latency?
Latency is the **time delay** between requesting data and receiving it.

Measured in:
- **CPU cycles**
- **Nanoseconds**

### Typical Latencies (Approximate)
| Level | Latency |
|-------|---------|
| Registers | ~1 cycle |
| L1 Cache | 1–4 cycles |
| L2 Cache | 10–20 cycles |
| L3 Cache | 30–50 cycles |
| DRAM | 100–200 cycles |
| SSD | 50,000–100,000 cycles |
| HDD | millions of cycles |

### Key Insight
A DRAM access is **100× slower** than an L1 cache access.  
This is why cache locality is critical.

---

# 4. Bandwidth

## 4.1 What Is Bandwidth?
Bandwidth is the **rate at which data can be transferred**.

Measured in:
- **GB/s**
- **Bytes per cycle**

### Typical Bandwidths
| Level | Bandwidth |
|-------|-----------|
| Registers | Extremely high |
| L1 Cache | Hundreds of GB/s |
| L2 Cache | Tens to hundreds of GB/s |
| L3 Cache | Tens of GB/s |
| DRAM | 20–200 GB/s (Jetson: ~50–200 GB/s depending on model) |
| PCIe | 8–32 GB/s |
| SSD | 3–7 GB/s |

### Key Insight
GPUs require **very high bandwidth** because thousands of threads access memory simultaneously.

---

# 5. NUMA (Non‑Uniform Memory Access)

## 5.1 What Is NUMA?
NUMA means **not all memory is equally close to all processors**.

In NUMA systems:
- Each CPU socket has its **own local memory**
- Accessing remote memory is **slower** and **lower bandwidth**

### NUMA Characteristics
- Local memory: **low latency, high bandwidth**
- Remote memory: **higher latency, lower bandwidth**

### Why NUMA Exists
- Multi-socket servers
- Large memory systems
- High-performance computing

---

# 6. NUMA in Jetson and Embedded Systems

Jetson SoCs are **not NUMA systems**.

They use:
- A **unified memory architecture**
- A single DRAM pool shared by CPU, GPU, and accelerators
- A coherent interconnect (CCI/CCN/NVLink fabric)

### Implications:
- No remote vs local memory distinction
- CPU–GPU memory sharing is efficient
- Zero-copy buffers are possible
- Latency is uniform across the SoC

---

# 7. Putting It All Together

## 7.1 Latency vs Bandwidth vs NUMA

| Concept | Meaning | Impact |
|---------|---------|--------|
| **Latency** | Time to access data | Affects single-thread performance |
| **Bandwidth** | Data transfer rate | Affects parallel workloads (GPU, DMA) |
| **NUMA** | Memory locality differences | Affects multi-socket systems |

---

# 8. Key Takeaways

- Memory hierarchy exists to balance **speed**, **capacity**, and **cost**.  
- Latency increases as you move away from the core.  
- Bandwidth decreases as you move down the hierarchy.  
- NUMA affects large multi-socket systems, but **Jetson uses a unified memory model**.  
- Optimizing for locality (L1/L2 hits) is essential for high performance.  
- GPUs rely on high bandwidth, while CPUs rely on low latency.

# Jetson-Specific Latency & Bandwidth Table  
*Approximate values for Nano, TX2, Xavier, and Orin*

This table summarizes the **latency** (how long it takes to access data) and **bandwidth** (how much data can be moved per second) across the memory hierarchy of NVIDIA Jetson SoCs.

---

## 1. Latency Table (Approximate)

| Memory Level | Jetson Nano | Jetson TX2 | Jetson Xavier NX / AGX | Jetson Orin | Notes |
|--------------|-------------|------------|--------------------------|--------------|-------|
| **Registers** | ~1 cycle | ~1 cycle | ~1 cycle | ~1 cycle | Fastest storage |
| **L1 Cache (CPU)** | 1–3 cycles | 1–3 cycles | 1–3 cycles | 1–3 cycles | Private per core |
| **L2 Cache (CPU)** | 10–15 cycles | 10–20 cycles | 12–20 cycles | 10–20 cycles | Shared across CPU cluster |
| **L1 Cache (GPU SM)** | ~20–30 cycles | ~20–30 cycles | ~20–30 cycles | ~20–30 cycles | Not coherent across SMs |
| **L2 Cache (GPU)** | ~100–150 cycles | ~100–150 cycles | ~100–150 cycles | ~80–120 cycles | Coherent across SMs |
| **DRAM Access** | 150–250 cycles | 150–250 cycles | 180–300 cycles | 150–250 cycles | Depends on frequency & load |
| **Unified Memory CPU↔GPU Sync** | 300–800 cycles | 300–800 cycles | 300–800 cycles | 200–600 cycles | Includes interconnect overhead |

---

## 2. Bandwidth Table (Approximate Peak Values)

| Component | Jetson Nano | Jetson TX2 | Jetson Xavier NX | Jetson AGX Xavier | Jetson Orin | Notes |
|-----------|-------------|------------|-------------------|--------------------|--------------|-------|
| **L1 Cache (CPU)** | ~1–2 TB/s | ~1–2 TB/s | ~1–2 TB/s | ~1–2 TB/s | ~1–2 TB/s | Extremely high internal bandwidth |
| **L2 Cache (CPU)** | ~200–400 GB/s | ~200–400 GB/s | ~300–500 GB/s | ~300–500 GB/s | ~400–600 GB/s | Depends on core count |
| **GPU L1 (per SM)** | ~1–2 TB/s | ~1–2 TB/s | ~2–3 TB/s | ~2–3 TB/s | ~3–4 TB/s | SM-local SRAM bandwidth |
| **GPU L2 (shared)** | ~100–150 GB/s | ~150–200 GB/s | ~200–300 GB/s | ~300–400 GB/s | ~400–600 GB/s | Depends on GPU architecture |
| **DRAM Bandwidth** | ~25.6 GB/s | ~59.7 GB/s | ~51.2 GB/s | ~137 GB/s | ~204–275 GB/s | LPDDR4/LPDDR5 dependent |
| **CPU↔GPU Coherent Interconnect** | ~10–20 GB/s | ~20–30 GB/s | ~30–50 GB/s | ~50–80 GB/s | ~80–120 GB/s | CCN/NVLink-like fabric |

---

## 3. Key Observations

### Latency
- CPU L1 is extremely fast (1–3 cycles).  
- GPU L1 is slower than CPU L1 because it is larger and configurable.  
- GPU L2 is the main coherence point and has higher latency.  
- DRAM latency is similar across Jetson models but varies with load and frequency.

### Bandwidth
- GPU L1 and shared memory have **massive bandwidth**, ideal for CUDA kernels.  
- DRAM bandwidth increases significantly from Nano → TX2 → Xavier → Orin.  
- Orin’s LPDDR5 memory provides the highest bandwidth of all Jetson modules.

### CPU–GPU Interaction
- Jetson uses a **unified memory architecture**, not NUMA.  
- CPU and GPU share DRAM through a coherent interconnect.  
- Bandwidth between CPU and GPU is much lower than internal GPU bandwidth.

---

## 4. Why This Matters for Performance

- **CUDA kernels** should maximize L1/shared memory usage to avoid L2/DRAM latency.  
- **Robotics workloads** benefit from predictable CPU L2 behavior.  
- **Deep learning inference** is often DRAM-bandwidth bound on Nano/TX2.  
- **Orin** dramatically improves memory bandwidth, enabling larger models and higher FPS.  
- **Zero-copy buffers** reduce latency but require careful synchronization.

  # Jetson Performance Insights  
## Detailed Explanations in Markdown Format

This document explains five key performance principles relevant to NVIDIA Jetson platforms, CUDA workloads, and embedded robotics systems.

---

# 1. CUDA Kernels Should Maximize L1/Shared Memory Usage  
### Why?
On Jetson GPUs (Maxwell, Pascal, Volta, Ampere), memory latency varies dramatically:

| Memory Level | Approx Latency |
|--------------|----------------|
| L1 / Shared Memory | 20–30 cycles |
| L2 Cache | 100–150 cycles |
| DRAM | 150–300+ cycles |

L1/shared memory is **5× faster than L2** and **10× faster than DRAM**.

### What This Means for CUDA
- Repeated DRAM access causes massive stalls.
- Shared memory allows threads in a block to **reuse data at L1 speed**.
- Shared memory is **software-managed**, giving full control over layout and reuse.

### When to Use Shared Memory
- Convolutions  
- Matrix multiplication  
- Stereo block matching  
- SLAM feature extraction  
- Any kernel with data reuse across threads

**Rule:** If multiple threads reuse the same data, load it into shared memory.

---

# 2. Robotics Workloads Benefit from Predictable CPU L2 Behavior  
Robotics workloads (SLAM, sensor fusion, control loops) require:
- Low jitter  
- Deterministic timing  
- Predictable memory access  

### Why CPU L2 Matters
- CPU L2 cache on Jetson is **shared and coherent**.
- Latency is stable (~10–20 cycles).
- Avoids unpredictable DRAM stalls (~200+ cycles).

### Example
A 200 Hz control loop:
- L2 hit → predictable timing  
- DRAM miss → sudden 200+ cycle stall → jitter → unstable control  

**Conclusion:** Keeping working sets inside CPU L2 improves real-time stability.

---

# 3. Deep Learning Inference Is Often DRAM-Bandwidth Bound on Nano/TX2  
Jetson DRAM bandwidth:

| Device | DRAM Bandwidth |
|--------|----------------|
| Nano | ~25.6 GB/s |
| TX2 | ~59.7 GB/s |

Deep learning inference (CNNs, DNNs) requires:
- Large tensor loads  
- Streaming feature maps  
- High memory reuse  

### Why Nano/TX2 Become Memory-Bound
- CNN layers require constant DRAM access.
- If weights/activations don’t fit in L2, GPU stalls waiting for data.
- GPU compute units sit idle due to insufficient memory bandwidth.

### Symptoms
- GPU utilization < 60%  
- Memory controller at 90–100%  
- FPS does not increase even with higher GPU clocks  

**Conclusion:** Nano/TX2 inference is limited by DRAM bandwidth, not compute.

---

# 4. Orin Dramatically Improves Memory Bandwidth  
Jetson Orin uses LPDDR5:

| Device | DRAM Bandwidth |
|--------|----------------|
| Orin | 204–275 GB/s |

This is:
- ~4× Xavier  
- ~10× TX2  
- ~8–12× Nano  

### Why This Matters
Deep learning inference is often bottlenecked by:
- Weight loading  
- Feature map streaming  
- Tensor reshaping  

With Orin:
- Larger models (YOLOv8, Transformers) run smoothly.
- Higher FPS is achievable.
- Multi-stream inference becomes possible.
- GPU stays fed → higher utilization.

**Conclusion:** Orin removes the memory bottleneck that limited earlier Jetsons.

---

# 5. Zero-Copy Buffers Reduce Latency but Require Synchronization  
Jetson uses a **unified memory architecture**:
- CPU and GPU share the same DRAM.
- Zero-copy buffers allow GPU to read CPU memory directly.

### Benefits
- No CPU↔GPU memcpy  
- Lower latency  
- Lower CPU overhead  
- Ideal for camera frames, sensor data, ROS2 messages, VPI pipelines  

### The Catch: Synchronization
Zero-copy memory is coherent but **not automatically synchronized**.

### Risks Without Sync
- CPU writes → GPU reads stale data  
- GPU writes → CPU reads stale data  
- Race conditions  
- Partial writes  

### Required Synchronization Tools
- `cudaDeviceSynchronize()`  
- `__threadfence_system()`  
- CUDA streams + events  
- CPU cache flushes  
- VPI/DeepStream pipeline barriers  

**Rule:** Zero-copy is fast, but visibility must be explicitly controlled.

---

# Final Summary

| Concept | Why It Matters |
|--------|----------------|
| Maximize L1/shared memory | Avoid 10× slower DRAM access in CUDA kernels |
| Predictable CPU L2 | Reduces jitter in robotics control loops |
| Nano/TX2 DRAM-bound | Memory bandwidth limits inference FPS |
| Orin’s high bandwidth | Enables large models + multi-stream real-time inference |
| Zero-copy buffers | Fastest CPU–GPU path but requires synchronization |

