# Caches in Computer Architecture  
## Coherency • Associativity • Eviction

Modern processors rely heavily on caches to bridge the speed gap between the CPU and main memory.  
This document explains three fundamental cache concepts in computer architecture:

- **Cache Coherency**  
- **Cache Associativity**  
- **Cache Eviction Policies**

---

# 1. Cache Coherency

## 1.1 What Is Cache Coherency?
In multi-core processors, each core typically has its own private L1 and sometimes L2 caches.  
When multiple cores access the same memory location, they must observe **consistent values**.

Cache coherency ensures:
- No core reads stale data  
- Writes become visible to other cores  
- Shared-memory multithreading behaves correctly  

Without coherency, parallel programs would produce incorrect results.

---

## 1.2 The Coherency Problem
Example scenario:

1. Core 0 loads `x = 10` into its cache  
2. Core 1 loads `x = 10` into its cache  
3. Core 0 updates `x = 20`  

Now:
- Core 0’s cache has `20`  
- Core 1’s cache still has `10` → **stale**

A coherency protocol must resolve this.

---

## 1.3 MESI Coherency Protocol
The most widely used protocol is **MESI**, which assigns one of four states to each cache line:

| State | Meaning |
|-------|---------|
| **M – Modified** | Cache line changed; main memory is stale |
| **E – Exclusive** | Cache line is clean and only in this cache |
| **S – Shared** | Cache line is clean and may exist in other caches |
| **I – Invalid** | Cache line is invalid; must be reloaded |

### Key behaviors:
- A write typically **invalidates** other cores’ copies  
- A read may cause a line to transition to **Shared**  
- A write to a Shared line transitions it to **Modified**  

This ensures all cores eventually see the same value.

---

# 2. Cache Associativity

## 2.1 What Is Associativity?
Associativity describes **how memory addresses map to cache lines**.

Caches are divided into:
- **Sets**  
- **Ways (lines per set)**  

A memory block maps to exactly **one set**, but may occupy **one of several ways**.

---

## 2.2 Types of Associativity

### 1. Direct-Mapped Cache (1-way)
- Each memory block maps to exactly **one** cache line  
- Fast and simple  
- High conflict rate  

If two addresses map to the same line, they repeatedly evict each other → **thrashing**.

---

### 2. Fully Associative Cache
- A memory block can be placed **anywhere** in the cache  
- Lowest conflict rate  
- Expensive hardware (comparators for every line)  

Used mainly in:
- TLBs (Translation Lookaside Buffers)

---

### 3. N-Way Set Associative Cache
The practical compromise used in modern CPUs.

Example: **4-way associative**
- Cache is divided into sets  
- Each set contains 4 lines  
- A memory block maps to **one set**, but can occupy **any of the 4 lines**

### Why it matters:
- Higher associativity → fewer conflict misses  
- Lower associativity → faster and cheaper hardware  

Typical modern caches:
- L1: 4-way or 8-way  
- L2/L3: 8-way to 16-way  

---

# 3. Cache Eviction Policies

## 3.1 What Is Eviction?
When a cache set is full and a new block must be loaded, the cache must **evict** one of the existing lines.

Eviction policy determines **which line to remove**.

---

## 3.2 Common Eviction Policies

### 1. LRU — Least Recently Used
Evicts the line that has not been used for the longest time.

Pros:
- Good approximation of real program behavior  

Cons:
- Expensive to implement for high associativity  

---

### 2. FIFO — First In, First Out
Evicts the oldest line in the set.

Pros:
- Simple hardware  

Cons:
- Not always optimal  

---

### 3. Random
Evicts a random line.

Pros:
- Very cheap  
- Surprisingly effective  

Cons:
- Unpredictable  

---

### 4. Pseudo-LRU (PLRU)
A hardware-friendly approximation of LRU.

Used in:
- Many ARM and x86 L1/L2 caches  

---

# 4. Summary

| Concept | Purpose | Key Idea |
|--------|----------|----------|
| **Coherency** | Multi-core correctness | Ensures all cores see consistent data |
| **Associativity** | Cache organization | Controls where data can be placed |
| **Eviction** | Cache replacement | Decides what gets removed when full |

Together, these mechanisms allow caches to be:
- Fast  
- Predictable  
- Efficient  
- Correct in multi-core environments  

---

# 5. Final Notes
In computer architecture, cache behavior directly affects:
- Performance  
- Power consumption  
- Real-time determinism  
- Multi-core correctness  

Understanding coherency, associativity, and eviction is essential for designing efficient embedded systems, optimizing software, and analyzing performance bottlenecks.

# NVIDIA Jetson Architecture Notes  
*A concise technical overview for embedded systems engineers*

NVIDIA Jetson modules (Nano, TX2, Xavier, Orin) are heterogeneous SoCs designed for edge AI, robotics, and real‑time embedded workloads. They combine ARM CPUs, an NVIDIA GPU, dedicated accelerators, and a coherent memory subsystem optimized for high‑bandwidth parallel processing.

---

# 1. Jetson Hardware Architecture Overview

## 1.1 Key Components
- **ARM CPU Complex**
  - Multi-core ARM Cortex-A series (A57, A72, Carmel, or Cortex-A78AE depending on generation)
  - Supports out-of-order execution, NEON SIMD, and hardware virtualization

- **NVIDIA GPU**
  - Maxwell (Nano), Pascal (TX2), Volta (Xavier), Ampere (Orin)
  - CUDA cores organized into Streaming Multiprocessors (SMs)
  - Tensor Cores (Xavier, Orin) for deep learning acceleration

- **Dedicated Accelerators**
  - NVDLA (Deep Learning Accelerator)
  - PVA (Programmable Vision Accelerator)
  - ISP (Image Signal Processor)
  - NVENC/NVDEC (Hardware video encoder/decoder)

- **Unified Memory Subsystem**
  - Shared DRAM for CPU, GPU, and accelerators
  - Coherent interconnect (ACE/CCI/CCN/NVLink depending on model)

---

# 2. Memory Architecture

## 2.1 Unified Physical Memory
Jetson uses a **unified DRAM pool** shared by:
- ARM CPUs  
- GPU  
- NVDLA  
- ISP  
- DMA engines  

This eliminates the need for explicit CPU–GPU memory copies.

## 2.2 Cache Hierarchy
### CPU Caches
- Private L1 caches per core  
- Shared L2 cache  
- Fully coherent across CPU cores  

### GPU Caches
- **L1 (per SM):**  
  - Not coherent across SMs  
  - Acts as a private scratchpad  
- **L2 (shared across GPU):**  
  - Coherent across all SMs  
  - Coherence point for CPU–GPU interactions  

### System-Level Coherency
Jetson SoCs use ARM’s coherent interconnects:
- ACE / CCI / CCN / NVLink (depending on generation)

This ensures:
- CPU caches stay coherent with GPU L2  
- Zero-copy buffers work reliably  
- Real-time robotics workloads maintain predictable behavior

---

# 3. Cache Coherency on Jetson

## 3.1 GPU Internal Coherency
- L1 caches on SMs are **not coherent**
- L2 cache is **globally coherent**
- Synchronization primitives required:
  - `__threadfence()`
  - `__threadfence_system()`
  - `__syncthreads()`

## 3.2 CPU–GPU Coherency
- Managed by hardware interconnect
- CPU sees GPU writes after:
  - GPU flushes to L2  
  - Memory fence instructions  
- GPU sees CPU writes after:
  - CPU cache flush or memory barrier  
  - Coherent interconnect propagation  

---

# 4. Jetson Software Stack

## 4.1 JetPack SDK
JetPack includes:
- L4T (Linux for Tegra)
- CUDA Toolkit
- cuDNN, TensorRT
- Multimedia API (ISP, NVENC/NVDEC)
- VPI (Vision Programming Interface)
- DeepStream SDK

## 4.2 Linux for Tegra (L4T)
- Ubuntu-based distribution
- Custom kernel with NVIDIA drivers
- Supports:
  - GStreamer accelerated plugins
  - V4L2 camera pipeline
  - Real-time patches (optional)

---

# 5. Performance Considerations

## 5.1 Memory Bandwidth
Jetson modules rely heavily on DRAM bandwidth.  
Optimizations include:
- Using pinned memory  
- Minimizing CPU–GPU transfers  
- Preferring zero-copy buffers  
- Using shared memory inside CUDA kernels  

## 5.2 Power Modes
Jetson supports multiple power modes:
- nvpmodel profiles  
- DVFS (Dynamic Voltage and Frequency Scaling)  
- Per-core and per-accelerator power gating  

## 5.3 Real-Time Behavior
- CPU cores can run PREEMPT_RT kernel  
- GPU is not real-time deterministic  
- Use PVA or NVDLA for deterministic workloads  

---

# 6. Jetson Use Cases
- Autonomous robots  
- Drones  
- Industrial automation  
- Medical imaging  
- Smart cameras  
- Edge AI inference  
- SLAM and stereo vision  
- Real-time video analytics  

---

# 7. Summary

NVIDIA Jetson combines:
- A coherent ARM CPU cluster  
- A massively parallel GPU  
- Dedicated AI and vision accelerators  
- A unified memory architecture  
- A rich software ecosystem (JetPack)

This makes Jetson ideal for embedded AI and robotics applications requiring high performance, low latency, and efficient power usage.


# Cache Associativity in NVIDIA Jetson  
*A computer‑architecture perspective*

NVIDIA Jetson modules use ARM CPUs + NVIDIA GPUs inside a unified SoC.  
Both the CPU and GPU have their own cache hierarchies, and **associativity differs between them**.

This note explains how associativity works on Jetson platforms.

---

# 1. CPU Cache Associativity (ARM Cores)

Jetson SoCs use ARM Cortex‑A series CPUs (A57, A72, Carmel, Cortex‑A78AE).  
These CPUs follow standard ARM cache designs.

## 1.1 L1 Cache Associativity
- **Instruction L1 (I‑cache):** typically **2‑way or 4‑way associative**  
- **Data L1 (D‑cache):** typically **4‑way associative**

L1 caches are:
- Private per core  
- Write‑back  
- Coherent across cores via ACE/CCI/CCN interconnect  

---

## 1.2 L2 Cache Associativity
- Shared across all CPU cores  
- Typically **8‑way associative**  
- Inclusive or exclusive depending on architecture  
- Fully coherent across CPU cluster  

---

# 2. GPU Cache Associativity (NVIDIA SMs)

NVIDIA GPUs inside Jetson (Maxwell, Pascal, Volta, Ampere) have a different cache design than CPUs.

## 2.1 L1 Cache (per SM)
- Configurable as **shared memory + L1**  
- Associativity varies by architecture:
  - Maxwell (Nano): **4‑way associative**
  - Pascal (TX2): **4‑way associative**
  - Volta (Xavier): **4‑way associative**
  - Ampere (Orin): **4‑way associative**

### Important:
- **L1 caches are NOT coherent across SMs**
- They act more like scratchpads for CUDA kernels

---

## 2.2 L2 Cache (shared across GPU)
- Unified L2 cache for all SMs  
- Typically **16‑way associative**  
- This is the **coherency point** for the GPU  
- All SMs see consistent data through L2  

---

# 3. System-Level Cache Coherency and Associativity

Jetson uses ARM’s coherent interconnects:
- CCI (Nano, TX2)
- CCN (Xavier)
- NVLink‑based fabric (Orin)

These interconnects ensure:
- CPU L2 ↔ GPU L2 coherence  
- DMA engines see consistent memory  
- Zero‑copy buffers behave predictably  

Associativity at this level is handled by:
- CPU L2 (8‑way)
- GPU L2 (16‑way)
- System cache (varies by SoC)

---

# 4. Why Associativity Matters on Jetson

## 4.1 For CPU Workloads
Higher associativity:
- Reduces conflict misses  
- Improves real‑time predictability  
- Helps with multi-threaded robotics workloads  

## 4.2 For GPU Workloads
GPU L1 associativity affects:
- Shared memory bank conflicts  
- Warp-level memory access patterns  
- CUDA kernel performance  

GPU L2 associativity affects:
- Global memory bandwidth  
- Unified Memory performance  
- CPU–GPU data sharing  

---

# 5. Summary Table

| Component | Jetson Nano | TX2 | Xavier | Orin | Associativity |
|----------|-------------|-----|--------|------|----------------|
| CPU L1 | A57 | A57 | Carmel | A78AE | 2–4 way |
| CPU L2 | Shared | Shared | Shared | Shared | 8‑way |
| GPU L1 | Maxwell | Pascal | Volta | Ampere | 4‑way |
| GPU L2 | Maxwell | Pascal | Volta | Ampere | 16‑way |
| System Coherency | CCI | CCI | CCN | NVLink fabric | N/A |

---

# 6. Key Takeaways

- Jetson CPUs use **standard ARM associativity** (L1: 2–4 way, L2: 8‑way).  
- Jetson GPUs use **4‑way L1** and **16‑way L2** associativity.  
- GPU L1 caches are **not coherent**, but GPU L2 is.  
- CPU and GPU share memory through a **coherent interconnect**, not through shared L1.  
- Associativity directly impacts performance in CUDA, robotics, and real‑time workloads.

# Cache Eviction Policies  
*A computer‑architecture perspective with relevance to NVIDIA Jetson*

Cache eviction policies determine **which cache line to remove** when a cache set is full and a new memory block must be loaded.  
Eviction is essential because caches are limited in size and organized into sets (due to associativity).

This document explains the major eviction policies used in CPUs, GPUs, and specifically NVIDIA Jetson SoCs.

---

# 1. Why Eviction Policies Matter

Eviction policies directly influence:
- Cache hit rate  
- Memory bandwidth usage  
- Real‑time determinism  
- Power consumption  
- Performance of CPU, GPU, and accelerators  

On Jetson, eviction behavior affects:
- CUDA kernels  
- VPI pipelines  
- TensorRT inference  
- CPU–GPU shared memory  
- Robotics workloads with tight timing constraints  

---

# 2. Common Cache Eviction Policies

## 2.1 LRU — Least Recently Used
Evicts the cache line that has not been accessed for the longest time.

### Characteristics
- Good approximation of real program behavior  
- Reduces conflict misses  
- Expensive to implement for high associativity (tracking usage order)  

### Where it's used
- ARM CPU L1/L2 caches (Jetson Nano, TX2, Xavier, Orin use variants of LRU or pseudo‑LRU)  
- Some GPU L2 caches  

---

## 2.2 FIFO — First In, First Out
Evicts the oldest line in the set, regardless of usage.

### Characteristics
- Simple hardware  
- Predictable timing  
- Not always optimal for hit rate  

### Where it's used
- Real‑time systems where determinism matters  
- Some GPU caches  

---

## 2.3 Random Replacement
Evicts a random line from the set.

### Characteristics
- Very cheap to implement  
- Surprisingly effective for highly parallel workloads  
- Avoids pathological thrashing patterns  

### Where it's used
- Some NVIDIA GPU L1 caches  
- TLBs in many architectures  

---

## 2.4 Pseudo-LRU (PLRU)
A hardware‑friendly approximation of LRU.

### Characteristics
- Tracks usage with a small number of bits  
- Much cheaper than true LRU  
- Slightly worse hit rate than full LRU  
- Very common in modern CPUs and GPUs  

### Where it's used
- ARM Cortex‑A CPU caches  
- NVIDIA GPU L1 and L2 caches (architecture‑dependent)  

---

# 3. Eviction Policies in NVIDIA Jetson

Jetson SoCs combine ARM CPUs + NVIDIA GPUs, so eviction behavior differs by component.

## 3.1 CPU Eviction (ARM Cores)
ARM Cortex‑A57, A72, Carmel, and A78AE use:
- **Pseudo‑LRU** in L1  
- **Pseudo‑LRU or true LRU** in L2  

These policies balance:
- Good hit rate  
- Predictable performance  
- Low hardware cost  

---

## 3.2 GPU Eviction (NVIDIA SMs)
NVIDIA GPUs (Maxwell, Pascal, Volta, Ampere) use different policies for L1 and L2.

### L1 (per SM)
- Often **random** or **pseudo‑LRU**  
- Optimized for warp‑level parallelism  
- Designed to avoid thrashing under massive thread counts  

### L2 (shared across GPU)
- Typically **pseudo‑LRU**  
- Ensures fairness across SMs  
- Balances bandwidth and latency  

---

## 3.3 System-Level Caches (Jetson)
Jetson’s coherent interconnect (CCI/CCN/NVLink fabric) interacts with:
- CPU L2 (8‑way)  
- GPU L2 (16‑way)  
- System cache (varies by SoC)  

Eviction here is usually:
- **Pseudo‑LRU** or  
- **Vendor‑specific deterministic policy**  

These ensure:
- CPU–GPU coherency  
- Predictable DMA behavior  
- Stable performance for robotics and vision pipelines  

---

# 4. Why NVIDIA Avoids True LRU in GPUs

True LRU is expensive because it requires:
- Tracking exact usage order  
- Updating metadata on every access  
- Complex hardware for 16‑way sets  

GPUs have:
- Thousands of threads  
- Hundreds of concurrent memory requests  
- Very high bandwidth demands  

Thus, NVIDIA prefers:
- **Random** for L1  
- **Pseudo‑LRU** for L2  

This provides:
- High throughput  
- Good average hit rate  
- Scalable hardware  

---

# 5. Summary Table

| Cache Level | Jetson Component | Associativity | Eviction Policy |
|-------------|------------------|---------------|------------------|
| L1 (CPU) | ARM cores | 2–4 way | Pseudo‑LRU |
| L2 (CPU) | ARM cluster | 8‑way | Pseudo‑LRU / LRU |
| L1 (GPU) | SM private | 4‑way | Random / Pseudo‑LRU |
| L2 (GPU) | Shared | 16‑way | Pseudo‑LRU |
| System Cache | Interconnect | Varies | Deterministic PLRU |

---

# 6. Key Takeaways

- Eviction policies determine which cache line gets replaced when a set is full.  
- Jetson CPUs use **pseudo‑LRU** for efficiency and predictability.  
- Jetson GPUs use **random or pseudo‑LRU** to scale across thousands of threads.  
- GPU L2 is the main coherence point and uses **pseudo‑LRU**.  
- Eviction behavior affects CUDA, VPI, TensorRT, and real‑time robotics workloads.  

![CPU GPU Archiecture](/images/Cache.png)

# CPU vs GPU Architecture: Block-by-Block Breakdown  
*A detailed explanation of each architectural component*

This note explains the functional role of each block in a comparative diagram of CPU and GPU architectures. It highlights how CPUs are optimized for sequential, latency-sensitive tasks, while GPUs are designed for massively parallel workloads.

---

## 1. CPU Architecture Blocks

### 🟩 Core (Green)
- A CPU core is a powerful, general-purpose execution unit.
- Each core includes:
  - ALUs (Arithmetic Logic Units)
  - FPUs (Floating Point Units)
  - Branch predictors
  - Register files
- Designed for **low-latency**, **complex control flow**, and **single-thread performance**.

### 🟧 Control (Orange)
- Manages instruction decoding, scheduling, and execution.
- Handles out-of-order execution, speculative execution, and pipeline control.
- Sophisticated control logic enables high IPC (Instructions Per Cycle).

### 🟪 L1 Cache (Purple)
- Split into:
  - **L1 Instruction Cache (I-Cache)**
  - **L1 Data Cache (D-Cache)**
- Typically 2–4 way associative.
- Fastest cache level, private to each core.
- Latency: ~1–4 cycles.

### 🔵 L2 Cache (Blue)
- Shared across all CPU cores (or per cluster).
- Larger than L1, slower (~10–20 cycles).
- Typically 8-way associative.
- Maintains coherency across cores via ACE/CCI/CCN interconnect.

### 🔵 L3 Cache (Blue)
- Optional in embedded systems; present in desktop/server CPUs.
- Shared across all cores.
- Higher capacity, slower latency (~30–50 cycles).
- Inclusive or exclusive depending on architecture.

### 🟧 DRAM (Orange)
- Off-chip main memory.
- Latency: ~100–200 cycles.
- Accessed via memory controller and interconnect.
- DRAM bandwidth is a bottleneck for memory-intensive workloads.

---

## 2. GPU Architecture Blocks

### 🟩 GPU Cores (Green Grid)
- Thousands of lightweight cores arranged in **Streaming Multiprocessors (SMs)**.
- Each SM contains:
  - CUDA cores (integer/floating point units)
  - Tensor cores (for matrix ops)
  - Warp schedulers
  - Register files
- Designed for **SIMD-style parallelism** and **high throughput**.

### 🟧 Control (Orange)
- Much simpler than CPU control logic.
- Focuses on warp scheduling and instruction dispatch.
- No branch prediction or speculative execution.
- Optimized for **regular, data-parallel workloads**.

### 🟪 L1 Cache (Purple)
- Private to each SM.
- Often configurable as **shared memory + L1 cache**.
- Typically 4-way associative.
- Not coherent across SMs.
- Acts as a scratchpad for CUDA kernels.

### 🔵 L2 Cache (Blue)
- Shared across all SMs.
- Coherent across GPU cores.
- Typically 16-way associative.
- Acts as the **coherency point** for CPU–GPU interactions.
- Supports atomic operations and global memory access.

### 🟧 DRAM (Orange)
- Unified memory pool shared with CPU (on Jetson) or dedicated GDDR (on discrete GPUs).
- Accessed via memory controller.
- Latency: ~300–500 cycles.
- Bandwidth: hundreds of GB/s on modern GPUs.

---

## 3. Architectural Contrast Summary

| Feature | CPU | GPU |
|--------|-----|-----|
| Core Count | Few (2–16) | Many (hundreds–thousands) |
| Core Complexity | High | Low |
| Control Logic | Sophisticated | Lightweight |
| L1 Cache | Private, split I/D | Private, unified/shared |
| L2 Cache | Shared, coherent | Shared, coherent |
| L3 Cache | Optional | Not present |
| DRAM Access | Latency-sensitive | Bandwidth-sensitive |
| Parallelism | Task-level | Data-level (SIMD) |

---

## 4. Embedded Systems Implications (Jetson Context)

- Jetson CPUs use ARM cores with standard L1/L2 caches and pseudo-LRU eviction.
- Jetson GPUs use SMs with private L1 and shared L2 caches.
- CPU–GPU coherency is managed via CCN/NVLink interconnect.
- CUDA developers must manage memory visibility using fences and shared memory.

# Jetson-Specific Cache Flow  
*A detailed overview of cache hierarchy and data movement in NVIDIA Jetson SoCs*

NVIDIA Jetson modules combine ARM CPUs, NVIDIA GPUs, and dedicated accelerators into a unified SoC.  
Understanding the cache flow is essential for optimizing memory access, ensuring coherency, and designing real-time systems.

---

## 1. Cache Hierarchy Overview

### CPU Side (ARM Cortex-A Series)
- **L1 Cache (per core)**  
  - Separate instruction and data caches  
  - Typically 2–4 way associative  
  - Fastest access, private to each core  

- **L2 Cache (shared)**  
  - Shared across all CPU cores  
  - Typically 8-way associative  
  - Coherent across cores via ACE/CCI/CCN interconnect  

---

### GPU Side (Streaming Multiprocessors)
- **L1 Cache (per SM)**  
  - Unified or split with shared memory  
  - Typically 4-way associative  
  - Not coherent across SMs  
  - Acts as scratchpad for CUDA kernels  

- **L2 Cache (shared across GPU)**  
  - Coherent across all SMs  
  - Typically 16-way associative  
  - Coherency point for CPU–GPU interactions  

---

## 2. Cache Flow: CPU Access

1. CPU core accesses data → L1 D-Cache  
2. If miss → L2 Cache  
3. If L2 miss → DRAM via memory controller  
4. Coherency maintained across cores via interconnect  
5. DMA engines and accelerators access memory via coherent interconnect  

---

## 3. Cache Flow: GPU Access

1. CUDA thread accesses global memory → L1 Cache (per SM)  
2. If miss → L2 Cache (shared across SMs)  
3. If L2 miss → DRAM  
4. L2 Cache ensures coherency across SMs  
5. Shared memory accesses bypass L1 and go directly to SM-local SRAM  

---

## 4. CPU–GPU Coherency Flow

### On Jetson SoCs:
- CPU and GPU share **physical DRAM**  
- Coherency managed via **ARM interconnects**:
  - CCI (Nano, TX2)  
  - CCN (Xavier)  
  - NVLink-like fabric (Orin)  

### Flow:
1. GPU writes to L2 → visible to CPU after flush or fence  
2. CPU writes to L2 → visible to GPU after cache flush or barrier  
3. Unified Memory (UM) uses page migration + faulting for consistency  
4. Zero-copy buffers rely on hardware coherency  

---

## 5. Accelerator Cache Flow

### NVDLA / PVA / ISP:
- Access DRAM via coherent interconnect  
- May use internal buffers or SRAM  
- Rely on CPU/GPU cache flushes for visibility  
- Typically bypass L1 caches  

---

## 6. Summary Diagram (Textual)

```
[CPU Core] → L1 Cache → L2 Cache → DRAM
↑
[Other CPU Cores] ←→ Coherent Interconnect ←→ [GPU L2 Cache] ←→ [GPU SMs] → L1 Cache
↓
Shared Memory / Registers
```

---

## 7. Performance Tips

- Use **shared memory** in CUDA for predictable latency  
- Minimize CPU–GPU memory transfers  
- Use **pinned memory** for DMA efficiency  
- Use **threadfence_system()** for visibility across CPU–GPU  
- Align memory accesses to cache line boundaries  

---

## 8. Key Takeaways

- Jetson uses a **hierarchical cache flow** optimized for embedded AI workloads  
- CPU and GPU share DRAM but have separate L1/L2 caches  
- Coherency is enforced via hardware interconnects and memory fences  
- Understanding cache flow is critical for real-time robotics, vision, and inference pipelines


