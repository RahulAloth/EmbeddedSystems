# Jetson Performance Tuning Guide  
*Practical notes for CUDA, robotics, and deep learning on NVIDIA Jetson*

This guide summarizes concrete, architecture-aware tuning strategies for NVIDIA Jetson platforms (Nano, TX2, Xavier, Orin), focusing on:
- CUDA kernels  
- Robotics/control workloads  
- Deep learning inference  
- Memory and cache behavior  

---

# 1. Understand Your Bottleneck

Before tuning, identify whether your workload is:
- **Compute-bound** (ALUs/Tensor Cores saturated)
- **Memory-bound** (DRAM or L2 saturated)
- **Latency-bound** (cache misses, synchronization)
- **I/O-bound** (camera, disk, network)

### Tools
- `tegrastats`
- `nvpmodel` + `jetson_clocks`
- Nsight Systems / Nsight Compute
- `watch -n 0.5 tegrastats`

**Rule:** Never tune blindly—measure first.

---

# 2. Power Modes and Clocks

Jetson modules support multiple power/performance modes.

### Steps
- Use `sudo nvpmodel -q` to check current mode.
- Use `sudo nvpmodel -m <mode>` to select a higher performance profile.
- Use `sudo jetson_clocks` to lock clocks for consistent benchmarking.

### Notes
- Higher modes → more power, more heat.
- Ensure adequate cooling (heatsink, fan) to avoid thermal throttling.

**Tip:** For benchmarking and tuning, always enable `jetson_clocks`.

---

# 3. CUDA Kernel Optimization

## 3.1 Use Shared Memory Effectively
- Load reused data into **shared memory**.
- Avoid repeated global memory reads.
- Align shared memory access to avoid bank conflicts.

### Good Use Cases
- Convolutions
- Matrix multiplication
- Block matching (stereo)
- Local window operations

---

## 3.2 Optimize Memory Access Patterns
- Ensure **coalesced global memory access**:
  - Threads in a warp should access consecutive addresses.
- Avoid strided or random access patterns.
- Use `float4`, `int4` where appropriate for vectorized loads.

---

## 3.3 Control Occupancy
- Balance:
  - Threads per block
  - Registers per thread
  - Shared memory per block
- Use `nvcc --ptxas-options=-v` to inspect register usage.
- Use Nsight Compute to analyze occupancy.

**Rule:** Maximum occupancy is not always maximum performance—prioritize memory efficiency.

---

# 4. Deep Learning Inference Tuning

## 4.1 Use TensorRT
- Convert models (ONNX, PyTorch, TF) to TensorRT engines.
- Enable:
  - FP16 on Nano/TX2/Xavier/Orin
  - INT8 on Xavier/Orin (with calibration)

### Benefits
- Layer fusion
- Kernel auto-tuning
- Reduced memory footprint
- Higher FPS

---

## 4.2 Batch Size and Resolution
- Increase **batch size** to improve GPU utilization (if latency allows).
- Reduce **input resolution** if FPS is more important than accuracy.
- For real-time robotics:
  - Prefer smaller models (YOLO-tiny, MobileNet, etc.)
  - Use FP16/INT8 where possible.

---

## 4.3 Orin-Specific Tuning
- Orin has much higher DRAM bandwidth.
- You can:
  - Run larger models (YOLOv8, Transformers).
  - Run multiple models in parallel.
  - Use higher resolutions (1080p, 4K).

**Tip:** On Orin, you’re more likely to be compute-bound than memory-bound.

---

# 5. Robotics and Real-Time Workloads

## 5.1 CPU Core Affinity and Isolation
- Pin critical threads (control loops, SLAM) to specific CPU cores.
- Avoid running heavy background tasks on the same cores.
- Use `taskset` or `cgroups` to manage CPU affinity.

---

## 5.2 Use PREEMPT_RT (If Needed)
- For hard real-time requirements:
  - Use a PREEMPT_RT patched kernel.
  - Prioritize control threads with real-time scheduling (`SCHED_FIFO`).

---

## 5.3 Minimize Jitter
- Keep control loop working sets in **CPU L2**.
- Avoid dynamic memory allocation in real-time loops.
- Avoid blocking I/O in critical threads.

**Rule:** Determinism > raw speed for control.

---

# 6. Memory and Zero-Copy Tuning

## 6.1 Use Pinned (Page-Locked) Memory
- For CPU↔GPU transfers, use pinned memory:
  - Faster DMA transfers
  - More predictable latency

---

## 6.2 Zero-Copy Buffers
- Use zero-copy for:
  - Camera frames
  - Sensor data
  - VPI pipelines
- Avoid unnecessary `cudaMemcpy`.

### Synchronization Requirements
- Use:
  - `cudaDeviceSynchronize()`
  - CUDA streams + events
  - `__threadfence_system()`
- Ensure CPU does not read/write while GPU is using the buffer (and vice versa).

---

# 7. Jetson-Specific Tips

## 7.1 Use Hardware Accelerators
- Use:
  - NVDLA for DNN inference (where supported)
  - PVA for vision tasks
  - NVENC/NVDEC for video encoding/decoding
- Offload work from CPU/GPU when possible.

---

## 7.2 Optimize GStreamer Pipelines
- Use hardware-accelerated elements:
  - `nvvidconv`
  - `nvv4l2decoder`
  - `nvv4l2h264enc` / `nvv4l2h265enc`
- Keep frames in NVMM (zero-copy) where possible.

---

## 7.3 Monitor and Iterate
- Continuously monitor:
  - CPU/GPU utilization
  - Memory bandwidth
  - Temperature
  - Power draw
- Use `tegrastats` during real workloads, not just synthetic tests.

---

# 8. Quick Checklist

- [ ] Set appropriate `nvpmodel` mode  
- [ ] Run `jetson_clocks` for consistent performance  
- [ ] Use TensorRT for inference  
- [ ] Use shared memory in CUDA kernels  
- [ ] Ensure coalesced global memory access  
- [ ] Pin critical threads to specific CPU cores  
- [ ] Use pinned memory for transfers  
- [ ] Use zero-copy where appropriate (with proper sync)  
- [ ] Offload to NVDLA/PVA/NVENC/NVDEC when possible  
- [ ] Monitor with `tegrastats` and Nsight tools  

---

# 9. Final Thoughts

Jetson performance tuning is about:
- Respecting the **memory hierarchy**  
- Exploiting **shared memory and caches**  
- Matching **model size** to **bandwidth and compute**  
- Designing for **determinism** in robotics workloads  


# Throughput vs Latency Tradeoffs  
*A computer‑architecture perspective*

Throughput and latency are two fundamental performance metrics in computing systems.  
They often move in opposite directions — improving one can worsen the other.  
Understanding this tradeoff is essential for optimizing CPUs, GPUs, CUDA kernels, robotics loops, and deep learning inference on Jetson platforms.

---

# 1. Definitions

## 1.1 Latency  
Latency is the **time it takes to complete a single operation**.

Examples:
- Time to fetch one value from memory  
- Time for one CUDA thread to finish a task  
- Time for a control loop iteration  

Latency is measured in:
- Nanoseconds  
- CPU cycles  
- Milliseconds (for robotics loops)

---

## 1.2 Throughput  
Throughput is the **number of operations completed per unit time**.

Examples:
- Frames per second (FPS) in inference  
- Number of CUDA threads completed per second  
- Number of packets processed per second  

Throughput is measured in:
- Operations per second  
- GB/s  
- FPS  

---

# 2. Why They Trade Off

Improving latency often requires:
- More complex hardware  
- Larger caches  
- Deeper pipelines  
- More control logic  

These increase:
- Power consumption  
- Area  
- Cost  

Improving throughput often requires:
- More parallelism  
- More cores/SMs  
- Wider memory buses  
- Batch processing  

These can increase:
- Per‑operation latency  
- Queueing delays  
- Synchronization overhead  

**Key idea:**  
> Latency is about *how fast one thing finishes*.  
> Throughput is about *how many things finish per second*.  
> Optimizing for one often hurts the other.

---

# 3. CPU vs GPU: A Perfect Example

## CPU (Latency‑Optimized)
- Few, powerful cores  
- Sophisticated branch prediction  
- Large caches  
- Out‑of‑order execution  
- Low‑latency memory access  

**Goal:** Minimize the time for a single thread to finish.

---

## GPU (Throughput‑Optimized)
- Many simple cores (SMs)  
- Massive parallelism  
- High memory bandwidth  
- Simple control logic  
- High latency tolerated  

**Goal:** Maximize total operations per second, not minimize per‑thread latency.

---

# 4. Jetson-Specific Examples

## 4.1 CUDA Kernels  
- GPUs hide latency by running thousands of threads.  
- If one warp stalls, another warp runs.  
- Latency per thread is high, but throughput is enormous.

**Tradeoff:**  
- High throughput  
- High per-thread latency  

---

## 4.2 Robotics Control Loops  
- Require low latency and low jitter.  
- Throughput is irrelevant — only the *next* control update matters.

**Tradeoff:**  
- Low latency  
- Low throughput (only one loop at a time)

---

## 4.3 Deep Learning Inference  
- GPUs process many pixels/tensors in parallel.  
- Batch size increases throughput but increases latency.

**Example:**  
- Batch size 1 → low latency, low throughput  
- Batch size 16 → high throughput, high latency  

---

## 4.4 Jetson Orin vs Nano  
- Orin has huge memory bandwidth → high throughput  
- Nano has lower bandwidth → lower throughput  
- Latency per memory access is similar, but Orin can process far more data per second.

---

# 5. Practical Tradeoffs in Real Systems

## 5.1 When to Optimize for Latency
- Robotics control loops  
- Real-time SLAM  
- Sensor fusion  
- Safety-critical systems  
- Interactive applications  

**Goal:** Minimize delay.

---

## 5.2 When to Optimize for Throughput
- Deep learning inference  
- Video analytics  
- CUDA batch processing  
- Offline training  
- High‑volume data pipelines  

**Goal:** Maximize total work done.

---

# 6. Summary Table

| Metric | Latency | Throughput |
|--------|---------|------------|
| Meaning | Time for one operation | Operations per second |
| Optimized by | CPUs, control loops | GPUs, batch processing |
| Jetson Example | SLAM loop | TensorRT inference |
| Improves | Responsiveness | Total performance |
| Hurts | Throughput | Latency |

---

# 7. Key Takeaway

> **Latency is about speed of one task.  
> Throughput is about speed of many tasks.  
> CPUs optimize latency; GPUs optimize throughput.  
> Jetson workloads must choose based on application needs.**



Small, architecture-aware changes often yield large, measurable gains.

# 1. Mermaid Diagram — Latency vs Throughput

```mermaid
graph TD

A[Latency Focus] --> B[Minimize time per operation]
A --> C[Low jitter / predictable timing]
A --> D[CPU-optimized workloads]
A --> E[Robotics control loops, SLAM]

F[Throughput Focus] --> G[Maximize operations per second]
F --> H[High parallelism / batching]
F --> I[GPU-optimized workloads]
F --> J[Deep learning inference, video analytics]

A ---|Tradeoff| F
```mermaid



