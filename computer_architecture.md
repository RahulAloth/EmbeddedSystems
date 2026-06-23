# Computer Architecture → ARM Architecture → DSP Architecture
This section introduces the evolution from general-purpose CPU design to specialized compute 
architectures such as ARM, DSPs, GPUs, and NPUs. It explains why different workloads require 
different processor designs and how each architecture is optimized for its domain.

---

# 1. Normal Computer Architecture (General-Purpose CPU)

A general-purpose CPU is designed to execute a wide variety of instructions and handle diverse 
workloads. It prioritizes flexibility, programmability, and ease of compiling software.

## Key Characteristics
- Optimized for control flow and branching
- Balanced pipeline (typically 5–14 stages)
- Supports complex operating systems
- Large register file and cache hierarchy
- Good at sequential, decision-heavy workloads

## Classic 5-Stage Pipeline
1. **Instruction Fetch (IF)**
2. **Instruction Decode (ID)**
3. **Execute (EX)**
4. **Memory Access (MEM)**
5. **Write Back (WB)**

This pipeline is implemented entirely in hardware using digital logic (flip-flops, ALUs, 
multiplexers, control FSMs).

---

# 2. ARM Architecture (RISC Architecture)

ARM processors follow the **RISC (Reduced Instruction Set Computer)** philosophy.

## Why ARM is RISC
- Simple, fixed-length instructions
- Load/store architecture
- Uniform pipeline stages
- Single-cycle ALU operations
- Predictable timing

## ARM Pipeline (Typical)
- 3-stage (Cortex-M0)
- 5-stage (ARM9, Cortex-M3/M4)
- 8+ stage (Cortex-A series)

## ARM Strengths
- Low power consumption
- High efficiency
- Ideal for embedded and mobile systems
- Large ecosystem and toolchain support

ARM is a general-purpose RISC CPU, not a DSP.

---

# 3. DSP Architecture (Digital Signal Processor)

A DSP is a **specialized processor** designed for high-speed mathematical operations on continuous 
streams of data such as audio, radar, motor control, and communications.

## Why DSPs Exist
General-purpose CPUs struggle with:
- repetitive math-heavy loops
- real-time constraints
- high-throughput streaming data

DSPs solve this with specialized hardware.

## Key DSP Features
- **Single-cycle MAC (Multiply-Accumulate)**
- **Zero-overhead loops**
- **Circular and modulo addressing**
- **Harvard architecture**
- **Parallel execution units**
- **Deep deterministic pipelines**

DSPs are hardware-optimized for real-time math.

---

# 4. DSP vs RISC vs GPU vs NPU

## RISC CPU (ARM)
- General-purpose
- Good at branching and control flow
- Moderate parallelism
- Balanced pipeline

## DSP
- Optimized for real-time math
- Single-cycle MAC
- Zero-overhead loops
- Specialized addressing modes
- Deterministic timing

## GPU
- Massive parallelism (hundreds to thousands of ALUs)
- Optimized for vector and matrix operations
- High throughput, low determinism
- Ideal for graphics and parallel compute

## NPU (Neural Processing Unit)
- Specialized for neural networks
- Matrix multiply engines
- Convolution accelerators
- Low precision arithmetic (INT8, FP16)
- Extremely high parallelism

## Summary Table

| Feature | RISC CPU | DSP | GPU | NPU |
|--------|----------|-----|-----|-----|
| Purpose | General compute | Real-time math | Parallel compute | AI inference |
| MAC Units | Few | 1–2 per cycle | Many | Massive |
| Parallelism | Low | Medium | High | Very High |
| Determinism | High | Very High | Low | Medium |
| Best For | Control logic | Filters, FFT | Graphics, GPGPU | Neural nets |

---

# 5. MAC Units and FIR Filters

## MAC Unit (Multiply-Accumulate)
The MAC is the heart of every DSP:

