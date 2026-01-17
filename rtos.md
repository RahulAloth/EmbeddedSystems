# Study Note: Basics of RTOS & Embedded Operating Systems

## What Makes a Good RTOS?

### 1. Multithreading & Preemption
- Supports multiple concurrent threads.
- Higher‑priority threads can interrupt lower‑priority ones.

### 2. Priority-Based Scheduling
- Essential because no fully deadline-driven OS exists in mainstream use.
- Ensures deterministic execution.

### 3. Predictable Synchronization
- Mutexes, semaphores, and IPC must behave deterministically.
- Avoids unbounded blocking.

### 4. Priority Inheritance
- Prevents priority inversion.
- Ensures timing correctness when sharing resources.

---

## Major Embedded OS Players

### Commercial RTOS
- VxWorks (Wind River)
- Integrity (Green Hills)
- QNX
- Nucleus
- LynxOS
- µC/OS-III
- PikeOS
- embOS
- OSE
- Arctic Core

### Open-Source RTOS
- FreeRTOS
- ERIKA Enterprise
- eCos
- FreeOSEK
- ChibiOS/RT
- Trampoline
- QP Framework

---

## Real-Time Linux (RT-Linux)

### Key Characteristics
- Hard real-time mini-kernel.
- Linux runs as the lowest-priority thread.
- Real-time tasks and ISRs never delayed by Linux.
- Linux thread is fully preemptible.
- Supports user-level programming.

### Why Linux Is Attractive
- Open-source, no runtime licenses.
- Modular, scalable, robust.
- Excellent networking support.
- Large developer ecosystem.

---

## Real-Time Scheduling in Linux

### Traditional Policies
- **SCHED_OTHER** — Best-effort round-robin.
- **SCHED_FIFO / SCHED_RR** — Fixed-priority POSIX real-time scheduling.

### Modern Policy: SCHED_DEADLINE
- Introduced in Linux 3.14.
- Based on EDF (Earliest Deadline First).
- Provides temporal isolation.
- Uses reservation-based scheduling (budget + period).
- Updated with GRUB algorithm in Linux 4.13.

---

## Requirements for Choosing an RTOS

### 1. Programming Model
- Must support C/C++.
- Preferably POSIX-like.
- UNIX-like environment simplifies development.

### 2. Preemption Support
- Important for schedulability.
- In multicore systems, selective or “limited preemption” is useful.
- OS must save/restore thread context efficiently.

### 3. Migration Support
- **Partitioned** — thread fixed to one core.
- **Clustered** — thread runs on a subset of cores.
- **Global** — thread can run on any core.

### 4. Scheduling Characteristics
- **Static scheduling** — fixed priorities.
- **Dynamic scheduling** — priorities change at runtime.
- Must support predictable resource arbitration.

### 5. Timing Analysis Requirements
- OS must be deterministic and well-documented.
- No randomness in resource allocation.
- Must support:
  - Task-to-thread mapping
  - Thread-to-core mapping
  - Contract-based resource allocation (budgets)
  - Hard reservations
  - Memory isolation
  - Execution independence across cores

---

## RTOS Selection in Many-Core Systems

### Host Processor
- Linux chosen for:
  - Peripheral support
  - Communication stacks
  - HPC compatibility
- Soft real-time added via PREEMPT_RT.

### Many-Core Processor
Requirements:
- Open-source
- Lightweight
- Preemptive threads
- Active community

**Final Choice: ERIKA Enterprise**
- Smaller footprint than FreeRTOS.
- Advanced real-time features.
- Strong team expertise.

---

## Summary Table

| Concept | Key Idea |
|--------|----------|
| RTOS | Deterministic OS for real-time tasks |
| Preemption | High-priority tasks interrupt low-priority ones |
| Priority Inheritance | Prevents priority inversion |
| SCHED_DEADLINE | EDF-based Linux scheduler |
| RTOS Selection | Depends on footprint, determinism, licensing |
| Many-core RTOS | Needs migration + timing analysis support |
