# Study Note: Tasks and Scheduling in Embedded & Real-Time Systems

## 1. What Is a Task?

A **task** (also called a thread or process, depending on the OS) is the smallest unit of execution that the RTOS can schedule on the CPU.

A task typically includes:
- **Code** (the function it runs)
- **Stack** (local variables, return addresses)
- **Context** (CPU registers, program counter)
- **Priority** (used by the scheduler)
- **State** (running, ready, blocked, suspended)

### Common Task States
- **Running**: Task currently executing on the CPU.
- **Ready**: Task can run but is waiting for CPU time.
- **Blocked/Waiting**: Waiting for an event (mutex, semaphore, I/O).
- **Suspended**: Disabled by the application.
- **Terminated**: Finished execution.

### Why Tasks Exist
Tasks allow:
- Concurrency  
- Separation of responsibilities  
- Real-time responsiveness  
- Predictable timing behavior  

Examples:
- Sensor reading task  
- Motor control task  
- Communication task  
- Logging task  

---

## 2. What Is Scheduling?

**Scheduling** is the RTOS mechanism that decides *which task runs at any given moment*.

The scheduler enforces:
- Priority rules  
- Timing guarantees  
- Deadlines  
- Resource fairness  

Scheduling is what makes an RTOS “real-time”.

---

## 3. Types of Scheduling

### 3.1 Preemptive Scheduling
The scheduler can interrupt a running task to run a higher-priority one.

Characteristics:
- Low latency  
- Deterministic response  
- Used in most RTOS (FreeRTOS, ERIKA, VxWorks)  

### 3.2 Cooperative Scheduling
Tasks voluntarily yield the CPU.

Characteristics:
- No forced preemption  
- Simple but risky  
- A misbehaving task can block the system  

---

## 4. Scheduling Policies

### 4.1 Fixed-Priority Scheduling
Each task has a static priority.

Examples:
- **Rate Monotonic Scheduling (RMS)**  
- POSIX **SCHED_FIFO**, **SCHED_RR**

Used in:
- Automotive (OSEK/Autosar)
- Industrial control
- Most embedded RTOS

### 4.2 Dynamic-Priority Scheduling
Priorities change at runtime based on deadlines.

Example:
- **Earliest Deadline First (EDF)**  
- Linux **SCHED_DEADLINE**

Advantages:
- Higher CPU utilization  
- Better for soft real-time  

### 4.3 Round-Robin Scheduling
Each task gets a time slice in turn.

Used in:
- General-purpose OS  
- Soft real-time systems  

---

## 5. Scheduling in Multicore Systems

Multicore scheduling introduces new challenges:

### Approaches:
- **Partitioned**: Each task assigned to one core permanently.
- **Clustered**: Tasks run on a subset of cores.
- **Global**: Tasks can run on any core; migration allowed.

### Considerations:
- Cache effects  
- Migration overhead  
- Load balancing  
- Preemption points  

---

## 6. How Tasks and Scheduling Work Together

1. Tasks are created with priorities and timing requirements.
2. Scheduler maintains a **ready queue**.
3. When an event occurs (interrupt, timer tick, task unblock):
   - Scheduler checks if a higher-priority task is ready.
   - If yes → preemption occurs.
4. Context switch saves the old task state and loads the new one.
5. System continues with the highest-priority ready task.

This cycle repeats continuously and deterministically.

---

## 7. Why This Matters in Real-Time Systems

Real-time correctness depends on:
- Tasks being well-defined  
- Scheduling being predictable  
- Preemption being controlled  
- Timing analysis being possible  

A system is only “real-time” if:
- Tasks meet deadlines  
- Scheduling is deterministic  
- Worst-case execution times (WCET) are known  

---

## 8. Summary

| Concept | Meaning |
|--------|---------|
| Task | Unit of execution with its own context and priority |
| Scheduler | Decides which task runs next |
| Preemptive | Higher-priority tasks interrupt lower ones |
| Cooperative | Tasks yield voluntarily |
| Fixed-priority | Priorities assigned statically |
| Dynamic-priority | Priorities change based on deadlines |
| Multicore scheduling | Tasks may migrate across cores |



# Study Note: Preemption in Real-Time and Embedded Systems

## 1. What Is Preemption?
Preemption is the operating system’s ability to interrupt a currently running thread so that a higher‑priority or more urgent thread can run immediately.  
It ensures that time‑critical tasks meet their deadlines by giving them immediate access to the CPU.

---

## 2. Why Preemption Is Important
- Prevents low‑priority tasks from blocking high‑priority ones.
- Ensures responsiveness in real‑time systems.
- Enables deterministic behavior required for safety‑critical applications.

Without preemption, a long‑running or misbehaving task could delay critical operations indefinitely.

---

## 3. How Preemption Works (Mechanism)
1. A higher‑priority event occurs (interrupt, timer, or thread activation).
2. The scheduler is invoked to compare priorities.
3. If the new task has higher priority:
   - The OS saves the current thread’s context (registers, PC, stack pointer).
   - Loads the context of the higher‑priority thread.
   - CPU begins executing the new thread.

This process is known as a **context switch**.

---

## 4. Types of Preemption

### 4.1 Full Preemption
- The OS can interrupt a thread at almost any instruction.
- Used in hard real‑time RTOS (FreeRTOS, ERIKA, VxWorks, PREEMPT_RT Linux).
- **Pros:** Lowest latency, highest responsiveness.  
- **Cons:** More overhead, harder timing analysis.

### 4.2 Cooperative (Non‑Preemptive) Scheduling
- A thread runs until it voluntarily yields.
- **Pros:** Simple, predictable.  
- **Cons:** A misbehaving task can block the entire system.

### 4.3 Limited Preemption
- Preemption allowed only at safe points.
- Reduces context-switch overhead.
- Useful in multicore systems to avoid excessive migration and cache penalties.

---

## 5. Preemption and Priority Inversion
When tasks share resources, preemption can cause **priority inversion**.

### Example:
- Low‑priority task holds a mutex.
- High‑priority task needs the mutex.
- Medium‑priority task preempts the low‑priority one.
- High‑priority task is indirectly blocked.

### Solution:
**Priority inheritance**  
The low‑priority task temporarily inherits the high priority until it releases the mutex.

---

## 6. Preemption in Multicore Systems
Preemption becomes more complex when multiple cores are involved.

### Key considerations:
- All cores may be busy → OS must choose which task to preempt.
- Preempted tasks may resume on a different core (migration).
- Requires deterministic:
  - Cache handling  
  - Memory consistency  
  - Migration policies (partitioned, clustered, global)

---

## 7. Costs and Trade-Offs of Preemption
Preemption is powerful but introduces overhead:

- Context switch time
- Cache invalidation
- Pipeline flushes
- Increased jitter
- More complex timing analysis

Real-time systems must balance:
- Responsiveness  
- Predictability  
- Overhead  

---

## 8. Preemption in Linux
Linux originally had limited preemption, but modern kernels support:

- **CONFIG_PREEMPT** (kernel preemption)
- **PREEMPT_RT** (full preemption, threaded interrupts)
- **SCHED_FIFO / SCHED_RR** (fixed priority)
- **SCHED_DEADLINE** (EDF with reservations)

With PREEMPT_RT, Linux can behave like a soft or even hard RTOS.

---

## 9. When Preemption Must Be Disabled
Some sections must not be interrupted:

- Critical sections
- Interrupt handlers
- Kernel spinlocks
- Memory allocator internals

RTOS APIs provide mechanisms such as:
- `preempt_disable()` / `preempt_enable()` (Linux)
- `taskENTER_CRITICAL()` / `taskEXIT_CRITICAL()` (FreeRTOS)

---

## 10. Summary
Preemption ensures that high‑priority tasks run immediately, enabling deterministic timing behavior in real‑time systems.

### Key Points:
- Enables responsiveness and deadline compliance.
- Requires context switching.
- Interacts with priority inheritance.
- More complex in multicore systems.
- Must be used carefully to avoid overhead and jitter.

