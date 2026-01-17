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


# RTOS Task Stack Layout Inside RAM  
### (ASCII Diagram)

Below is a conceptual diagram showing how multiple RTOS task stacks live inside RAM.  
Each task gets its **own private stack**, and the RTOS stores each stack in RAM along with the TCB (Task Control Block).


```Text
+-------------------------------------------------------------+
|                         RAM (Top)                           |
|                                                             |
|  0x2000_FFFF  +------------------------------------------+  |
|               |  Main Stack (MSP)                        |  |
|               |  Used by:                                |  |
|               |   - Reset/Startup code                   |  |
|               |   - Interrupts (if using MSP)            |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  Task A Stack (High Priority)            |  |
|               |  Grows downward ↓                        |  |
|               |  [Local vars, function frames, ISRs]     |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  Task B Stack                            |  |
|               |  Grows downward ↓                        |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  Task C Stack                            |  |
|               |  Grows downward ↓                        |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  Idle Task Stack                         |  |
|               |  Smallest stack in system                |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  RTOS Kernel Objects                     |  |
|               |  (Queues, Semaphores, Event Groups)      |  |
|               +------------------------------------------+  |
|                                                             |
|               +------------------------------------------+  |
|               |  Heap (malloc/new, RTOS dynamic alloc)   |  |
|               |  Grows upward ↑                          |  |
|               +------------------------------------------+  |
|                                                             |
|               |                 Free RAM                  |  |
|               |   (must not let heap & stacks collide)    |  |
|                                                             |
|  0x2000_0000  +------------------------------------------+  |
|                         RAM (Bottom)                       |
+-------------------------------------------------------------+
```

---

## Key Points

### **1. Each RTOS task has its own stack**
- Allocated in RAM
- Size defined at task creation
- Used for:
  - Local variables  
  - Function calls  
  - Interrupt context (if using PSP)  
  - RTOS context switching  

### **2. Stacks grow downward**
- From high memory → low memory  
- Prevents collision with heap (which grows upward)

### **3. The Idle Task has the smallest stack**
- Runs when no other task is ready  
- Usually minimal stack usage

### **4. The Main Stack (MSP) is separate**
- Used during reset  
- Often used by interrupts  
- Not used by RTOS tasks (they use PSP)

### **5. Heap and stacks must never collide**
- If they do → corruption → HardFault  
- RTOS often provides stack overflow detection

---
# RTOS Task Memory Layout  
## Diagram: TCB + Stack Layout Per Task

Below is a conceptual diagram showing how each RTOS task has:

- A **TCB (Task Control Block)** stored in RAM  
- A **dedicated stack** stored in RAM  
- A **saved CPU context** pushed onto the stack during context switching  

```
+-------------------------------------------------------------+
|                     RAM (Task Region)                       |
+-------------------------------------------------------------+

Task A (High Priority)
----------------------

+----------------------+        +-----------------------------+
|   TCB for Task A     |        |     Task A Stack (RAM)      |
|----------------------|        |-----------------------------|
| - Task Name          |        |  High Address (Stack Top)   |
| - Task State         |        |  +------------------------+  |
| - Priority           |        |  | Saved CPU Registers    |  |
| - Stack Pointer ---> |------> |  | (R0–R12, LR, PC, xPSR) |  |
| - Stack Base Address |        |  +------------------------+  |
| - Stack Size         |        |  |   Local Variables      |  |
| - CPU Context Info   |        |  +------------------------+  |
| - RTOS Lists/Queues  |        |  |   Function Frames      |  |
+----------------------+        |  +------------------------+  |
|  |   ISR Saved Context    |  |
|  +------------------------+  |
|  |                        |  |
|  |   (Stack grows ↓)      |  |
|  |                        |  |
|  +------------------------+  |
|  Low Address (Stack Base) |
+-----------------------------+

Task B (Medium Priority)
------------------------

+----------------------+        +-----------------------------+
|   TCB for Task B     |        |     Task B Stack (RAM)      |
|----------------------|        |-----------------------------|
| - Task Name          |        |  High Address (Stack Top)   |
| - Task State         |        |  +------------------------+  |
| - Priority           |        |  | Saved CPU Registers    |  |
| - Stack Pointer ---> |------> |  | (Context Switch Frame) |  |
| - Stack Base Address |        |  +------------------------+  |
| - Stack Size         |        |  |   Local Variables      |  |
| - CPU Context Info   |        |  +------------------------+  |
+----------------------+        |  |   Function Frames      |  |
|  +------------------------+  |
|  |   ISR Saved Context    |  |
|  +------------------------+  |
|  |                        |  |
|  |   (Stack grows ↓)      |  |
|  |                        |  |
|  +------------------------+  |
|  Low Address (Stack Base) |
+-----------------------------+

Task C (Low Priority)
----------------------

+----------------------+        +-----------------------------+
|   TCB for Task C     |        |     Task C Stack (RAM)      |
|----------------------|        |-----------------------------|
| - Task Name          |        |  High Address (Stack Top)   |
| - Task State         |        |  +------------------------+  |
| - Priority           |        |  | Saved CPU Registers    |  |
| - Stack Pointer ---> |------> |  | (Context Switch Frame) |  |
| - Stack Base Address |        |  +------------------------+  |
| - Stack Size         |        |  |   Local Variables      |  |
| - CPU Context Info   |        |  +------------------------+  |
+----------------------+        |  |   Function Frames      |  |
|  +------------------------+  |
|  |   ISR Saved Context    |  |
|  +------------------------+  |
|  |                        |  |
|  |   (Stack grows ↓)      |  |
|  |                        |  |
|  +------------------------+  |
|  Low Address (Stack Base) |
```

---

## Key Concepts Illustrated

### **1. Each Task Has Its Own TCB**
Stored in RAM, containing:
- Priority  
- State (Ready, Running, Blocked)  
- Stack pointer  
- Stack base & size  
- CPU context metadata  
- RTOS scheduling links  

### **2. Each Task Has Its Own Stack**
Used for:
- Local variables  
- Function calls  
- Interrupt context  
- Saved CPU registers during context switch  

### **3. Stack Grows Downward**
- High address → low address  
- Prevents collision with heap  

### **4. Context Switching Saves Registers on the Stack**
When switching tasks:
- CPU registers (R0–R12, LR, PC, xPSR) are pushed onto the task’s stack  
- TCB’s stack pointer is updated  
- Next task’s stack pointer is loaded  
- CPU restores registers from that stack  

### **5. TCB Points to the Top of the Stack**
This is how the RTOS knows where to resume execution.

# Cortex‑M Stack Pointer Architecture  
## Diagram: MSP vs PSP Usage in Cortex‑M

Cortex‑M CPUs have **two stack pointers**:

- **MSP (Main Stack Pointer)** — used by reset, exceptions, interrupts  
- **PSP (Process Stack Pointer)** — used by RTOS tasks (thread mode)

Below is a conceptual diagram showing how both stacks live in RAM and how the CPU switches between them.

```
+-------------------------------------------------------------+
|                         RAM (Top)                           |
|                                                             |
|  0x2000_FFFF  +------------------------------------------+  |
|               |              MSP (Main Stack)            |  |
|               |------------------------------------------|  |
|               |  Used by:                                |  |
|               |   - Reset handler                        |  |
|               |   - All exceptions (HardFault, SysTick)  |  |
|               |   - All interrupts (NVIC)                |  |
|               |   - Boot code / startup                  |  |
|               |                                          |  |
|               |  (Grows downward ↓)                      |  |
|               +------------------------------------------+  |
|                                                             |
|               |                 Free RAM                   |  |
|                                                             |
|               +------------------------------------------+  |
|               |              PSP (Process Stack)          |  |
|               |------------------------------------------|  |
|               |  Used by:                                |  |
|               |   - RTOS tasks (Thread Mode)             |  |
|               |   - User application threads             |  |
|               |                                          |  |
|               |  Each task gets its own PSP region       |  |
|               |  (Grows downward ↓)                      |  |
|               +------------------------------------------+  |
|                                                             |
|  0x2000_0000  +------------------------------------------+  |
|                         RAM (Bottom)                       |
+-------------------------------------------------------------+
```

---

# How Cortex‑M Chooses MSP vs PSP

## 1. After Reset
- CPU starts in **Thread Mode**
- Uses **MSP** by default
- Vector table loads initial MSP value

## 2. When an Exception Occurs
- CPU **always switches to MSP**
- Saves context on MSP
- Runs ISR using MSP

## 3. When Returning to Thread Mode
- CPU uses **CONTROL register** to decide:

# RTOS Ready Lists & Priority Queue  
## Diagram: How an RTOS Organizes Tasks by Priority

Most RTOS kernels (FreeRTOS, ERIKA, RTX, ThreadX) maintain **multiple ready lists**,  
one list per priority level.  
The scheduler always selects the **highest‑priority non‑empty list**.

Below is a conceptual diagram.

```
+-------------------------------------------------------------+
|                     RTOS READY LISTS                        |
|          (One linked list per priority level)               |
+-------------------------------------------------------------+

Priority 7 (Highest)
+-------------------------------+
| Ready List P7 → [T7A]→[T7B]→Ø |
+-------------------------------+
| Tasks:                        |
|   T7A, T7B                    |
| Scheduler picks: T7A          |
+-------------------------------+

Priority 6
+-------------------------------+
| Ready List P6 → [T6A]→Ø       |
+-------------------------------+
| Tasks:                        |
|   T6A                         |
+-------------------------------+

Priority 5
+-------------------------------+
| Ready List P5 → Ø             |
+-------------------------------+
| (No ready tasks)              |
+-------------------------------+

Priority 4
+-------------------------------+
| Ready List P4 → [T4A]→[T4B]→Ø |
+-------------------------------+
| Tasks:                        |
|   T4A, T4B                    |
+-------------------------------+

Priority 3
+-------------------------------+
| Ready List P3 → [T3A]→Ø       |
+-------------------------------+
| Tasks:                        |
|   T3A                         |
+-------------------------------+

Priority 2
+-------------------------------+
| Ready List P2 → Ø             |
+-------------------------------+

Priority 1
+-------------------------------+
| Ready List P1 → [T1A]→Ø       |
+-------------------------------+

Priority 0 (Idle Task)
+-------------------------------+
| Ready List P0 → [Idle]→Ø      |
+-------------------------------+
```

---

---

# How the Scheduler Chooses the Next Task



---

# How the Scheduler Chooses the Next Task

```
Scheduler Scan:
P7 → non‑empty → choose T7A
P6 → skip
P5 → skip
P4 → skip
...
P0 → idle (only if all others empty)
```

### Rules:
- Highest priority wins  
- If multiple tasks share the same priority → **round‑robin** within that list  
- If a higher‑priority task becomes ready → **preemption occurs immediately**

---

# Visual: Priority Queue as a Stack of Lists

```
Highest Priority
↓
+-------------+
P7 --> | T7A → T7B   |
+-------------+
P6 --> | T6A         |
+-------------+
P5 --> | (empty)     |
+-------------+
P4 --> | T4A → T4B   |
+-------------+
P3 --> | T3A         |
+-------------+
P2 --> | (empty)     |
+-------------+
P1 --> | T1A         |
+-------------+
P0 --> | Idle Task   |
+-------------+
↑
Lowest Priority
```


---

# How Tasks Move Between Lists

### **1. Task becomes ready**
- Added to the tail of its priority list  
- Example: `T4B` unblocks → appended to P4 list  

### **2. Task blocks (waiting for event)**
- Removed from ready list  
- Moved to a **blocked list** or **delay list**

### **3. Task yields**
- Moved to end of its ready list (round‑robin)

### **4. Task preempted**
- Its context saved  
- Next highest‑priority ready task runs  

---

# Where TCB Fits In

Each node in the ready list is a **TCB pointer**:

```
Ready List P7:
head → [TCB_T7A] → [TCB_T7B] → Ø
```

Each TCB contains:
- Stack pointer  
- Priority  
- State  
- Next pointer (for linked list)  
- CPU context info  

---

# Summary

- RTOS maintains **one ready list per priority level**  
- Scheduler always picks the **highest non‑empty list**  
- Tasks of equal priority use **round‑robin**  
- TCBs form the nodes of these lists  
- Preemption occurs when a higher‑priority task becomes ready  


# RTOS Ready Lists, Queues, and Scheduler Flow  
### (ASCII Diagram)

This diagram shows how an RTOS organizes tasks internally and how the scheduler selects the next task to run.

```
+=====================================================================+
|                         RTOS TASK MANAGEMENT                        |
+=====================================================================+

+------------------+
|   INTERRUPT /    |
|   SysTick Tick   |
+---------+--------+
|
v
+------------------+
|   Scheduler      |
| (Priority Scan)  |
+---------+--------+
|
v
+---------------------------------------------------------------------+
|                         READY LISTS (PER PRIORITY)                  |
|   Highest priority at top. Scheduler picks first non-empty list.    |
+---------------------------------------------------------------------+

Priority 7  →  [T7A] → [T7B] → Ø
Priority 6  →  [T6A] → Ø
Priority 5  →  Ø
Priority 4  →  [T4A] → [T4B] → Ø
Priority 3  →  [T3A] → Ø
Priority 2  →  Ø
Priority 1  →  [T1A] → Ø
Priority 0  →  [Idle] → Ø

Scheduler selects → T7A (highest ready)

+---------------------------------------------------------------------+
|                         BLOCKED / WAIT QUEUES                       |
|   Tasks waiting for events, semaphores, mutexes, or messages.       |
+---------------------------------------------------------------------+

Event Queue (e.g., semaphore)
[T4B] → [T3A] → Ø

Message Queue
[T6A] → Ø

Mutex Wait Queue
[T7B] → Ø

+---------------------------------------------------------------------+
|                         DELAY / TIMER LIST                          |
|   Tasks sleeping for a timeout (vTaskDelay, sleep, delayUntil).     |
+---------------------------------------------------------------------+

Delay List (sorted by wake-up time)
[T1A] (10 ms) → [T4A] (25 ms) → Ø

When timeout expires → task moves to READY LIST (its priority bucket)

+=====================================================================+
|                         SCHEDULER FLOW DIAGRAM                      |
+=====================================================================+

+---------------------------+
|   SysTick Interrupt       |
|   or Event Occurs         |
+-------------+-------------+
|
v
+---------------------------+
|   Update Delay List       |
|   Move expired tasks →    |
|   READY LIST              |
+-------------+-------------+
|
v
+---------------------------+
|   Check Event Queues      |
|   Unblock tasks waiting   |
|   for semaphores, etc.    |
+-------------+-------------+
|
v
+---------------------------+
|   Priority Scan           |
|   (Find highest ready)    |
+-------------+-------------+
|
v
+---------------------------+
|   Compare with current    |
|   running task priority   |
+-------------+-------------+
|
+-------------+-------------+
|   Preempt if needed       |
|   (PendSV triggers        |
|    context switch)        |
+-------------+-------------+
|
v
+---------------------------+
|   Load next task’s PSP    |
|   Restore CPU registers   |
+-------------+-------------+
|
v
+---------------------------+
|   Task Runs (Thread Mode) |
+---------------------------+

+=====================================================================+
|                         TASK STATE TRANSITIONS                      |
+=====================================================================+

+-----------+     Event/Timeout     +-----------+
|  Running  | --------------------> |  Ready    |
+-----------+                       +-----------+
|                                   ^
| Block (wait for event)            |
v                                   |
+-----------+     Event Occurs     +-----------+
|  Blocked  | --------------------> |  Ready    |
+-----------+                       +-----------+
|
| Sleep (delay)
v
+-----------+     Timeout          +-----------+
|  Delayed  | --------------------> |  Ready    |
+-----------+                       +-----------+
```


---

## Key Concepts Illustrated

### **1. Ready Lists**
- One list per priority.
- Scheduler always picks the highest non-empty list.
- Tasks of equal priority use round‑robin.

### **2. Blocked Queues**
Tasks waiting for:
- Semaphores  
- Mutexes  
- Message queues  
- Events  

### **3. Delay List**
- Sorted by wake-up time.
- When timeout expires → task returns to READY list.

### **4. Scheduler Flow**
- Triggered by SysTick or an event.
- Moves tasks between lists.
- Performs priority scan.
- Uses PendSV for context switching.

### **5. Task State Machine**
- Running → Blocked  
- Blocked → Ready  
- Ready → Running  
- Delayed → Ready  

---
# PendSV Context Switch Internals (Cortex‑M)
### Full ASCII Diagram — Hardware + RTOS + Stack Interaction

PendSV is the dedicated **context-switch exception** on Cortex‑M.  
It runs at the **lowest priority**, ensuring it only executes when no other interrupt is active.

Below is the complete internal flow.

```
+=====================================================================+
| 1. A Higher-Priority Task Becomes Ready (SysTick or Event)          |
+=====================================================================+

SysTick_Handler or ISR:
→ RTOS decides: "A higher-priority task should run"
→ RTOS sets PendSV pending bit:

ICSR.PENDSVSET = 1

NVIC schedules PendSV (lowest priority exception)

+=====================================================================+
| 2. Exception Entry: Hardware Saves Part of Context                  |
+=====================================================================+

CPU finishes current instruction
CPU switches to MSP (Main Stack Pointer)
CPU automatically pushes registers onto current task’s PSP:

Hardware Stacking (8 registers):

PSP-0x20 → +------------------------+
| xPSR                  |
+------------------------+
| PC (return address)   |
+------------------------+
| LR                    |
+------------------------+
| R12                   |
+------------------------+
| R3                    |
+------------------------+
| R2                    |
+------------------------+
| R1                    |
+------------------------+
| R0                    |
+------------------------+

PSP now points to the saved frame of the old task.

+=====================================================================+
| 3. PendSV_Handler Executes (Software Context Save)                  |
+=====================================================================+

PendSV_Handler runs using MSP.

RTOS saves remaining registers (software stacking):

Software Stacking (R4–R11):

PSP-0x20 → +------------------------+
| R11                   |
+------------------------+
| R10                   |
+------------------------+
| R9                    |
+------------------------+
| R8                    |
+------------------------+
| R7                    |
+------------------------+
| R6                    |
+------------------------+
| R5                    |
+------------------------+
| R4                    |
+------------------------+

RTOS updates the TCB of the old task:
TCB[current].stack_ptr = PSP

RTOS selects next task:
next = highest_priority_ready_task()

Load next task’s saved PSP:
PSP = TCB[next].stack_ptr

+=====================================================================+
| 4. PendSV Exit: Restore Context of Next Task                        |
+=====================================================================+

RTOS restores R4–R11 from PSP (software unstacking):
pop {R4–R11}

On exception exit, CPU automatically restores:
R0–R3, R12, LR, PC, xPSR

CPU switches back to Thread Mode
CPU uses PSP (Process Stack Pointer)
Execution resumes at the new task’s PC.

+=====================================================================+
| 5. New Task Begins Running                                          |
+=====================================================================+

The new task continues exactly where it left off,
with all registers restored and PSP pointing to its stack.
```

---

## Key Takeaways

### **Hardware does:**
- Switch to MSP  
- Push R0–R3, R12, LR, PC, xPSR  
- Pop them on exit  

### **RTOS does:**
- Save R4–R11  
- Update TCB with PSP  
- Choose next task  
- Restore R4–R11  
- Set PSP to next task’s stack  

### **PendSV is ideal for context switching because:**
- Lowest priority → never interrupts real interrupts  
- Triggered only when scheduler decides  
- Clean separation between ISR and task-level execution  

---

# Full RTOS Architecture Diagram  
### Tasks • Scheduler • Memory • Interrupts • TCB • Stacks • Queues • PendSV • SysTick

Below is a complete ASCII architecture diagram showing how all RTOS components interact.

```
+=====================================================================+
|                           CPU (Cortex‑M)                            |
|                 Fetch → Decode → Execute (Thread Mode)              |
+=====================================================================+

+----------------------+
|   Running Task       |
|   (Thread Mode, PSP) |
+----------+-----------+
|
v
+=====================================================================+
|                           RTOS SCHEDULER                            |
+=====================================================================+

+---------------------------+       +-----------------------------+
|   Ready Lists (per prio) |       |   Blocked / Event Queues   |
|---------------------------|       |-----------------------------|
| P7 → [T7A]→[T7B]→Ø        |       | Semaphores → [T4B]→Ø       |
| P6 → [T6A]→Ø             |       | Mutex Wait → [T7B]→Ø       |
| P5 → Ø                   |       | Msg Queue → [T3A]→Ø        |
| P4 → [T4A]→[T4B]→Ø       |       +-----------------------------+
| P3 → [T3A]→Ø             |
| P2 → Ø                   |       +-----------------------------+
| P1 → [T1A]→Ø             |       |   Delay / Timer List        |
| P0 → [Idle]→Ø            |       | [Looks like the result wasn't safe to show. Let's switch things up and try something else!]→[Looks like the result wasn't safe to show. Let's switch things up and try something else!]→Ø   |
+---------------------------+       +-----------------------------+

Scheduler chooses highest ready task → triggers PendSV if switch needed

+=====================================================================+
|                           INTERRUPT SYSTEM                          |
+=====================================================================+

+---------------------------+       +-----------------------------+
| SysTick Interrupt         |       | External Interrupts (NVIC) |
|---------------------------|       |-----------------------------|
| - Generates RTOS tick     |       | - GPIO, UART, SPI, I2C     |
| - Updates delay list      |       | - Timers, ADC, DMA         |
| - Moves tasks to READY    |       | - Wake blocked tasks        |
| - Requests PendSV         |       | - May cause preemption      |
+---------------------------+       +-----------------------------+

SysTick → Scheduler → PendSV → Context Switch

+=====================================================================+
|                       PENDSV CONTEXT SWITCH                         |
+=====================================================================+

+---------------------------+       +-----------------------------+
| Hardware Stacking         |       | Software Stacking (RTOS)   |
|---------------------------|       |-----------------------------|
| Push R0–R3, R12, LR, PC,  |       | Push R4–R11 onto PSP       |
| xPSR onto PSP             |       | Update TCB[old].SP         |
+---------------------------+       +-----------------------------+

+---------------------------+       +-----------------------------+
| Scheduler Selects Next    |       | Restore Context             |
|---------------------------|       |-----------------------------|
| PSP = TCB[next].SP        |       | Pop R4–R11                  |
|                           |       | Hardware pops R0–R3, etc.  |
+---------------------------+       +-----------------------------+

CPU returns to Thread Mode → new task runs using PSP

+=====================================================================+
|                          MEMORY ARCHITECTURE                        |
+=====================================================================+

+---------------------------+       +-----------------------------+
| Flash (ROM)               |       | RAM                         |
|---------------------------|       |-----------------------------|
| - Vector table            |       | - TCBs for all tasks        |
| - .text (code)            |       | - Task stacks (PSP regions) |
| - .rodata                 |       | - Kernel objects (queues)   |
| - .data initial values    |       | - .data, .bss               |
+---------------------------+       | - Heap (malloc/new)         |
| - MSP (interrupt stack)     |
+-----------------------------+

+=====================================================================+
|                         TASK STRUCTURE (TCB)                        |
+=====================================================================+

Each task has a TCB stored in RAM:

TCB {
* Stack Pointer (PSP)
* Stack Base & Size
* Priority
* Task State (Ready/Blocked/Delayed)
* Linked-list pointers (ready list, event list)
* CPU context metadata
}

Each task has its own stack:

    Local variables

    Function frames

    Saved registers

    ISR stacking (if using PSP)

+=====================================================================+
|                         FULL EXECUTION FLOW                         |
+=====================================================================+
```
- Task runs in Thread Mode using PSP
- SysTick fires → moves tasks between lists
- Scheduler decides a new task should run
- Scheduler sets PendSV
- PendSV saves old task context
- PendSV restores next task context
- CPU resumes next task in Thread Mode
- Interrupts may preempt tasks at any time
- RTOS manages ready/blocked/delayed lists continuously

---
