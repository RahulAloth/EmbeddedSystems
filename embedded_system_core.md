# Embedded Systems Master Note  
## CPU Principles • Memory Architecture • RTOS • Interrupts • Exceptions • Watchdog • Scheduling • Performance Concepts

---

# 1. CPU Execution Model: The Core Principle

A CPU operates on a simple but powerful cycle:

### **Fetch → Decode → Execute**
1. **Fetch**  
   - CPU reads the next instruction from **Flash/ROM** via the instruction bus.
2. **Decode**  
   - Instruction is interpreted by the control unit.
3. **Execute**  
   - ALU operations  
   - Memory read/write (RAM)  
   - Peripheral access  
   - Branching  
   - Interrupt/exception handling  

This cycle repeats continuously as long as the CPU is powered.

### Key Components Involved
- **Registers**: Fastest storage (PC, SP, LR, general registers)
- **Flash/ROM**: Stores program code
- **RAM**: Stores variables, stack, heap
- **Bus Matrix**: Routes CPU requests to memory/peripherals
- **Pipeline**: Multiple instructions processed in parallel (in advanced CPUs)

---

# 2. Memory Architecture and Its Role in RTOS

RTOS behavior is tightly coupled with memory layout.

## 2.1 Memory Types
### **Flash / ROM**
- Stores:
  - Program code (`.text`)
  - Constants (`.rodata`)
  - Interrupt vector table
  - Bootloader
- CPU fetches instructions from here.

### **RAM**
- Stores:
  - Stack (per task)
  - Heap
  - Global/static variables
  - RTOS Task Control Blocks (TCBs)
  - Scheduler data structures
- CPU reads/writes data here.

### **Memory-Mapped Peripherals**
- Timers, UART, GPIO, NVIC, SysTick
- Accessed like normal memory addresses

### **Why Memory Matters for RTOS**
- Task stacks must be placed in RAM
- Context switching saves registers to RAM
- Interrupt handlers push CPU state to RAM
- RTOS scheduling tables live in RAM
- Deterministic timing requires predictable memory access

---

# 3. How RTOS Uses CPU + Memory

## 3.1 Task Creation
When a task is created:
- RTOS allocates a **stack** in RAM
- Initializes a **TCB (Task Control Block)** in RAM
- Stores:
  - Task priority
  - Task state
  - Stack pointer
  - CPU register snapshot
  - Scheduling metadata

## 3.2 Context Switching
During a context switch:
1. CPU registers are saved to the current task’s stack (RAM)
2. Scheduler selects next task
3. CPU loads registers from the next task’s stack
4. Execution resumes

This is why **RAM speed and determinism** matter.

---

# 4. Interrupts: How CPU Handles External Events

An **interrupt** is a hardware signal that forces the CPU to pause the current task and run a special function called an **ISR (Interrupt Service Routine)**.

### Workflow:
1. Interrupt occurs  
2. CPU saves current context to RAM  
3. CPU jumps to ISR address in Flash  
4. ISR runs  
5. CPU restores context from RAM  
6. Execution resumes

### Why Interrupts Matter in RTOS
- Interrupts wake tasks
- Interrupts trigger scheduling
- Timers generate periodic ticks
- RTOS often uses **SysTick** interrupt for timekeeping

### Interrupt Priority
Higher-priority interrupts can preempt lower ones.

---

# 5. Exceptions: CPU Internal Error or Event Handling

Exceptions are **CPU-generated** events (not external hardware).

Examples:
- Reset
- HardFault
- BusFault
- UsageFault
- SysTick
- Supervisor Call (SVC)

### Exception Workflow
Similar to interrupts:
- CPU saves context to RAM
- Jumps to exception handler in Flash
- Executes handler
- Restores context

### RTOS Use of Exceptions
- **SVC** is used to enter kernel mode
- **PendSV** is used for context switching
- **SysTick** is used for time slicing

---

# 6. Watchdog Timer

A **watchdog** is a hardware timer that resets the system if software becomes unresponsive.

### How It Works
- Watchdog counts down continuously
- Software must periodically “kick” or “feed” it
- If not fed → watchdog expires → system reset

### Purpose
- Prevents system lockups
- Ensures reliability in safety-critical systems
- Detects infinite loops or deadlocks

### RTOS Integration
- A high-priority watchdog task feeds the watchdog
- If scheduler stalls → watchdog resets system

---

# 7. Scheduling in RTOS

Scheduling decides **which task runs next**.

## 7.1 Types of Scheduling
### **Preemptive Scheduling**
- Higher-priority task interrupts lower-priority one
- Most RTOS use this

### **Cooperative Scheduling**
- Tasks yield voluntarily
- Rare in real-time systems

### **Fixed-Priority Scheduling**
- Priority assigned at design time
- Used in FreeRTOS, OSEK, ERIKA

### **Dynamic Scheduling**
- Priorities change based on deadlines
- Example: EDF (Earliest Deadline First)

---

# 8. Sleep States, Idle Task, and Power Saving

### **Sleep / Idle Mode**
When no tasks are ready:
- RTOS runs the **idle task**
- CPU enters low-power mode (WFI: Wait For Interrupt)

### Benefits
- Saves power
- Reduces heat
- Extends battery life

### Wake-Up
- Interrupt occurs (timer, GPIO, UART)
- CPU wakes and resumes scheduling

---

# 9. Throughput, Latency, and Real-Time Performance

## 9.1 Throughput
Amount of work done per unit time.

Affected by:
- CPU frequency
- Memory speed
- Cache behavior
- Interrupt load
- Scheduling overhead

## 9.2 Latency
Time between event and response.

Types:
- **Interrupt latency**
- **Scheduling latency**
- **Context switch time**

## 9.3 Jitter
Variation in timing.

Real-time systems aim for **low jitter**.

---

# 10. How Everything Connects (Unified View)

### CPU
- Executes instructions from Flash
- Reads/writes data from RAM
- Handles interrupts/exceptions
- Performs context switching

### Memory
- Flash holds code and vectors
- RAM holds stacks, TCBs, scheduler data
- Peripherals mapped into memory space

### RTOS
- Manages tasks
- Uses interrupts for timing
- Uses exceptions for context switching
- Stores all runtime state in RAM

### Watchdog
- Ensures system does not hang

### Scheduler
- Chooses which task runs
- Uses priorities, deadlines, or time slices

### Performance Concepts
- Throughput = work done
- Latency = response time
- Jitter = timing variation
- Sleep states = power saving

---

# 11. Summary Table

| Component | Role |
|----------|------|
| CPU | Executes instructions, handles interrupts, runs tasks |
| Flash | Stores program code and vectors |
| RAM | Stores stacks, variables, RTOS data |
| Interrupts | External events that preempt tasks |
| Exceptions | CPU internal events (faults, SVC, SysTick) |
| Watchdog | Resets system if software hangs |
| Scheduler | Chooses next task to run |
| Sleep States | Reduce power when idle |
| Throughput | Work done per unit time |
| Latency | Time to respond to events |

