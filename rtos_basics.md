# Study Note: Internal Memory (RAM, ROM) and How the CPU Accesses Them

## 1. What Is Internal Memory?
Internal memory refers to memory located **inside the microcontroller or processor chip**.  
It is directly accessible by the CPU through internal buses.

Internal memory typically includes:
- **ROM / Flash** – non-volatile program storage
- **RAM (SRAM)** – volatile data storage
- **Registers** – fastest memory, inside the CPU core
- **Caches** – small, fast memory between CPU and RAM (in advanced MCUs/MPUs)

---

## 2. ROM / Flash Memory
### What It Is
- **Non-volatile** memory (retains data without power)
- Stores:
  - Program code (firmware)
  - Constants
  - Bootloader
  - Lookup tables

### Characteristics
- Read-only during normal execution
- Slow to write, fast to read
- Accessed via the **instruction bus** (I‑Bus)

### Typical Use
- CPU fetches instructions from Flash
- Startup code and interrupt vectors stored here

---

## 3. RAM (SRAM)
### What It Is
- **Volatile** memory (loses data when power is off)
- Stores:
  - Stack
  - Heap
  - Global variables
  - Temporary data

### Characteristics
- Much faster than Flash
- Read/write
- Accessed via the **data bus** (D‑Bus)

### Typical Use
- CPU reads/writes variables
- Function calls push/pop stack frames
- RTOS stores task control blocks (TCBs) here

---

## 4. CPU Memory Access: Basic Workflow

### 4.1 CPU Fetch–Decode–Execute Cycle
Every instruction follows this cycle:

1. **Fetch**  
   CPU reads the next instruction from Flash/ROM into the instruction register.

2. **Decode**  
   CPU decodes what the instruction means.

3. **Execute**  
   CPU performs the operation:
   - ALU operations
   - Memory read/write
   - Branching
   - Peripheral access

This cycle repeats continuously.

---

## 5. How CPU Accesses ROM and RAM

### 5.1 Memory Map
All memory (Flash, RAM, peripherals) is placed in a **single address space**.

Example (simplified):
```
0x0000_0000 – 0x0003_FFFF : Flash (ROM)
0x2000_0000 – 0x2000_FFFF : SRAM (RAM)
0x4000_0000 – 0x4000_FFFF : Peripherals
```

The CPU does not “know” what is Flash or RAM.  
It simply reads/writes addresses.  
The **bus matrix** routes the request to the correct memory block.

---

## 6. Instruction Access vs Data Access

### Instruction Access (I‑Bus)
- CPU fetches instructions from Flash/ROM.
- Happens every cycle.
- Often cached for speed.

### Data Access (D‑Bus)
- CPU reads/writes variables in RAM.
- Load/store instructions use this bus.

Some MCUs have **Harvard architecture**:
- Separate buses for instructions and data → faster parallel access.

---

## 7. Example Workflow: CPU Running a Program

### Step-by-step:
1. **Reset occurs**  
   CPU jumps to reset vector in Flash.

2. **Startup code runs**  
   Initializes RAM, stack pointer, global variables.

3. **Main program begins**  
   CPU fetches instructions from Flash.

4. **Variables stored in RAM**  
   CPU loads/stores data via D‑Bus.

5. **Function call**  
   Stack frame created in RAM.

6. **Interrupt occurs**  
   CPU saves context in RAM and jumps to ISR in Flash.

7. **RTOS scheduling**  
   Task control blocks stored in RAM  
   Task code stored in Flash

---

## 8. Why RAM and ROM Are Separate
| Feature | ROM/Flash | RAM |
|--------|-----------|-----|
| Volatility | Non-volatile | Volatile |
| Speed | Slower | Faster |
| Usage | Program code | Variables, stack |
| Write | Rare, slow | Frequent, fast |
| Cost | Cheap | Expensive |

Separation allows:
- Cheap program storage
- Fast runtime data access
- Predictable timing

---

## 9. Internal Memory in Microcontrollers
Typical MCU internal memory includes:
- **Flash** (32 KB – 2 MB)
- **SRAM** (4 KB – 512 KB)
- **EEPROM** (optional)
- **Caches** (in high-end MCUs)
- **Register banks**
- **Peripheral memory-mapped registers**

All accessed through the same address space.

---

## 10. Summary

- **ROM/Flash** stores program code; CPU fetches instructions from it.
- **RAM** stores variables, stack, heap; CPU reads/writes data here.
- CPU uses a **memory map** to access all memory.
- Fetch–decode–execute cycle drives instruction execution.
- Internal buses route CPU requests to Flash, RAM, or peripherals.
- RAM is fast and volatile; Flash is slow to write but permanent.

## Once Again:
# Memory Architecture in Embedded Systems  
### Master Study Note

---

## 1. Introduction
Memory architecture defines **how a microcontroller or processor stores, organizes, and accesses data and program code**.  
Understanding this is essential for embedded systems because timing, determinism, and correctness depend heavily on memory behavior.

Embedded memory typically includes:
- **ROM / Flash** – non‑volatile program storage  
- **RAM (SRAM/DRAM)** – volatile data storage  
- **Registers** – CPU internal storage  
- **Caches** – high‑speed buffers between CPU and RAM  
- **Memory‑mapped peripherals** – hardware registers exposed as memory addresses  

---

## 2. Types of Internal Memory

### 2.1 ROM / Flash (Non‑Volatile Memory)
Used for:
- Program code (firmware)
- Bootloader
- Interrupt vector table
- Constant data

Characteristics:
- Retains data without power  
- Slow to write, fast to read  
- Accessed via the **instruction bus**  
- Usually mapped at low addresses (e.g., 0x0000_0000)

---

### 2.2 RAM (Volatile Memory)
Used for:
- Stack  
- Heap  
- Global/static variables  
- Temporary buffers  
- RTOS task control blocks  

Characteristics:
- Fast read/write  
- Loses data on power loss  
- Accessed via the **data bus**  
- Typically mapped at addresses like 0x2000_0000 (ARM Cortex‑M)

Types:
- **SRAM** – fast, low power, used in MCUs  
- **DRAM** – high density, used in
```
0x0000_0000 – 0x000F_FFFF : Flash (ROM)
0x1000_0000 – 0x1000_FFFF : CCM RAM (fast RAM)
0x2000_0000 – 0x2001_FFFF : SRAM
0x4000_0000 – 0x400F_FFFF : Peripherals
0xE000_0000 – 0xE00F_FFFF : System control space (NVIC, SysTick)
```

The CPU does not “know” what is Flash or RAM.  
It simply reads/writes addresses.  
The **bus matrix** routes the request to the correct memory block.

---

## 4. CPU Access Workflow

### 4.1 Fetch–Decode–Execute Cycle
1. **Fetch**  
   CPU reads the next instruction from Flash into the instruction register.

2. **Decode**  
   Instruction is interpreted by the control unit.

3. **Execute**  
   CPU performs:
   - ALU operations  
   - Memory read/write  
   - Branching  
   - Peripheral access  

This cycle repeats continuously.

---

### 4.2 Instruction Access vs Data Access

#### Instruction Access (I‑Bus)
- CPU fetches instructions from Flash/ROM.
- Often cached for speed.

#### Data Access (D‑Bus)
- CPU reads/writes variables in RAM.
- Load/store instructions use this bus.

Some MCUs use **Harvard architecture**:
- Separate buses for instructions and data  
- Enables parallel access  
- Improves performance  

---

## 5. Startup Memory Workflow

### On Reset:
1. CPU loads the **initial stack pointer** from Flash.
2. CPU loads the **reset vector** (address of startup code).
3. Startup code:
   - Initializes RAM  
   - Copies `.data` from Flash → RAM  
   - Zeroes `.bss`  
   - Sets up C runtime environment  
4. Jumps to `main()`.

---

## 6. Memory Usage During Program Execution

### 6.1 Stack
Used for:
- Function calls  
- Local variables  
- Interrupt context  

Grows downward in memory.

### 6.2 Heap
Used for:
- Dynamic memory (`malloc`, `new`)  
- RTOS objects (queues, semaphores)  

Grows upward.

### 6.3 Global/Static Variables
Stored in RAM sections:
- `.data` (initialized)
- `.bss` (zero‑initialized)

### 6.4 Code and Constants
Stored in Flash:
- `.text` (code)
- `.rodata` (read‑only data)

---

## 7. Memory in Real-Time Systems

### Requirements:
- Deterministic access  
- No unpredictable delays  
- Avoid cache misses  
- Avoid dynamic allocation (heap fragmentation)  

### Techniques:
- Use **static allocation**  
- Use **scratchpad RAM**  
- Lock critical code in TCM (tightly coupled memory)  
- Disable caches for critical tasks  

---

## 8. Summary Table

| Memory Type | Volatile | Speed | Usage |
|-------------|----------|-------|-------|
| Flash/ROM | No | Medium | Program code, constants |
| SRAM | Yes | Fast | Variables, stack, heap |
| Registers | Yes | Very fast | CPU operations |
| Cache | Yes | Very fast | Reduce latency |
| EEPROM | No | Slow | Configuration storage |
| Memory‑mapped I/O | N/A | Medium | Peripheral control |

---

## 9. Key Takeaways
- Flash stores **code**, RAM stores **data**.  
- CPU accesses memory through a **unified address space**.  
- Fetch–decode–execute drives instruction execution.  
- Memory architecture affects **timing, determinism, and performance**.  
- Real-time systems require **predictable** memory behavior.  

