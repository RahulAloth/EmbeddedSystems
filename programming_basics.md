# Embedded Systems Master Note  
## Code → Object → Linker → Binary • Profiling • Optimization • Toolchain Workflow

---

# 1. What Is “Code” in Embedded Systems?

**Code** refers to the human‑written source files:
- `.c`, `.cpp` → C/C++ source
- `.h` → header files
- `.s` → assembly files

These files contain:
- Functions
- Variables
- Logic
- Hardware register access
- Interrupt handlers
- RTOS tasks

The CPU **cannot** execute source code directly.  
It must be translated into machine instructions.

---

# 2. Compilation: Code → Object File

The compiler (e.g., `arm-none-eabi-gcc`) converts each `.c` file into an **object file**:

### Object File (`.o`)
Contains:
- Machine code (not yet linked)
- Symbol table (function/variable names)
- Relocation entries (addresses not yet fixed)
- Section data (`.text`, `.data`, `.bss`)

Object files are **incomplete** programs.  
They cannot run alone.

---

# 3. Linking: Object Files → Final Binary

The **linker** (e.g., `ld`) combines all object files into a single executable.

### Linker Responsibilities
- Resolve symbols (match function calls to definitions)
- Assign memory addresses
- Merge sections (`.text`, `.data`, `.bss`)
- Apply relocation
- Build the final memory layout

### Linker Script (`.ld`)
Defines:
- Flash start address
- RAM start address
- Stack location
- Vector table location
- Section placement

Example:
FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
RAM   (rwx): ORIGIN = 0x20000000, LENGTH = 128K


The linker script is **critical** in embedded systems because memory is fixed and limited.

---

# 4. Final Output: Binary / HEX / ELF

After linking, you get:

### **ELF File**
- Contains machine code + debug info
- Used for debugging (GDB, OpenOCD)

### **Binary (`.bin`)**
- Raw machine code
- Loaded directly into Flash

### **Intel HEX (`.hex`)**
- ASCII representation of binary
- Used by many bootloaders

### How It Connects to Embedded Systems
- The binary is flashed into the MCU’s **Flash memory**
- CPU fetches instructions from Flash
- RAM is initialized at startup
- Execution begins at the reset vector

---

# 5. Toolchain Workflow (Full Pipeline)
```
Source Code (.c/.h)
↓
Compiler
↓
Object Files (.o)
↓
Linker + Linker Script
↓
ELF Executable
↓
Objcopy → .bin / .hex
↓
Flashing Tool (OpenOCD, ST-Link, J-Link)
↓
Microcontroller Flash Memory
↓
CPU Executes Instructions
```


---

# 6. Profiling in Embedded Systems

Profiling means **measuring where time is spent** in your code.

### Common Profiling Methods

#### 6.1 GPIO Toggling (Most Common)
- Toggle a GPIO pin at function entry/exit
- Measure with oscilloscope or logic analyzer
- Very accurate for real-time systems

#### 6.2 Cycle Counters (DWT on Cortex-M)
- Use CPU cycle counter registers
- Measure execution time in cycles

#### 6.3 Software Profiling
- Instrumentation (adds overhead)
- RTOS trace hooks
- Logging timestamps

#### 6.4 Hardware Trace (ETM, SWO)
- High-end MCUs support instruction trace
- Tools: Segger Ozone, Lauterbach

---

# 7. Code Optimization Techniques

### 7.1 Compiler Optimizations
Flags:
- `-O0` → no optimization (debugging)
- `-O1` → basic optimization
- `-O2` → balanced optimization
- `-O3` → aggressive optimization
- `-Os` → optimize for size
- `-Ofast` → fastest but unsafe

### 7.2 Algorithmic Optimization
- Replace loops with lookup tables
- Use fixed-point instead of floating-point
- Reduce branching
- Use DMA instead of CPU copying

### 7.3 Memory Optimization
- Place constants in Flash
- Use static allocation (avoid heap)
- Reduce stack usage
- Use bitfields or packed structs

### 7.4 RTOS Optimization
- Reduce context switches
- Use event groups instead of polling
- Increase tick period if possible
- Use direct-to-task notifications

---

# 8. How CPU Executes the Final Binary

### Step-by-Step:
1. MCU resets
2. CPU loads initial stack pointer from Flash
3. CPU loads reset handler address
4. Startup code initializes RAM
5. Global variables copied from Flash → RAM
6. Zero `.bss`
7. Call `main()`
8. RTOS starts (if used)
9. Scheduler selects first task
10. CPU fetches instructions from Flash and executes

---

# 9. How All Concepts Connect

| Concept | Role in Embedded System |
|--------|--------------------------|
| Source Code | Human-readable logic |
| Compiler | Converts code → object files |
| Object Files | Machine code fragments |
| Linker | Combines objects, assigns addresses |
| Linker Script | Defines memory layout |
| Binary/HEX | Final machine code |
| Flash Memory | Stores program code |
| RAM | Stores runtime data |
| CPU | Executes instructions |
| RTOS | Manages tasks, scheduling |
| Profiling | Measures performance |
| Optimization | Improves speed/size |
| Watchdog | Ensures system reliability |
| Interrupts | Handle external events |
| Exceptions | Handle CPU events |

---

# 10. Summary

- Code is compiled into object files.
- Linker + linker script create the final binary.
- Binary is flashed into MCU memory.
- CPU executes instructions from Flash.
- RAM holds runtime data.
- Profiling measures performance.
- Optimization improves speed, size, and efficiency.
- RTOS coordinates tasks, interrupts, and timing.

# Master Note: Linker Scripts and Memory Sections  
### (.text • .data • .bss • .stack • .heap • Vector Table • Startup Flow)

---

# 1. Introduction

In embedded systems, memory is limited and fixed.  
The **linker script** defines *exactly* where every part of your program lives in Flash and RAM.

A linker script controls:
- Memory layout (Flash, RAM, peripherals)
- Placement of code and data sections
- Stack and heap boundaries
- Vector table location
- Startup initialization behavior

Understanding linker scripts is essential for:
- Bare-metal programming
- RTOS porting
- Bootloader design
- Memory optimization
- Debugging crashes and hard faults

---

# 2. Memory Regions in Embedded Systems

Most microcontrollers have at least two main memory regions:

### **Flash (ROM)**
- Non-volatile
- Stores program code and constants
- Execution starts here after reset

### **RAM**
- Volatile
- Stores variables, stack, heap, RTOS data

A typical memory map:

```
FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
RAM   (rwx): ORIGIN = 0x20000000, LENGTH = 128K
```

---

# 3. What Is a Linker Script?

A linker script (`.ld`) tells the linker:

- Where memory regions start and end  
- Where to place each section (`.text`, `.data`, `.bss`, etc.)  
- Where the stack begins  
- Where the heap begins  
- Where the vector table is located  

Example structure:

```
MEMORY {
FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
RAM   (rwx): ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS {
.text : { ... } > FLASH
.data : { ... } > RAM AT > FLASH
.bss  : { ... } > RAM
}
```

---

# 4. Memory Sections Explained

## 4.1 `.text` — Code Section
Contains:
- Compiled machine instructions
- Constant data
- Interrupt vector table (usually at the start of Flash)

Characteristics:
- Read-only
- Stored in Flash
- CPU fetches instructions from here

Example:
```
.text : {
*(.isr_vector)
(.text)
(.rodata)
} > FLASH
```

---

## 4.2 `.rodata` — Read-Only Data
Contains:
- String literals
- Constant lookup tables
- `const` variables

Stored in Flash.

Often merged into `.text`.

---

## 4.3 `.data` — Initialized Data
Contains:
- Global/static variables with **initial values**

Example:
int counter = 5;


### How `.data` works:
- Initial values stored in Flash
- Copied to RAM during startup
- RAM version is used at runtime

Linker script:
```
.data : {
(.data)
} > RAM AT > FLASH
```

---

## 4.4 `.bss` — Zero-Initialized Data
Contains:
- Global/static variables initialized to zero
- Uninitialized variables

Example:
int buffer[100];
static int flag;

Characteristics:
- Stored in RAM
- Not stored in Flash (saves space)
- Zeroed by startup code

Linker script:
```
.bss : {
(.bss)
*(COMMON)
} > RAM
```

---

## 4.5 `.stack` — Stack Memory
Used for:
- Function calls
- Local variables
- Interrupt context
- RTOS task switching

Characteristics:
- Grows downward (most architectures)
- Located at the top of RAM

Example:
```
_estack = ORIGIN(RAM) + LENGTH(RAM);
```

Startup code sets:
MSP = _estack;

---

## 4.6 `.heap` — Dynamic Memory
Used for:
- `malloc()`, `new`
- RTOS objects (queues, semaphores)
- Dynamic buffers

Characteristics:
- Grows upward
- Must not collide with stack

Linker script often defines:
```
_heap_start = .;
_heap_end   = _estack - STACK_SIZE;
```

---

# 5. Vector Table Placement

The vector table contains:
- Initial stack pointer
- Reset handler
- Interrupt handlers

Placed at the start of Flash:

```
.isr_vector : {
KEEP(*(.isr_vector))
} > FLASH
```


CPU reads the first word after reset:
- Loads stack pointer
- Loads reset handler address

---

# 6. Startup Code and Section Initialization

Before `main()` runs, startup code performs:

### 1. Copy `.data` from Flash → RAM
```
for each byte in LOADADDR(.data):
copy to VMA(.data)
```


### 2. Zero `.bss`

```
memset(.bss, 0)
```

### 3. Initialize stack pointer
```
MSP = _estack
````

### 4. Call `SystemInit()` (clock setup)

### 5. Call `main()`

---

# 7. How RTOS Uses Memory Sections

### RTOS stores:
- Task stacks → RAM
- Task control blocks (TCBs) → RAM
- Scheduler structures → RAM
- ISR handlers → Flash
- Vector table → Flash

### RTOS requires:
- Enough RAM for multiple stacks
- Deterministic memory layout
- No stack/heap collision

---

# 8. Common Memory Problems and Debugging

### **Stack overflow**
Symptoms:
- HardFault
- Random crashes
- Corrupted variables

Fix:
- Increase stack size
- Use stack guards
- Enable RTOS stack checking

---

### **Heap fragmentation**
Symptoms:
- `malloc()` fails
- RTOS objects fail to allocate

Fix:
- Avoid dynamic allocation
- Use static buffers

---

### **Incorrect linker script**
Symptoms:
- Code runs from wrong address
- Startup code fails
- Interrupts not firing

Fix:
- Verify memory regions
- Check vector table placement

---

# 9. Summary Table

| Section | Stored In | Contains | Initialized By |
|--------|-----------|----------|----------------|
| `.text` | Flash | Code, constants | Compiler |
| `.rodata` | Flash | Read-only data | Compiler |
| `.data` | RAM (copied from Flash) | Initialized globals | Startup code |
| `.bss` | RAM | Zero-initialized globals | Startup code |
| `.stack` | RAM | Function frames, interrupts | Startup code |
| `.heap` | RAM | Dynamic memory | Runtime |

---

# 10. Key Takeaways

- Linker scripts define **where** everything lives in memory.
- `.text` and `.rodata` live in Flash; `.data`, `.bss`, stack, and heap live in RAM.
- Startup code copies `.data` and zeros `.bss`.
- Stack grows downward; heap grows upward.
- RTOS relies heavily on RAM layout for task stacks and scheduling.
- Understanding linker scripts is essential for debugging, optimization, and system reliability.


# Memory Layout Diagram (Embedded Systems)

Below is a conceptual diagram showing how memory is organized in a typical microcontroller  
(Flash + RAM + peripheral space).  
This matches the linker‑script master note you already have.


```Text
+-------------------------------------------------------------+
|                     FLASH / ROM (Code)                      |
|                 (Non‑volatile, read‑only)                   |
|                                                             |
|  0x0800_0000  +------------------------------------------+  |
|               |  Vector Table (Reset, IRQ handlers)      |  |
|               +------------------------------------------+  |
|               |  .text  (Program Instructions)           |  |
|               +------------------------------------------+  |
|               |  .rodata (Read‑only constants)           |  |
|               +------------------------------------------+  |
|               |  .data (Initial values for RAM variables)|  |
|               |   Stored in Flash, copied to RAM       |  |
|               +------------------------------------------+  |
|               |  (Optional) Bootloader                   |  |
|               +------------------------------------------+  |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                         RAM (Data)                          |
|                     (Volatile, read/write)                  |
|                                                             |
|  0x2000_0000  +------------------------------------------+  |
|               |  .data (Runtime copy of initialized vars)|  |
|               +------------------------------------------+  |
|               |  .bss  (Zero‑initialized variables)      |  |
|               +------------------------------------------+  |
|               |  Heap (grows upward ↑)                   |  |
|               |   malloc(), new, RTOS objects            |  |
|               +------------------------------------------+  |
|               |                                          |  |
|               |      Free RAM / Unused Space             |  |
|               |                                          |  |
|               +------------------------------------------+  |
|               |  Stack (grows downward ↓)                |  |
|               |   Local vars, function frames, ISRs      |  |
|  0x2000_FFFF  +------------------------------------------+  |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                 PERIPHERAL REGISTER SPACE                   |
|            (Memory‑mapped I/O: UART, GPIO, SPI, etc.)       |
|                                                             |
|  0x4000_0000  +------------------------------------------+  |
|               |  Peripheral Registers                     |  |
|               |  (Timers, GPIO, UART, ADC, I2C, SPI...)  |  |
|               +------------------------------------------+  |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                 SYSTEM CONTROL SPACE (SCS)                  |
|           (NVIC, SysTick, SCB, Debug, MPU, etc.)           |
|                                                             |
|  0xE000_0000  +------------------------------------------+  |
|               |  NVIC (Interrupt Controller)              |  |
|               +------------------------------------------+  |
|               |  SysTick Timer                            |  |
|               +------------------------------------------+  |
|               |  SCB (System Control Block)               |  |
|               +------------------------------------------+  |
|               |  Debug / Trace Units                      |  |
|               +------------------------------------------+  |
|                                                             |
+-------------------------------------------------------------+
```
