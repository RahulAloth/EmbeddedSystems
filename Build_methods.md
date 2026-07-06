# 📦 Build Method & Memory Sections (Compiler → Assembler → Linker)

A typical embedded/Linux/automotive build process (MCU, R‑Car SoC, IVI, ADAS) produces an executable with well‑defined memory sections such as **.text**, **.data**, **.bss**, **.rodata**, etc.  
These sections are created through the standard pipeline:
```
Source Code → Compiler → Assembler → Linker → ELF/BIN/HEX
```

---

## 🔧 Build Pipeline Overview

### 1. Compilation (C/C++ → Assembly)
The compiler converts your `.c` / `.cpp` files into assembly and generates object files (`.o`).

- Performs optimizations  
- Separates code and data  
- Generates section metadata  

### 2. Assembly (Assembly → Machine Code)
The assembler converts assembly into machine code and produces:

- `.text`  
- `.data`  
- `.bss`  
- `.rodata`  

### 3. Linking (Combine all `.o` → final ELF)
The linker merges all object files and libraries into a final executable:

- ELF binary  
- Section table  
- Memory map  
- Optional BIN/HEX/SREC for flashing  

---

# 🧩 Memory Sections Explained

### **.text** — Program Instructions  
Contains compiled machine code.  
Marked **read‑only + executable**.

### **.data** — Initialized Global Variables  
Stored in RAM, but initialized from flash.

Example:
```c
int speed = 50;
```
### **.bss** — Uninitialized Globals

Allocated in RAM, zero‑initialized at startup.

Example:
```c
int counter;
```
### **.rodata** — Read‑Only Constants

Strings, lookup tables, calibration data.

Example:
```c
const char* msg = "Hello";
```

### **.init** — Initialization Code

Used in kernels/bootloaders.
May be discarded after boot.
.stack — Function Call Frames

Local variables, return addresses.
Grows downward.
.heap — Dynamic Memory

Allocated via malloc() / new.
Grows upward.
## 🧱 Typical Embedded Memory Layout
```c
+---------------------------+
|        Flash (ROM)        |
|---------------------------|
|   .text (code)            |
|   .rodata (constants)     |
+---------------------------+
|         RAM               |
|---------------------------|
|   .data (init vars)       |
|   .bss  (zero vars)       |
|   Heap                    |
|   Stack                   |
+---------------------------+
```

## 🚗 Automotive Context (R‑Car SoC / RH850 MCU)

### Automotive builds also generate:

    ELF → full binary with sections

    MAP file → memory layout (critical for ASIL partitioning)

    BIN/HEX/SREC → flashable images

    Linker Script (.ld) → defines where each section goes in Flash/RAM

#### This is essential for:

    Bootloader design

    Safety isolation

    Deterministic startup

    Memory protection

    IVI/ADAS reliability




