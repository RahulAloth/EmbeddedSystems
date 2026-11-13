# Memory Layout in Threads and Embedded Systems

---

## Thread Memory Model (Multithreading)

When multiple threads are spawned within a process, memory is split into what is **shared** across all threads and what is **private** to each individual thread.

### 1. What is SHARED (Process-Level)
* **Code / Text Segment:** All threads run the same compiled machine instructions from a single shared code region.
* **Heap:** There is only **one heap per process**. Any thread can allocate or access dynamic memory using pointers.
* **Global & Static Data:** Variables shared across the process scope.

### 2. What is PRIVATE (Thread-Level)
* **Stack:** Every time you spawn a new thread, it gets its **own separate stack** for local variables, function parameters, and return addresses. (Spawning 2 threads = 2 separate stacks).
* **Registers & Program Counter (PC):** Independent execution context tracking where the thread is currently executing.
* **Thread ID (TID):** Unique identifier.

---

## Explanation of Embedded System Compilation Sections

When compiling code for an embedded system (microcontroller), memory is strictly divided between **Non-Volatile Memory (Flash/ROM)** and **Volatile Memory (RAM)**.

### Standard Memory Sections

| Section | Content Type | Storage (Powered Off) | Execution/Active State (Powered On) | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`.text`** | Executable machine code / instructions | Flash / ROM | Flash / ROM | Read-only |
| **`.rodata`** | Constants (`const`) and literal strings | Flash / ROM | Flash / ROM | Read-only |
| **`.data`** | Initialized global/static variables | Flash (initial values) | RAM (active values) | Copied from Flash to RAM by startup code at boot |
| **`.bss`** | Uninitialized global/static variables | *None (zeroed at boot)* | RAM (initialized to 0) | Zeroed out automatically before `main()` |
| **Heap** | Dynamic memory allocation (`malloc`, `new`) | *None* | RAM | Grows upward; often avoided in bare-metal systems to prevent fragmentation |
| **Stack** | Local variables, function calls, ISR contexts | *None* | RAM | Grows downward from top of RAM; risk of Stack Overflow if it hits the heap |

### Key Takeaway for Embedded Systems:
* **Flash/ROM** holds permanent code and constants (`.text`, `.rodata`, and initial values for `.data`).
* **RAM** holds active runtime workspace (`.data`, `.bss`, Stack, and Heap).
