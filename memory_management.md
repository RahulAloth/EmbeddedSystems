# Memory Management

Memory Management is one of the core responsibilities of an Operating System.

Responsibilities include:

- Memory allocation
- Memory protection
- Address translation
- Memory sharing
- Efficient RAM utilization
- Process isolation

```text
Application
      |
Virtual Memory
      |
MMU
      |
Physical Memory (RAM)
```

---

# 1 Virtual Memory

## Definition

Virtual Memory provides each process with its own independent logical address space.

Every process believes it owns the entire memory.

```text
Process A

0x00000000
    |
    |
0xFFFFFFFF

Process B

0x00000000
    |
    |
0xFFFFFFFF
```

Both processes see identical addresses, but mappings are different.

---

## Why Virtual Memory?

### Process Isolation

```text
Process A
      X
Process B
```

A process cannot directly access another process's memory.

---

### Security

```text
User Space
     X
Kernel Space
```

Protects the operating system from faulty applications.

---

### Efficient Memory Usage

Physical memory can be shared among multiple processes.

---

## Address Translation

```text
Virtual Address
       |
       v
      MMU
       |
       v
Physical Address
```

Example:

```text
Virtual Address  : 0x1000
Physical Address : 0x8A341000
```

---

## Key Point

Virtual memory is an abstraction.

Applications never directly see physical addresses.

---

# 7.2 Physical Memory

## Definition

Physical Memory is the actual RAM installed in the system.

Examples:

```text
DDR4
DDR5
LPDDR4
LPDDR5
```

---

## Memory Access Flow

```text
Application
     |
Virtual Address
     |
MMU
     |
Physical Address
     |
RAM
```

---

## Key Point

```text
Virtual Memory = Logical address space

Physical Memory = Actual RAM hardware
```

---

# 3 Paging

## Definition

Paging divides memory into fixed-size blocks.

### Virtual Memory

```text
Pages
```

### Physical Memory

```text
Frames
```

Typically:

```text
Page Size = 4 KB
```

---

## Structure

```text
Virtual Memory

+-------+
| Page0 |
+-------+
| Page1 |
+-------+
| Page2 |
+-------+

       |
       v

Page Table

       |
       v

Physical Memory

+--------+
| Frame5 |
+--------+
| Frame2 |
+--------+
| Frame8 |
+--------+
```

Pages need not be contiguous.

---

## Why Paging?

Without Paging:

```text
Need continuous memory
```

With Paging:

```text
Memory can be scattered
```

Improves memory utilization.

---

## Page Table

Stores:

```text
Virtual Page Number
          ->
Physical Frame Number
```

Example:

```text
Virtual Page 1 -> Frame 9
Virtual Page 2 -> Frame 3
Virtual Page 3 -> Frame 12
```

---

## TLB (Translation Lookaside Buffer)

A cache for page table entries.

```text
CPU
 |
TLB
 |
Page Table
 |
RAM
```

Purpose:

```text
Speed up address translation
```

---

## Interview Question

### What happens on every memory access?

```text
Virtual Address
      |
     TLB
      |
Page Table
      |
Physical Address
```

---

# 7.4 Memory Mapping

## Definition

Memory Mapping maps one memory region into another address space.

Allows:

- File Mapping
- Shared Memory
- Device Mapping

---

## File Mapping

```text
File
 |
Mapped
 |
Virtual Memory
```

Application accesses file like normal memory.

---

## Shared Memory Mapping

```text
Process A
      |
Shared Memory
      |
Process B
```

Both access the same physical pages.

---

## Device Mapping

```text
UART Registers
        |
      mmap()
        |
User Process
```

Widely used in Embedded Linux.

---

## Interview Point

Memory Mapping avoids unnecessary data copies.

---

# 7.5 malloc() / free()

## malloc()

Allocates memory dynamically from Heap.

Example:

```c
int *ptr = malloc(sizeof(int));
```

---

## free()

Releases previously allocated memory.

```c
free(ptr);
```

---

## Allocation Flow

```text
malloc()
     |
Heap
     |
Kernel
     |
Physical Memory
```

---

## Example

```c
int *arr;

arr = malloc(100 * sizeof(int));

if(arr == NULL)
{
    return -1;
}

free(arr);
```

---

## Common Mistake

```c
int *p = malloc(100);

p = NULL;
```

Memory becomes unreachable.

Memory leak occurs.

---

## Some Questions: 

### Where does malloc allocate memory?

```text
Heap Region
```

### What happens if free() is not called?

```text
Memory Leak
```

---

# 7.6 mmap()

## Definition

`mmap()` maps files or devices into a process's virtual address space.

Prototype:

```c
void *mmap(
    void *addr,
    size_t length,
    int prot,
    int flags,
    int fd,
    off_t offset
);
```

---

## Traditional File Access

```text
Disk
 |
read()
 |
Kernel Buffer
 |
User Buffer
```

Multiple copies occur.

---

## mmap()

```text
File
 |
Page Cache
 |
Memory Mapping
 |
Application
```

File behaves like memory.

---

## Example

```c
int fd = open("file.txt", O_RDONLY);

char *data = mmap(
    NULL,
    4096,
    PROT_READ,
    MAP_PRIVATE,
    fd,
    0
);
```

---

## Uses

### Shared Memory

```c
MAP_SHARED
```

Used for IPC.

---

### Device Mapping

```text
GPIO
UART
SPI
PCIe
```

Hardware registers mapped into Virtual Memory.

---

### Database Systems

```text
SQLite
PostgreSQL
Redis
```

Often use mmap for high performance.

---


### Why use mmap instead of read()?

```text
Less copying
Better performance
Direct access to data
```

---

# 7.7 Copy-On-Write (COW)

One of the most important Linux concepts.
## Problem

Suppose a process has:

```text
2 GB Memory
```

When fork() is called:

```c
pid_t pid = fork();
```

Should Linux copy all 2 GB immediately?

```text
No
```

That would be extremely slow.

---

## Solution: Copy-On-Write

Parent and Child initially share pages.

```text
Parent
    |
 Shared Pages
    |
Child
```

No memory copy occurs initially.

---

## Before Write

```text
Parent -> Page A

Child  -> Page A
```

Both point to the same physical page.

---

## After Child Modifies Data

```text
Child Writes
       |
Page Fault
       |
Kernel Creates Copy
```

Now:

```text
Parent -> Page A

Child  -> Page B
```

Separate copies exist.

---

## Visualization

### Before Write

```text
Parent
   |
   +------+
          |
          v
      Page X
          ^
          |
   +------+
   |
Child
```

---

### After Write

```text
Parent ---> Page X

Child ----> Page Y
```

New page created only when modification occurs.

---

## Advantages

### Faster fork()

```text
No immediate memory copy
```

---

### Reduced RAM Usage

```text
Memory copied only when necessary
```

---

### Better Performance

Especially useful for:

```text
fork()
+
exec()
```

Because:

```text
Child usually replaces itself
with a new program via exec()
```

No copy is ever needed.

---

### Why is fork() efficient in Linux?

Because Linux uses:

```text
Copy-On-Write (COW)
```

which delays memory copying until a page is modified.

---

# Memory Leaks

## Definition

Memory is allocated but never released.

Example:

```c
void bad()
{
    int *p = malloc(100);

    return;
}
```

Allocated memory becomes unreachable.

---

## Consequences

```text
Increasing Memory Usage

System Slowdown

Out Of Memory (OOM)

Application Crash
```

---

## Detection

### Valgrind

```bash
valgrind --leak-check=full app
```

### Address Sanitizer

```bash
gcc -fsanitize=address main.c
```

---

# Complete Linux Process Memory Layout

```text
High Address
+-----------------------+
| Kernel Space          |
+-----------------------+
| Stack                 |
|           ↓           |
+-----------------------+
| Shared Libraries      |
+-----------------------+
| mmap Region           |
+-----------------------+
|                       |
| Free Area             |
|                       |
+-----------------------+
| Heap                  |
|           ↑           |
+-----------------------+
| BSS Segment           |
+-----------------------+
| Data Segment          |
+-----------------------+
| Text / Code Segment   |
+-----------------------+
Low Address
```

---

# Quick Revision

```text
Memory Management
│
├── Virtual Memory
│   ├── Logical Address Space
│   ├── Process Isolation
│   └── MMU Translation
│
├── Physical Memory
│   └── Actual RAM
│
├── Paging
│   ├── Pages
│   ├── Frames
│   ├── Page Tables
│   └── TLB
│
├── Memory Mapping
│   ├── Files
│   ├── Shared Memory
│   └── Devices
│
├── malloc/free
│   ├── Heap Allocation
│   └── Dynamic Memory
│
├── mmap()
│   ├── File Mapping
│   ├── IPC
│   └── Hardware Access
│
├── Copy-On-Write
│   ├── fork()
│   ├── Shared Pages
│   └── Delayed Copy
│
└── Memory Leaks
    ├── Lost Heap Memory
    ├── OOM Issues
    └── Valgrind / ASan
```
