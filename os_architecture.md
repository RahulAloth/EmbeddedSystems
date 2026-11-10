# 🧠 Operating System Overview

An Operating System (OS) is the software layer between applications and hardware. It manages system resources and provides common services to applications.

---

# 🏛️ High-Level OS Architecture

```text
+----------------------------------+
| Applications                     |
+----------------------------------+
| Frameworks / Runtimes            |
+----------------------------------+
| System Services / Daemons        |
+----------------------------------+
| System Libraries                 |
+----------------------------------+
| Hardware Abstraction Layer (HAL) |
+----------------------------------+
| Device Drivers                   |
+----------------------------------+
| Kernel                           |
+----------------------------------+
| Hardware                         |
+----------------------------------+
```

---

# 📦 1. Kernel

The kernel is the core component of the operating system and runs with the highest privileges.

### Responsibilities

- Process management
- Thread management
- CPU scheduling
- Memory management
- Device management
- Interrupt handling
- Inter-process communication (IPC)
- Security and access control

### Kernel Types

- **Monolithic Kernel** (Linux)
- **Microkernel** (QNX, MINIX)
- **Hybrid Kernel** (Windows, macOS)
- **Exokernel / Nanokernel**

---

# 🧩 2. System Libraries

System libraries provide APIs that allow applications and services to access operating system functionality.

### Examples

- `libc`
- `libm`
- `OpenSSL`
- POSIX libraries

### Purpose

- Simplify application development
- Provide standardized interfaces
- Hide low-level kernel details

---

# ⚙️ 3. System Services / Daemons

Background processes that provide operating system functionality.

### Examples

- `init`
- `systemd`
- `NetworkManager`
- Logging services
- Time synchronization services
- Media services

### Responsibilities

- System startup
- Service management
- Networking
- Logging
- Resource monitoring

---

# 🧱 4. Hardware Abstraction Layer (HAL)

The Hardware Abstraction Layer (HAL) provides a standard interface between software components and hardware devices.

### Benefits

- Hardware independence
- Platform portability
- Simplified driver integration

### Examples

- Camera HAL
- Audio HAL
- Sensor HAL
- Bluetooth HAL

> Note: HAL is commonly used in Android and many embedded systems. Not all operating systems implement a distinct HAL layer.

---

# 🔌 5. Device Drivers

Device drivers are software components that communicate directly with hardware devices.

### Examples

- Display driver
- Audio driver
- Network driver
- USB driver
- Storage driver
- GPU driver

### Purpose

- Hardware control
- Device configuration
- Data transfer

### Driver Location by OS Type

| OS Type | Driver Location |
|----------|----------------|
| Monolithic (Linux) | Inside kernel |
| Microkernel (QNX) | Mostly user space |
| Hybrid (Windows/macOS) | Mostly kernel space |

---

# 🧰 6. Frameworks and Runtimes

Frameworks and runtimes provide higher-level services for applications.

### Examples

- Android Framework
- ART Runtime
- Java Runtime Environment (JRE)
- .NET Runtime

### Responsibilities

- Application lifecycle management
- Resource management
- UI services
- Application APIs

---

# 📱 7. Applications

Applications are user-facing programs that run on top of the operating system.

### Examples

- Web browsers
- Messaging applications
- Navigation software
- Media players
- Productivity tools

Applications access hardware indirectly through OS services, libraries, drivers, and the kernel.

---

# 📊 OS Component Summary

| Component | Primary Role |
|------------|-------------|
| Kernel | Core system management |
| Device Drivers | Hardware control |
| HAL | Hardware abstraction |
| System Libraries | Operating system APIs |
| System Services | Background functionality |
| Frameworks / Runtimes | Application support |
| Applications | User-facing software |

---

# 📚 Related Topics

The following topics are typically studied in more detail as separate subjects:

- Processes and Threads
- Scheduling
- Context Switching
- Memory Management
- Inter-Process Communication (IPC)
- Synchronization
- Interrupt Handling
- File Systems
- Device Drivers
- Linux Architecture
- QNX Architecture
- Android Architecture

---

# Key Takeaway

An Operating System can be viewed as a layered architecture:

```text
Applications
    ↓
Frameworks / Runtimes
    ↓
System Services
    ↓
System Libraries
    ↓
HAL
    ↓
Device Drivers
    ↓
Kernel
    ↓
Hardware
```

Each layer abstracts complexity from the layer above, making software development more portable, maintainable, and secure.

```text
For an Embedded systems Operating Systems can organize like this:
Operating Systems
│
├── 1. OS Fundamentals
│   ├── Kernel
│   ├── User Space
│   ├── System Calls
│   ├── Trap & Exceptions
│   └── Boot Process
│
├── 2. Process Management
│   ├── Process Creation
│   │   ├── fork()
│   │   ├── spawn()
│   │   └── exec()
│   │
│   ├── Process Termination
│   │   ├── exit()
│   │   ├── wait()
│   │   └── zombie process
│   │
│   ├── Process States
│   │   ├── Running
│   │   ├── Ready
│   │   ├── Blocked
│   │   └── Suspended
│   │
│   └── Process Hierarchy
│       ├── Parent
│       ├── Child
│       └── Orphan
│
├── 3. Thread Management
│   ├── pthread_create()
│   ├── pthread_join()
│   ├── pthread_cancel()
│   ├── Detached Threads
│   └── Thread Lifecycle
│
├── 4. CPU Scheduling
│   ├── Scheduler
│   ├── Context Switching
│   ├── Time Slicing
│   ├── Priority Scheduling
│   ├── Round Robin
│   ├── FIFO
│   └── Real-Time Scheduling
│
├── 5. Synchronization
│   ├── Mutex
│   ├── Semaphore
│   ├── Spinlock
│   ├── Condition Variables
│   ├── Reader/Writer Locks
│   └── Barriers
│
├── 6. Inter Process Communication (IPC)
│   ├── Pipes
│   ├── Named Pipes
│   ├── Message Queues
│   ├── Shared Memory
│   ├── Signals
│   ├── Sockets
│   └── QNX MsgSend/Receive/Reply
│
├── 7. Memory Management
│   ├── Virtual Memory
│   ├── Physical Memory
│   ├── Paging
│   ├── Memory Mapping
│   ├── malloc/free
│   ├── mmap()
│   └── Copy-On-Write
│
├── 8. File Systems
│   ├── VFS
│   ├── ext4
│   ├── FAT
│   ├── NFS
│   ├── File Descriptors
│   ├── open()
│   ├── read()
│   ├── write()
│   └── close()
│
├── 9. Device Drivers
│   ├── Character Drivers
│   ├── Block Drivers
│   ├── UART
│   ├── SPI
│   ├── I2C
│   ├── CAN
│   └── PCIe
│
├── 10. Interrupt Handling
│   ├── IRQ
│   ├── ISR
│   ├── Bottom Half
│   ├── SoftIRQ
│   ├── Tasklets
│   └── Interrupt Latency
│
├── 11. Networking
│   ├── TCP/IP
│   ├── UDP
│   ├── Ethernet
│   ├── Routing
│   ├── Socket API
│   └── Network Stack
│
├── 12. Security
│   ├── User Accounts
│   ├── Permissions
│   ├── ACL
│   ├── Capabilities
│   └── Authentication
│
├── 13. Real-Time Systems
│   ├── Hard Real-Time
│   ├── Soft Real-Time
│   ├── Determinism
│   ├── Priority Inversion
│   ├── Priority Inheritance
│   └── Response Time Analysis
│
├── 14. Kernel Architecture
│   ├── Monolithic Kernel
│   │   └── Linux
│   │
│   ├── Microkernel
│   │   └── QNX
│   │
│   └── Hybrid Kernel
│
└── 15. POSIX
    ├── Process APIs
    ├── Thread APIs
    ├── Scheduling APIs
    ├── IPC APIs
    ├── File APIs
    ├── Signal APIs
    └── Time APIs
```
