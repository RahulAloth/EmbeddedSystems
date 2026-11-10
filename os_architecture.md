# 🧠 Operating System Architecture  
A deep, structured overview of all major components of an OS.

---

# 📦 1. Kernel (Core of the OS)

The **kernel** is the privileged core of the operating system.  
It directly manages hardware and provides essential low‑level services.

### Responsibilities
- Process scheduling  
- Memory management  
- Interrupt handling  
- Device access  
- System calls  
- Security enforcement  

### Kernel Types
- **Monolithic Kernel** — Linux, Android  
- **Microkernel** — QNX, MINIX, Integrity  
- **Hybrid Kernel** — Windows NT, macOS, iOS  
- **Exokernel/Nanokernel** — seL4, Jailhouse

---

# 🧩 2. System Libraries

System libraries provide reusable APIs so applications don’t need to interact with the kernel directly.

### Examples
- `libc` — C standard library  
- `libm` — math library  
- `OpenSSL` — cryptography  
- `libbinder` — Android IPC  
- `liblog` — Android logging  

---

# ⚙️ 3. System Services / Daemons

Background processes that provide OS functionality.  
They run in **user space**, not inside the kernel.

### Examples
- `init` — boot process  
- `systemd` — Linux service manager  
- `Zygote` — Android app launcher  
- `MediaServer` — Android audio/video  
- `SurfaceFlinger` — Android display compositor  
- `NetworkManager` — network control  

---

# 🧱 4. Hardware Abstraction Layer (HAL)

HAL provides a **standard interface** to hardware, hiding hardware differences from upper layers.

### Android HAL Examples
- Camera HAL  
- Audio HAL  
- GPS HAL  
- Sensor HAL  
- Bluetooth HAL  

HAL makes the OS portable across different hardware platforms.

---

# 🔌 5. Device Drivers

Drivers control hardware devices.

### Examples
- GPU driver  
- WiFi driver  
- Touchscreen driver  
- Camera driver  
- USB driver  
- Storage driver  

### Driver Location by OS Type

| OS Type | Driver Location |
|--------|------------------|
| Monolithic (Linux/Android) | Inside kernel |
| Microkernel (QNX) | Outside kernel (user space) |
| Hybrid (Windows/macOS) | Mostly inside kernel |

---

# 🧰 6. User‑Space Frameworks & Applications

These are the highest-level components of the OS.

### Android Examples
- **ART Runtime** — runs Java/Kotlin apps  
- **Android Framework** — ActivityManager, WindowManager  
- **System UI**  
- **Launcher**  
- **Apps** — Chrome, WhatsApp, Maps  

These sit on top of HAL, drivers, and the kernel.

---

# 🧩 Full Android OS Architecture

```
+-------------------------------------------+
| Apps (Chrome, WhatsApp, Maps)             |
+-------------------------------------------+
| Android Framework (ActivityManager, etc.) |
+-------------------------------------------+
| ART Runtime (Java/Kotlin VM)              |
+-------------------------------------------+
| Android HAL (Camera, Audio, Sensors)      |
+-------------------------------------------+
| Linux Kernel (Monolithic)                 |
|  - Drivers                                |
|  - Scheduler                              |
|  - Memory Manager                         |
|  - Binder IPC                             |
|  - SELinux                                |
+-------------------------------------------+
| Hardware                                  |
+-------------------------------------------+
```

---

# 📊 Summary Table


| OS Part | Role | Android Example |
|---------|------|------------------|
| **Kernel** | Core hardware control | Linux kernel |
| **System Libraries** | APIs for apps | libc, libbinder |
| **System Services** | Background OS processes | Zygote, MediaServer |
| **HAL** | Hardware abstraction | Camera HAL |
| **Drivers** | Hardware control | GPU, WiFi drivers |
| **User Framework** | App management | Android Framework |
| **Apps** | User programs | WhatsApp, Chrome |

---

```
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
