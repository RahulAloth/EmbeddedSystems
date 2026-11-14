# Concurrency

## What is Concurrency?

**Concurrency** is the ability of a system to make progress on multiple tasks during the same time period.

It does NOT necessarily mean tasks execute at the exact same moment.

```text
Task A --->|--|    |--|    |--|
Task B ------>|--|    |--|    |
Task C ------------>|--|    |--|

CPU switches between tasks
```

The operating system rapidly switches execution between tasks, giving the illusion that they run simultaneously.

---

# Concurrency vs Parallelism

## Concurrency

Multiple tasks make progress together.

```text
Core 1

Task A
Task B
Task C

CPU switches rapidly
```

### Example

Listening to music while downloading a file.

The CPU keeps switching between both tasks.

---

## Parallelism

Multiple tasks execute literally at the same time.

```text
Core 1 -> Task A
Core 2 -> Task B
Core 3 -> Task C
```

### Example

An 8-core CPU rendering 8 video frames simultaneously.

---

# Why Do We Need Concurrency?

Without concurrency:

```text
Wait for network
Then process data
Then wait for disk
Then update UI
```

CPU spends most of its time idle.

With concurrency:

```text
Thread 1 -> Network
Thread 2 -> Disk I/O
Thread 3 -> UI
Thread 4 -> Processing
```

CPU utilization improves significantly.

---

# Where is Concurrency Used?

## Operating Systems

Linux schedules hundreds of processes concurrently.

```text
Chrome
Terminal
Music Player
VS Code
```

All appear to run at the same time.

---

## Web Servers

A server may handle thousands of clients.

```text
Client 1
Client 2
Client 3
Client 1000
```

Using threads, processes, or async I/O.

Examples:

```text
Nginx
Apache
Node.js
```

---

## Embedded Systems

Automotive ECU:

```text
Sensor Reading
CAN Communication
Diagnostics
Safety Monitoring
```

All run concurrently.

---

## GPU Drivers

Nvidia Driver:

```text
Command Submission
Memory Management
Interrupt Handling
DMA Transfers
```

Multiple activities are active concurrently.

---

## Mobile Phones

Android runs:

```text
UI Thread
Audio Playback
Bluetooth
WiFi
Background Services
```

Concurrently.

---

## Databases

MySQL/PostgreSQL:

```text
User A Query
User B Query
User C Query
```

Handled concurrently.

---

# Common Concurrency Problems

When multiple threads share resources:

```text
Shared Memory
Shared Variables
Files
Sockets
Devices
```

Problems arise.

### Race Condition

```c
counter++;
```

Two threads update the same variable.

Result becomes unpredictable.

---

### Deadlock

```text
Thread A waits for B
Thread B waits for A
```

Both get stuck forever.

---

### Starvation

```text
High-priority thread always runs

Low-priority thread never gets CPU
```

---

# How Concurrency is Achieved

## Processes

```text
Process A
Process B
Process C
```

Independent memory spaces.

---

## Threads

```text
Process
 ├── Thread 1
 ├── Thread 2
 └── Thread 3
```

Share memory.

Lightweight compared to processes.

---

## Async/Event Driven

```text
Request Sent
↓
Do Other Work
↓
Response Arrives
```

Used heavily in:

```text
Node.js
Nginx
Boost ASIO
libuv
```

---

# Synchronization in Concurrency

Because threads share data:

```text
Thread A ---> Shared Data <--- Thread B
```

Synchronization mechanisms are required.

### Mutex

```text
One thread at a time
```

### Semaphore

```text
Control N resources
```

### Condition Variable

```text
Thread signaling
```

### Atomic Operations

```text
Lock-free updates
```

---

# So Questions : 

**What is Concurrency?**

> Concurrency is the ability of a system to make progress on multiple tasks during the same period of time. It improves CPU utilization and responsiveness by allowing tasks to overlap their execution. Concurrency can be achieved using processes, threads, or asynchronous programming models. Common challenges include race conditions, deadlocks, and synchronization issues, which are addressed using mutexes, semaphores, condition variables, and atomic operations.

---

# Quick Revision

```text
Concurrency
=
Multiple tasks making progress together

Why?
----
Better CPU utilization
Higher throughput
Improved responsiveness

Where?
------
Operating Systems
Web Servers
Databases
Embedded Systems
Mobile Phones
GPU Drivers

Problems?
---------
Race Conditions
Deadlocks
Starvation

Solutions?
----------
Mutex
Semaphore
Condition Variable
Atomic Operations
```


# Concurrency Notes

## Mutex
### Synchronization

A **Mutex (Mutual Exclusion)** allows only one thread at a time to access a shared resource.

```text
Thread A -> LOCK -> Critical Section -> UNLOCK
Thread B -> Wait
```

Used to prevent race conditions.

---

## Semaphore
### Resource Management

A **Semaphore** controls access to a limited number of resources.

```text
Printer Pool = 3 Printers

Semaphore = 3

Thread 1 -> Take Printer
Thread 2 -> Take Printer
Thread 3 -> Take Printer
Thread 4 -> Wait
```

Types:

```text
Binary Semaphore
Counting Semaphore
```

---

## Condition Variable
### Thread Signaling

A **Condition Variable** allows one thread to wait until another thread signals that an event occurred.

```text
Producer -> Signal()
Consumer -> Wake Up
```

Common APIs:

```c
pthread_cond_wait()
pthread_cond_signal()
pthread_cond_broadcast()
```

---

## Atomic
### Lock-Free Operations

An **Atomic Operation** executes as a single indivisible operation.

```c
counter++;
```

is NOT atomic.

```c
atomic_fetch_add(&counter, 1);
```

is atomic.

Benefits:

```text
No Mutex
Less Overhead
Lock-Free Programming
```

---

## Race Conditions
### Detection & Fix

A **Race Condition** occurs when multiple threads access shared data simultaneously and the result depends on execution order.

Example:

```text
counter = 0

Thread A -> counter++
Thread B -> counter++

Expected = 2
Actual   = 1 or 2
```

Fix:

```text
Mutex
Semaphore
Atomic Operations
```

---

## Deadlocks
### Detection & Prevention

A **Deadlock** occurs when two or more threads wait forever for each other.

Example:

```text
Thread A:
Lock(M1)
Wait(M2)

Thread B:
Lock(M2)
Wait(M1)
```

Result:

```text
A waits for B
B waits for A
System Stuck
```

Prevention:

```text
Lock resources in same order
Use timeout
Avoid circular wait
Use trylock()
```

---

# Quick Revision

```text
Mutex
→ One thread at a time

Semaphore
→ Manage N resources

Condition Variable
→ Thread signaling

Atomic
→ Lock-free operation

Race Condition
→ Unsynchronized shared data access

Deadlock
→ Threads waiting forever
```
