# Inter-Process Communication (IPC)

## What is IPC?

**Inter-Process Communication (IPC)** is a mechanism that allows two or more processes to exchange data and synchronize their actions.

In an operating system, processes run in **isolated memory spaces**. Since one process cannot directly access another process's memory, IPC mechanisms are required for communication.

### Why IPC is Needed?

- Data sharing between processes
- Process synchronization
- Resource sharing
- Client-Server communication
- Parallel and distributed computing

### IPC Classification

```text
IPC
├── Pipes
├── FIFO (Named Pipes)
├── Shared Memory
├── Message Queues
└── Unix Domain Sockets
```

---

# 1. Pipes

## Concept

A **Pipe** is the simplest IPC mechanism.

It allows communication between related processes (typically parent and child).

```text
Parent Process -----> Pipe -----> Child Process
```

### Characteristics

- Unidirectional (one-way communication)
- Temporary
- Exists only while processes are running
- Usually used between parent-child processes
- Kernel maintains an internal buffer

### System Call

```c
int pipe(int fd[2]);
```

Creates two file descriptors:

```text
fd[0] → Read end
fd[1] → Write end
```

### Example

```c
pipe(fd);

write(fd[1], "Hello", 5);
read(fd[0], buffer, 5);
```

### Advantages

✅ Simple

✅ Fast for small data

✅ Easy parent-child communication

### Disadvantages

❌ One-way communication

❌ Works mainly for related processes

---

# 2. FIFO (Named Pipe)

## Concept

FIFO stands for:

```text
First In First Out
```

A FIFO is essentially a **named pipe** stored in the filesystem.

Unlike ordinary pipes, unrelated processes can communicate through it.

```text
Process A ---> FIFO File ---> Process B
```

### Characteristics

- Bidirectional possible using two FIFOs
- Has a filename
- Can communicate between unrelated processes
- Persists until deleted

### Create FIFO

```c
mkfifo("/tmp/myfifo", 0666);
```

### Communication

Writer:

```c
write(fd, data, size);
```

Reader:

```c
read(fd, buffer, size);
```

### Example

```text
Terminal 1:
echo "Hello" > myfifo

Terminal 2:
cat myfifo
```

### Advantages

✅ Simple

✅ Processes need not be related

✅ Filesystem visible

### Disadvantages

❌ Kernel copies data

❌ Slower than shared memory

---

# 3. Shared Memory

## Concept

Shared Memory is the fastest IPC mechanism.

The operating system maps the same physical memory region into multiple process address spaces.

```text
               Shared Memory Segment
            +-------------------------+
Process A ->|                         |
            |       Shared Data       |
Process B ->|                         |
            +-------------------------+
```

### How it Works

Step 1:

Create shared memory region.

Step 2:

Attach region to multiple processes.

Step 3:

Both processes access same memory.

### System Calls

Create:

```c
shmget()
```

Attach:

```c
shmat()
```

Detach:

```c
shmdt()
```

Delete:

```c
shmctl()
```

### Example

```c
int shmid = shmget(key, 1024, 0666);

char *ptr = shmat(shmid, NULL, 0);

strcpy(ptr, "Hello");
```

### Advantages

✅ Fastest IPC

✅ No kernel copy after mapping

✅ Suitable for large data

### Disadvantages

❌ Synchronization required

❌ Risk of race conditions

Usually combined with:

```text
Semaphore
Mutex
Condition Variable
```

---

# 4. Message Queues

## Concept

Message Queues allow processes to communicate by sending structured messages.

Instead of exchanging raw bytes, processes exchange messages.

```text
Sender
   |
   v
+----------+
| Queue    |
+----------+
   |
   v
Receiver
```

### Characteristics

- Messages stored by kernel
- Asynchronous communication
- Sender and receiver need not run simultaneously
- Messages can have priorities

### System Calls

Create Queue:

```c
msgget()
```

Send Message:

```c
msgsnd()
```

Receive Message:

```c
msgrcv()
```

Delete Queue:

```c
msgctl()
```

### Message Structure

```c
struct msgbuf {
    long mtype;
    char text[100];
};
```

### Example

Sender:

```c
msgsnd(queueid, &msg, sizeof(msg), 0);
```

Receiver:

```c
msgrcv(queueid, &msg, sizeof(msg), 1, 0);
```

### Advantages

✅ Structured communication

✅ Message priorities

✅ Built-in buffering

✅ Synchronization easier

### Disadvantages

❌ Kernel overhead

❌ Slower than shared memory

---

# 5. Unix Domain Sockets

## Concept

Unix Domain Sockets provide socket-based communication between processes on the same machine.

They work similarly to network sockets but do not use the network stack.

```text
Client Process
      |
      v
Unix Domain Socket
      |
      v
Server Process
```

### Characteristics

- Local machine only
- Faster than TCP sockets
- Supports stream and datagram modes
- Allows bidirectional communication

### Socket Types

### Stream Socket

Similar to TCP:

```text
SOCK_STREAM
```

### Datagram Socket

Similar to UDP:

```text
SOCK_DGRAM
```

### Create Socket

```c
socket(AF_UNIX, SOCK_STREAM, 0);
```

### Communication Steps

Server:

```text
socket()
bind()
listen()
accept()
```

Client:

```text
socket()
connect()
```

### Advantages

✅ Full duplex communication

✅ Secure

✅ Efficient local communication

✅ Client-server architecture

### Disadvantages

❌ More complex

❌ Local machine only

---

# IPC Comparison

| IPC Method | Speed | Related Processes Required | Bidirectional | Data Size |
|------------|--------|---------------------------|--------------|-----------|
| Pipe | Medium | Yes | No | Small |
| FIFO | Medium | No | Limited | Small |
| Message Queue | Medium | No | Yes | Medium |
| Shared Memory | Very Fast | No | Yes | Large |
| Unix Domain Socket | Fast | No | Yes | Medium-Large |

---

# Some quick Tests

## Q1: Which IPC mechanism is fastest?

**Shared Memory**

Reason:

```text
No data copying between processes after memory mapping.
```

---

## Q2: Pipe vs FIFO?

### Pipe

```text
Unnamed
Parent-child communication
Temporary
```

### FIFO

```text
Named
Unrelated processes
Persistent until deleted
```

---

## Q3: Why is Shared Memory fastest?

Because both processes access the same physical memory region directly.

```text
Process A ---> Shared Memory <--- Process B
```

No repeated kernel-buffer copying.

---

## Q4: Message Queue vs Shared Memory?

### Message Queue

```text
Kernel-managed
Easy synchronization
Slower
```

### Shared Memory

```text
User-managed synchronization
Much faster
```

---

## Q5: Unix Domain Socket vs TCP Socket?

### Unix Domain Socket

```text
Same machine only
Faster
No network stack
```

### TCP Socket

```text
Across network
Uses TCP/IP stack
Slower than UDS locally
```

---

# Quick Revision

```text
IPC
│
├── Pipes
│   ├── Parent-Child
│   ├── Unnamed
│   └── One-way
│
├── FIFO
│   ├── Named Pipe
│   ├── Unrelated Processes
│   └── Filesystem Object
│
├── Shared Memory
│   ├── Fastest IPC
│   ├── Shared Physical Memory
│   └── Needs Synchronization
│
├── Message Queues
│   ├── Structured Messages
│   ├── Priority Supported
│   └── Kernel Managed
│
└── Unix Domain Sockets
    ├── Client-Server
    ├── Full Duplex
    └── Local Communication
```

### Quick Glance:

```text
Fastest IPC            → Shared Memory

Parent-Child IPC       → Pipe

Unrelated Processes    → FIFO

Message-Based IPC      → Message Queue

Client-Server IPC      → Unix Domain Socket

Need Synchronization   → Shared Memory
```

## PIPE
----
pipe()
read()
write()
close()

## FIFO
----
mkfifo()
open()
read()
write()

## SHARED MEMORY (System V)
------------------------
shmget()
shmat()
shmdt()
shmctl()

## SHARED MEMORY (POSIX)
---------------------
shm_open()
mmap()
munmap()
shm_unlink()

## MESSAGE QUEUE (System V)
------------------------
msgget()
msgsnd()
msgrcv()
msgctl()

## MESSAGE QUEUE (POSIX)
---------------------
mq_open()
mq_send()
mq_receive()
mq_close()
mq_unlink()

## UNIX DOMAIN SOCKETS
-------------------
socket()
bind()
listen()
accept()
connect()
send()
recv()
close()

## SYNCHRONIZATION
---------------
sem_wait()
sem_post()

pthread_mutex_lock()
pthread_mutex_unlock()

pthread_cond_wait()
pthread_cond_signal()
