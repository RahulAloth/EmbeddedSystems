## 1. What Is a Process?

A **process** is a running instance of a program with its own:
- Virtual address space  
- Code, data, heap, stack  
- File descriptors  
- Environment variables  
- One or more threads  

### Key Properties
- **Memory isolation:** Each process has its own protected memory.  
- **Heavyweight:** Creating a process requires allocating a full address space.  
- **Independent:** A crash in one process does not crash others.  

### Example (POSIX)
```c
pid_t pid = fork();
if (pid == 0) {
    // Child process
} else {
    // Parent process
}
```

---

## 2. What Is a Task?
The meaning of **task** depends on the OS.

### Linux
Linux uses a unified structure called `task_struct`.  
A **task** can be:
- A process  
- A thread  

Linux schedules **tasks**, not processes or threads separately.

### QNX Neutrino
QNX is a microkernel.  
It schedules **threads**, not processes.

In QNX:
- **Process** = container  
- **Thread** = schedulable unit  
- **Task** = thread  

### RTOS (FreeRTOS, AUTOSAR OS, VxWorks)
A **task** is a lightweight schedulable unit:
- Has its own stack  
- Runs under a priority  
- Often no virtual memory  
- Designed for deterministic real‑time behavior  

---

## 3. Processes vs Tasks vs Threads (Clear Table)

| Concept | Definition | Memory Space | Scheduler Level | Typical OS |
|--------|------------|--------------|------------------|------------|
| **Process** | Running program | Own address space | Schedules threads | Linux, QNX |
| **Thread** | Execution path inside a process | Shared with process | Directly scheduled | Linux, QNX |
| **Task** | Schedulable unit of work | Depends on OS | Directly scheduled | RTOS, QNX, Linux |

---
# Example of Process with QNX.

## 4. QNX procnto Architecture 
QNX’s core runtime is a special process called **procnto**.

### procnto = microkernel + process manager  
They share the same address space but have different roles.

#### **Microkernel**
- Handles **threads**, **IPC**, **scheduling**, **interrupts**  
- Reached via **kernel calls**  
- Provides deterministic real‑time behavior  

#### **Process Manager**
- Handles **process creation**, **memory management**, **path resolution**  
- Reached via **messages**  
- Runs as threads inside procnto  

### Key Characteristics
- Both components run inside **process ID 1**  
- They are **tightly bound** but architecturally distinct  
- QNX schedules **threads**, not processes  
- Processes are containers; threads are the actual tasks  

---

## 5. Example: Process with Multiple Tasks (Threads)

```c
#include <pthread.h>
#include <stdio.h>

void* task_fn(void* arg) {
    printf("Task running\n");
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, task_fn, NULL);
    pthread_create(&t2, NULL, task_fn, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
```

Here:
- The **process** is the program itself.  
- The **tasks** are the two threads (`t1`, `t2`).  
- In QNX, these threads are what the microkernel schedules.

---

## 6. Why This Matters (Automotive Context)
- QNX schedules **threads**, giving deterministic timing for ADAS, IVI, gateways.  
- AUTOSAR OS uses **tasks**, not processes.  
- Linux uses **tasks** internally for both processes and threads.  
- Understanding this helps you explain scheduling, IPC, and safety behavior in interviews.

---

## 7. Summary
- A **process** is a running program with its own memory.  
- A **task** is a schedulable unit of work.  
- A **thread** is the POSIX name for a task inside a process.  
- Linux: task = process or thread.  
- QNX: task = thread.  
- RTOS: task = lightweight thread.  
- QNX’s `procnto` combines microkernel + process manager in one tightly bound unit.



# Processes vs Tasks  
A Clear, Developer‑Friendly Explanation

## 1. Overview
Modern operating systems use different terms to describe units of execution.  
Two of the most important concepts are **processes** and **tasks**.  
Although sometimes used interchangeably, they are **not the same thing**.

This document explains both concepts in a clean, engineering‑grade way.

---

## 2. What Is a Process?
A **process** is a running instance of a program.  
It has its own:

- Virtual address space  
- Code, data, heap, stack  
- File descriptors  
- Environment variables  
- One or more threads  

### Key Characteristics
- **Memory isolation:** Each process has its own protected memory.  
- **Heavyweight:** Creating a process requires allocating a full address space.  
- **Independent:** A crash in one process does not crash others.  

### Example (POSIX)
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        printf("Child process\n");
    } else {
        printf("Parent process\n");
    }
    return 0;
}
```

---

## 3. What Is a Task?
The meaning of **task** depends on the operating system.

### In Linux
A **task** is the kernel’s internal representation of a schedulable entity.  
Both **processes** and **threads** are stored in a structure called `task_struct`.

So in Linux:
- A process = a task  
- A thread = also a task  

### In QNX Neutrino
QNX is a microkernel.  
It schedules **threads**, not processes.

So in QNX:
- A process = container  
- A thread = schedulable unit  
- A task = thread  

### In RTOS (FreeRTOS, VxWorks, AUTOSAR OS)
A **task** is a lightweight schedulable unit similar to a thread, but simpler:

- Has its own stack  
- Runs under a priority  
- Often no virtual memory  
- Used for deterministic real‑time behavior  

---

## 4. Processes vs Tasks vs Threads

| Concept | Definition | Memory Space | Scheduler Level | Typical OS |
|--------|------------|--------------|------------------|------------|
| **Process** | Running program | Own address space | Schedules threads inside it | Linux, QNX, macOS |
| **Thread** | Execution path inside a process | Shared with process | Directly scheduled | Linux, QNX |
| **Task** | Schedulable unit of work | Depends on OS | Directly scheduled | RTOS, QNX, Linux |

---

## 5. Why the Distinction Matters
### In Linux
- Debugging uses **process IDs (PID)** and **thread IDs (TID)**  
- The scheduler sees everything as a **task**  
- Threads share memory; processes do not  

### In QNX
- Real‑time determinism comes from **thread‑level scheduling**  
- Tasks = threads  
- Processes are just containers  

### In Automotive RTOS (AUTOSAR OS)
- No processes  
- Only tasks  
- Each task has a fixed priority and stack  
- Deterministic timing for safety‑critical systems  

---

## 6. Example: Process with Multiple Tasks (Threads)

```c
#include <pthread.h>
#include <stdio.h>

void* task_fn(void* arg) {
    printf("Task running\n");
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, task_fn, NULL);
    pthread_create(&t2, NULL, task_fn, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
```

Here:
- The **process** is the program itself.  
- The **tasks** are the two threads (`t1`, `t2`).  

---

## 7. Summary

- A **process** is a running program with its own memory.  
- A **task** is a schedulable unit of work.  
- A **thread** is the POSIX name for a task inside a process.  
- Linux: task = process or thread.  
- QNX: task = thread.  
- RTOS: task = lightweight thread.

# Processes vs Tasks (with QNX Architecture Context)

This document explains **processes**, **tasks**, and how QNX Neutrino structures its system using the `procnto` architecture.  
It also includes examples of real QNX processes such as drivers, daemons, and managers.

---

## 1. What Is a Process?

A **process** is a running instance of a program.  
It contains:

- Its own virtual address space  
- Code, data, heap, stack  
- File descriptors  
- Environment variables  
- One or more threads  

### Key Properties
- **Memory isolation:** Each process has its own protected memory.  
- **Heavyweight:** Creating a process requires allocating a full address space.  
- **Independent:** A crash in one process does not crash others.  

### POSIX Example
```c
pid_t pid = fork();
if (pid == 0) {
    // Child process
} else {
    // Parent process
}
```

---

## 2. What Is a Task?

The meaning of **task** depends on the operating system.

### Linux
Linux uses a unified structure called `task_struct`.  
A **task** can be:
- A process  
- A thread  

Linux schedules **tasks**, not processes or threads separately.

### QNX Neutrino
QNX is a microkernel.  
It schedules **threads**, not processes.

In QNX:
- **Process** = container  
- **Thread** = schedulable unit  
- **Task** = thread  

### RTOS (FreeRTOS, AUTOSAR OS, VxWorks)
A **task** is a lightweight schedulable unit:
- Has its own stack  
- Runs under a priority  
- Often no virtual memory  
- Designed for deterministic real‑time behavior  

---

## 3. Processes vs Tasks vs Threads

| Concept | Definition | Memory Space | Scheduler Level | Typical OS |
|--------|------------|--------------|------------------|------------|
| **Process** | Running program | Own address space | Schedules threads | Linux, QNX |
| **Thread** | Execution path inside a process | Shared with process | Directly scheduled | Linux, QNX |
| **Task** | Schedulable unit of work | Depends on OS | Directly scheduled | RTOS, QNX, Linux |

---

## 4. QNX procnto Architecture

QNX’s core runtime is a special process called **procnto**.

### procnto = microkernel + process manager  
They share the same address space but have different roles.

#### Microkernel
- Handles **threads**, **IPC**, **scheduling**, **interrupts**  
- Reached via **kernel calls**  
- Provides deterministic real‑time behavior  

#### Process Manager
- Handles **process creation**, **memory management**, **path resolution**  
- Reached via **messages**  
- Runs as threads inside procnto  

### Key Characteristics
- Both components run inside **process ID 1**  
- They are **tightly bound** but architecturally distinct  
- QNX schedules **threads**, not processes  
- Processes are containers; threads are the actual tasks  

---

## 5. Examples of QNX System Processes

These are typical processes running in a QNX system:

### Disk Drivers
- `devb-eide`  
- `devb-virtio`  

### Network Stack
- `io-sock`  

### Character Drivers
- `devc-ser8250`  
- `devc-con`  

### GUI Components
- `screen`  

### Bus Managers
- `pci-server`  
- `io-usb-otg`  

### System Daemons
- `cron`  
- `sshd`  
- `mqueue`  
- `qconn`  

These processes run **outside** the microkernel and communicate using QNX’s message‑passing IPC.

---

## 6. Example: Process with Multiple Tasks (Threads)

```c
#include <pthread.h>
#include <stdio.h>

void* task_fn(void* arg) {
    printf("Task running\n");
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, task_fn, NULL);
    pthread_create(&t2, NULL, task_fn, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
```

Here:
- The **process** is the program itself.  
- The **tasks** are the two threads (`t1`, `t2`).  
- In QNX, these threads are what the microkernel schedules.

---

## 7. Summary

- A **process** is a running program with its own memory.  
- A **task** is a schedulable unit of work.  
- A **thread** is the POSIX name for a task inside a process.  
- Linux: task = process or thread.  
- QNX: task = thread.  
- RTOS: task = lightweight thread.  
- QNX’s `procnto` combines microkernel + process manager in one tightly bound unit.  
- System processes (drivers, daemons, managers) run as separate processes on top of the microkernel.

# Process Creation in Operating Systems

Process creation is the mechanism by which an operating system starts a new program. In Unix/Linux systems, the three most important concepts are:

1. `fork()`
2. `exec()`
3. `spawn()`

Understanding these is critical because almost every shell, server, and daemon relies on them.

---

# Why Do We Need Process Creation?

Suppose a shell receives:

```bash
ls -l
```

The shell itself should continue running after `ls` finishes.

So the shell:

1. Creates a new process.
2. Runs `ls` in the new process.
3. Waits for it to finish.

This is where `fork()` and `exec()` are used.

---

# Process Creation Overview

```text
Parent Process
      |
      | fork()
      v
Child Process
      |
      | exec()
      v
New Program
```

Example:

```text
Bash Shell
    |
    | fork()
    v
Child Bash
    |
    | exec("ls")
    v
ls Process
```

---

# 1. fork()

## Definition

`fork()` creates a new process by duplicating the calling process.

Prototype:

```c
pid_t fork(void);
```

After a successful fork:

- Parent continues running
- Child continues running
- Both execute from the next instruction

---

# fork() Example

```c
#include <stdio.h>
#include <unistd.h>

int main()
{
    fork();

    printf("Hello\n");

    return 0;
}
```

Output:

```text
Hello
Hello
```

Why?

```text
Before fork:

Process A

After fork:

Process A (Parent)
Process B (Child)
```

Both execute:

```c
printf("Hello");
```

---

# Memory View After fork()

Before:

```text
+----------------+
| Process A      |
| Code           |
| Heap           |
| Stack          |
+----------------+
```

After:

```text
+----------------+
| Parent         |
+----------------+

+----------------+
| Child          |
+----------------+
```

At first they seem identical.

---

# Fork Return Values

This is very important.

```c
pid_t pid = fork();
```

Parent gets:

```c
pid > 0
```

Child gets:

```c
pid == 0
```

Failure:

```c
pid < 0
```

---

# Example

```c
pid_t pid = fork();

if(pid == 0)
{
    printf("I am child\n");
}
else
{
    printf("I am parent\n");
}
```

Possible output:

```text
I am parent
I am child
```

or

```text
I am child
I am parent
```

Scheduling determines the order.

---

# Copy-on-Write (COW)

Modern Linux does NOT immediately copy all memory.

Instead:

```text
Parent
   |
   +---- Shared Pages ----+
   |                      |
Child
```

Memory is shared until one process writes.

Example:

```c
int x = 10;

fork();
```

Initially:

```text
Parent x -> Same RAM Page
Child  x -> Same RAM Page
```

If child modifies:

```c
x = 20;
```

Linux creates a separate copy.

This is called:

**Copy-On-Write (COW)**

Advantages:

- Faster fork
- Less RAM usage

---

# fork() Use Cases

## Shells

```bash
bash
zsh
sh
```

Use fork before running commands.

---

## Web Servers

Nginx, Apache, etc.

```text
Master Process
       |
       +--> Worker 1
       +--> Worker 2
       +--> Worker 3
```

Workers are often created using fork.

---

# 2. exec()

## Definition

`exec()` replaces the current process image with a new program.

Unlike `fork()`:

```text
fork() -> Creates new process

exec() -> Reuses existing process
```

---

# Example

```c
#include <unistd.h>

int main()
{
    execl("/bin/ls","ls","-l",NULL);

    return 0;
}
```

Output:

```text
(directory listing)
```

The original program disappears.

---

# What Happens?

Before:

```text
Process
 |
 +--> My Program
```

After:

```text
Process
 |
 +--> ls Program
```

Same PID.

Different executable.

---

# Memory Replacement

Before exec:

```text
Code -> My Program
Heap
Stack
```

After exec:

```text

Code -> ls
Heap
Stack

```

Everything is replaced.

---

# Important Rule

Code after successful exec never executes.

Example:

```c
printf("Before Exec\n");
execl("/bin/ls","ls",NULL);
printf("After Exec\n");

```

Output:

```text
Before Exec
(ls output)

```

Never prints:

```text
After Exec
```

---

# exec() Family

Linux provides several variants.

## execl()

Arguments listed individually.

```c
execl(
    "/bin/ls",
    "ls",
    "-l",
    NULL
);
```

## execv()

Arguments passed as array.

```c
char* args[] = {
    "ls",
    "-l",
    NULL
};

execv("/bin/ls", args);
```

---

## execvp()

Searches PATH automatically.

```c
execvp("ls", args);
```

No need for:

```text
/bin/ls
```

---

# Why fork() + exec()?

A shell cannot simply exec itself.

If Bash executes:

```text
exec(ls)
```

Bash disappears.

So Bash does:

```c
fork();
exec();
```

Result:

```text
Bash
 |
 +--> Child
          |
          +--> ls
```

Shell survives.

---

# Shell Execution Example

User enters:

```bash
python app.py
```

Internally:

```text
Bash
 |
 +--> fork()
 |
 +--> Child
        |
        +--> exec("python")
```

# 3. spawn()

## What Is spawn()?

`spawn()` combines process creation and program execution in a single call.

Conceptually:

```text
spawn() = fork() +  exec()
```

# POSIX Example

```c
posix_spawn()
```

Prototype:

```c
int posix_spawn(
    pid_t *pid,
    const*char *path,
;
```

# Example

```c

*id_t pid;

posix_spawn(
    &pid*
    "/bin/ls",
    NULL,
*   NULL,
    argv,
    environ*);

```

Immediately starts:

*``text
ls
```

# Why*spawn()?

fork* can be expensive.

Imagine*

```text
Parent Memory = 10*GB
```

Even with Copy-On*Write:

```text
fork()
*``

requires*significant kernel work.

For embedded systems:

```text
Low RAM
Low CPU
``

spawn() is often better.

---
# Windows Uses spawn-like*APIs

Windows does not use fork.

instead:

```cpp

CreateProcess()

```

directly creates a process.

Conceptually:

```text

spawn()
```

# Linux vs Windows

## Linux

```c
fork() +  exec()

```c

Traditional model.

---

## Windows

```c
CreateProcess()

```

Single operation.

---

# Comparison Table

| Feature | fork() | exec() | spawn() |
|----------|---------|---------|---------|
| Creates new process | ✅ Yes | ❌ No | ✅ Yes |
| Replaces program image | ❌ No | ✅ Yes | ✅ Yes |
| New PID | ✅ Yes | ❌ No (same PID) | ✅ Yes |
| Parent survives | ✅ Yes | N/A | ✅ Yes |
| Copies Parent Address Space | ✅ Yes (Copy-On-Write) | ❌ No | ❌ No |
| Loads New Executable | ❌ No | ✅ Yes | ✅ Yes |
| Common in Linux | Very Common | Very Common | Increasing |
| Common in Windows | ❌ No | ❌ No | Similar via CreateProcess() |
| Typical Usage | Create child process | Run another program | Create & run program in one call |

---

# Real Kernel View

When running:

```bash
gcc main.c

```

The shell performs:

```text
Shell Process
      |
      | fork()
      v
Child Shell
      |
      | exec()
      v
gcc Process      |
      | Compiling ...
      |
Exit

```

Parent shell waits

```text
wait()
```

until compilation finishes.

# Quick Revision

```text
fork()
    -> Create Child Process

exec()
    -> Replace Current Process

fork() + exec() -> Standard Unix Process Launch

spawn() -> Create and Start Process in One Call

Windows -> CreateProcess()

Linux -> fork() + exec()

```

# wait() - Process Synchronization in Linux

After learning:

```text
fork()  -> Create Child Process
exec()  -> Run New Program
```

the next piece is:

```text
wait()
```

which allows a parent process to wait for its child.

---

# Why Do We Need wait()?

Suppose a shell executes:

```bash
gcc main.c
```

The shell should not immediately display another prompt.

It should wait until compilation finishes.

Without waiting:

```text
$ gcc main.c
$
```

Prompt appears immediately even though compilation is still running.

This would be incorrect.

So the shell does:

```c
fork();
exec();
wait();
```

---

# What is wait()?

`wait()` blocks the parent process until one of its child processes exits.

Prototype:

```c
pid_t wait(int *status);
```

---

# Basic Example:

```c
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main()
{
    pid_t pid = fork();
    if(pid == 0)
    {
        printf("Child Running*n");
    }
    else
    {
        wait(NULL);
        printf("Parent Running\n");
    }
   return 0;
}

```

Output:

```text
Child Running
Parent Running
```

The parent waits until the child finishes.

---

# Timeline

Without wait():

```text
Parent
   |
fork()
   |
   +--> Child
   |
Parent Continues
```

Execution order is unpredictable.

Possible output:

```text
Parent
Child
```

or

```text
Child
Parent
```
---

With wait():

``text
Parent
   |
fork()
   |
   +--> Child
   |
wait()
   |
Child Exits
   |
Parent Continues
```

Order becomes predictable.
---

# What Happens in Kernel?

Suppose:

```c
Wait(NULL);
```

The kernel:

1. Makes parent as BLOCKED.
2. Removes parent from run queue.
3. Schedules their processes.
4. Child eventually exits.
5. Kernel wakes parent.
6. Parent resumes execution.

---

# Visual State Diagram

```text
Parent
 |
 | wait()
 |
 v

BLOCKED
 |
 |--Child exits
 |
 v

READY
 |
 v
RUNNING
```

This is an important scheduler concept.

---

# 2. Process Management

Process Management is the responsibility of the Operating System for creating, scheduling, executing, and terminating processes.

A **process** is an instance of a program in execution.

---

# Process Termination

Process termination occurs when a process completes its execution or is stopped by the operating system.

```text
Process Management
├── Process Termination
│   ├── exit()
│   ├── wait()
│   └── Zombie Process
```

---

## Why Processes Terminate

A process may terminate because:

- It completed its task.
- It encountered an unrecoverable error.
- It was terminated by another process.
- It was terminated by the operating system.

---

# exit()

## Definition

`exit()` is a system/library function used by a process to terminate itself and return a status code to the operating system.

Conceptually:

```text
Process
   |
   +--> Task Completed
   |
   +--> exit()
   |
   +--> Terminated
```

The process informs the operating system:

> "My work is finished."

---

## Responsibilities of exit()

When a process calls `exit()`:

- Execution stops.
- Open files are closed.
- Allocated resources are released.
- Exit status is stored.
- Parent process can retrieve the termination status.

---

## Example

```c
#include <stdlib.h>

int main()
{
    exit(0);
}
```

---

## Exit Status

| Status | Meaning |
|----------|----------|
| 0 | Success |
| Non-Zero | Error/Failure |

Example:

```c
exit(1);
```

indicates abnormal termination.

---

# wait()

## Definition

`wait()` allows a parent process to wait for a child process to finish and collect its termination status.

Conceptually:

```text
Parent
   |
   +---- Child
            |
            +---- exit()
                    |
                    +---- wait()
```

The parent asks:

> "Has my child finished execution?"

---

## Why wait() Exists

After a child process terminates, the parent may need:

- Exit status
- Error information
- Resource usage statistics

Before the operating system completely removes the child process.

---

## Example

```c
int status;

wait(&status);
```

The parent blocks until the child finishes.

---

## Benefits of wait()

- Synchronizes parent and child.
- Retrieves child exit status.
- Prevents zombie processes.

---

# Zombie Process

## Definition

A Zombie Process is a process that has finished execution but still has an entry in the process table because its parent has not yet collected its termination status.

---

## Lifecycle

```text
Running
   |
   +--> exit()
            |
            +--> Zombie
                     |
                     +--> wait()
                              |
                              +--> Removed
```

---

## What Remains?

The process is already dead.

Removed:

- Program code
- Memory
- CPU resources
- Open files

Retained:

- PID
- Exit status
- Accounting information

Stored in the Process Table.

---

## Why Zombies Exist

The OS keeps minimal information so the parent can retrieve:

```text
Exit Status
Return Code
Execution Information
```

using `wait()`.

---

## Example Scenario

```text
Parent
   |
   +---- Child
            |
            +---- Finishes
            |
            +---- Parent ignores wait()
            |
            +---- Zombie
```

---

## Problems with Zombies

A single zombie is usually harmless.

Many zombies can:

- Exhaust Process Table entries.
- Prevent creation of new processes.
- Cause system management issues.

---

## Preventing Zombies

### Method 1

Call:

```c
wait(NULL);
```

### Method 2

Call:

```c
waitpid(pid, NULL, 0);
```

### Method 3

Handle:

```c
SIGCHLD
```

so terminated children are automatically reaped.

---

# Zombie vs Orphan Process

## Zombie Process

```text
Parent Alive
Child Dead
```

The parent has not collected the child's exit status.

---

## Orphan Process

```text
Parent Dead
Child Alive
```

The child is adopted by:

```text
init (PID 1)
```

or

```text
systemd
```

on modern Linux systems.

---

## Comparison

| Feature | Zombie | Orphan |
|----------|----------|----------|
| Child Alive | No | Yes |
| Parent Alive | Yes | No |
| Consumes CPU | No | Possibly |
| Needs wait() | Yes | No |
| Removed By | Parent wait() | init/systemd |

---

# Key Operating System Concepts

## exit()

```text
Purpose:
Terminate the current process.

Output:
Exit status returned to parent process.
```

---

## wait()

```text
Purpose:
Allow parent to collect child termination status.

Benefit:
Prevents Zombie Processes.
```

---

## Zombie Process

```text
Definition:
A terminated child process whose parent
has not yet called wait().

State:
Dead but still present in Process Table.
```

---

# Quick Revision

```text
exit()
├─ Terminates process
├─ Releases resources
└─ Returns exit status

wait()
├─ Parent waits for child
├─ Collects exit information
└─ Removes zombie entry

Zombie Process
├─ Child terminated
├─ Parent didn't call wait()
├─ No CPU usage
└─ Process table entry remains

Orphan Process
├─ Parent terminated
├─ Child still running
└─ Adopted by init/systemd
```

## Memory Trick

```text
exit()  = "I'm Done"

wait()  = "Collect My Result"

Zombie  = "Dead but Not Cleaned Up"

Orphan  = "Alive but No Parent"
```
# Process Hierarchy

Processes are organized in a parent-child relationship.

```text
Process Management
└── Process Hierarchy
    ├── Parent
    ├── Child
    └── Orphan Process
```

---

## Parent Process

**Definition:**  
A process that creates another process.

```text
Parent
   │
   └── Child
```

### Key Points

- Creates child processes.
- Can monitor child execution.
- Can collect child status using `wait()`.

---

## Child Process

**Definition:**  
A process created by another process.

### Key Points

- Has its own PID.
- Stores Parent PID (PPID).
- Executes independently.
- May create additional child processes.

---

## Orphan Process

**Definition:**  
A child process whose parent has terminated before the child.

```text
Parent
   │
   └── Child
```

After parent exits:

```text
Parent (Terminated)

Child (Still Running)
```

The child becomes an **Orphan Process**.

---

### What Happens to an Orphan?

The operating system automatically assigns the orphan to:

```text
init (PID 1)
```

or

```text
systemd
```

on modern Linux systems.

```text
systemd/init
      │
      └── Orphan Process
```

---

### Key Points

- Child process is still running.
- Parent process has terminated.
- Automatically adopted by init/systemd.
- Usually not a problem for the operating system.

---

## Orphan vs Zombie

| Feature | Orphan | Zombie |
|----------|---------|---------|
| Parent Alive | No | Yes |
| Child Alive | Yes | No |
| Running Code | Yes | No |
| Uses CPU | Yes (if scheduled) | No |
| Adopted by init/systemd | Yes | No |

---

## Quick Revision

```text
Parent
├─ Creates child processes

Child
├─ Created by parent
└─ Has its own PID

Orphan Process
├─ Parent terminated
├─ Child still running
└─ Adopted by init/systemd

Zombie Process
├─ Parent alive
├─ Child terminated
└─ Waiting for wait()
```

### Memory Trick

```text
Orphan = Alive Child, Dead Parent

Zombie = Dead Child, Alive Parent
```
