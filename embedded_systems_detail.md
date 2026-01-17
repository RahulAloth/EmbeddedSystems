# 📘 Importance of Embedded Systems and RTOS

## 🌍 Growing Importance of Embedded Systems
Embedded systems have become increasingly important over the last decades and are now widely used across many domains. They appear in devices such as:

- Mobile phones  
- Cars  
- Aircraft  
- Industrial control systems  

These systems often handle tasks like media management, sensor processing, or navigation. Because they typically operate under strict hardware constraints—limited memory and low computational power—classic General Purpose Operating Systems (GPOS) are unsuitable. GPOS require more memory and do not provide deterministic real-time behavior.

---

## ⚙️ Why Real-Time Operating Systems (RTOS)?
Real-Time Operating Systems are designed specifically for constrained embedded environments. Key characteristics include:

- Very small memory footprint  
- Deterministic, real-time task execution  
- Priority-based scheduling  
- Hardware-specific optimization  

RTOS implementations are often tightly coupled to the underlying hardware, enabling predictable and efficient operation even with limited resources.

---

## ✈️ Safety-Critical Applications
RTOS are widely used in safety‑critical environments such as:

- Aviation  
- Automotive systems  
- Power plants  
- Medical devices  

In these domains, missing a deadline can lead to severe consequences. Therefore, an RTOS must:

- Schedule tasks with strict timing guarantees  
- Handle unexpected inputs (e.g., faulty sensor data)  
- Recover from certain failures gracefully  

Robustness and predictability are essential.

---

## 🔍 Comparing RTOS Implementations
Because RTOS are highly specialized, evaluating them requires context. Comparisons should be based on:

- The specific use case intended by the developer  
- Performance across different application scenarios  
- Real-time responsiveness  
- Fault tolerance and recovery mechanisms  

There is no universally “best” RTOS—only the one best suited for a particular application.

# ⚙️ Real-Time Operating Systems (RTOS)

A Real-Time Operating System (RTOS) is a minimal, lightweight operating system designed to run on small embedded devices with very limited memory and computational resources. Because of these constraints, an RTOS must remain compact, efficient, and predictable. Typical applications include processing sensor data, managing media streams, or controlling safety‑critical components such as airbag systems in vehicles.

---

## 🧩 Tasks and Priority-Based Scheduling
Programs or functions executed by an RTOS are called **tasks**.  
Most RTOS implementations use **priority-based scheduling**, meaning:

- Each task is assigned a priority.
- The scheduler always selects the **highest‑priority ready task** for execution.
- Developers must assign priorities carefully, as system performance and responsiveness depend heavily on correct priority design.

---

## ⚡ Interrupt Service Routines (ISR)
Many RTOS support **Interrupt Service Routines**, which allow the system to interrupt the currently running task to handle urgent events.

Key points:

- An ISR immediately executes a designated interrupt handler.
- Before switching, the RTOS **saves the state** of the interrupted task.
- After the ISR completes, the system **restores the saved state** and resumes the suspended task.
- This mechanism is known as **context switching**, as the processor switches from one task’s context (registers, memory state) to another.

---

## 🔒 Semaphores and Critical Sections
Tasks may use **semaphores** to protect critical sections—parts of code that must not be interrupted or accessed concurrently.

Semaphores ensure:

- Only the task holding the semaphore can access the protected resource.
- Other tasks requesting the same resource must wait until the semaphore is released.

However, resource blocking introduces the risk of **priority inversion**, where a low‑priority task holding a semaphore delays a higher‑priority task. This issue is discussed further in Section 2.2.1.

---

## 📊 Scheduling Challenges in RTOS
To achieve high performance and predictable timing, an RTOS must:

- Use an efficient scheduling strategy.
- Handle edge cases introduced by priority-based scheduling.
- Mitigate issues such as priority inversion and ensure real-time guarantees.

The following subsections provide an overview of common scheduling techniques and explore the priority inversion problem along with established solutions.
# 🗂️ Scheduling in Real-Time Operating Systems (RTOS)

Unlike General Purpose Operating Systems (GPOS), which aim for *fairness* by giving tasks similar access to CPU time, Real-Time Operating Systems primarily rely on **preemptive, priority‑based scheduling**. In this model, each task is assigned a priority by the developer, and the RTOS always selects the **highest‑priority ready task** for execution.

A new task is selected for execution when:

- The system is idle and a new task becomes ready  
- A task finishes executing a critical section  
- A task completes its execution  

---

## 🔄 Additional Scheduling Algorithms in RTOS

Although priority-based scheduling is the default, RTOS may use other algorithms in specific scenarios—especially when tasks share the same priority.

### 🔁 Round Robin Scheduling
- Tasks with equal priority receive equal CPU time.  
- Each task is assigned a fixed **time slice**.  
- Frequent context switches occur, which can reduce performance when many tasks are active.  
- This approach resembles scheduling in GPOS.

### 📥 First-In, First-Out (FIFO)
- Tasks are placed in a queue.  
- The task that enters the queue first is executed first.  
- Execution continues until the task finishes or a higher‑priority task becomes ready.

### ⏳ Earliest Deadline First (EDF)
- The task with the **closest deadline** is scheduled next.  
- Tasks may interrupt each other when a newly released task has an earlier deadline.  
- EDF is dynamic and well-suited for systems with strict timing constraints.

---

## ⚡ Preemption and Resource Conflicts

Most RTOS scheduling algorithms are **preemptive**, meaning:

- A higher‑priority task can interrupt and take over CPU resources from a lower‑priority task.  
- However, if a lower‑priority task holds a **semaphore**, the higher‑priority task cannot proceed until the semaphore is released.  

This situation can cause **priority inversion**, where a high‑priority task is indirectly blocked by a low‑priority one. The next section explores how priority inversion occurs and the strategies used to mitigate it.
```
Time:      0   1   2   3   4   5   6   7   8   9  10
           |---|---|---|---|---|---|---|---|---|---|
J1         █       ░
J2             █ █     ░
J3                 █     █
J4                     ░         █ █

Legend:
█  = Task execution block
░  = Second execution block of same task
↑  = Release time (not shown in ASCII)
↓  = Deadline (not shown in ASCII)

```
### Explanation of EDF Scheduling
- Earliest Deadline First (EDF) is a dynamic scheduling algorithm used in real-time operating systems (RTOS). It selects the task with the closest deadline for execution, regardless of its priority level.
- Key Concepts:
    - Release Time: When a task becomes ready to execute.
    - Deadline: The latest time by which the task must finish.
    - Preemption: If a new task is released with an earlier deadline than the currently running task, the RTOS interrupts the current task and switches to the new one.
- In the diagram:
    - J1 starts at time 0, pauses, and resumes at time 5.
    - J2 runs from time 1 to 3, then again at time 7.
    - J3 runs at time 3 and resumes at time 6.
    - J4 starts at time 4 and finishes at time 10.
- Tasks are interrupted and resumed based on their deadlines, not fixed priorities. This allows EDF to adapt to changing task demands and maintain real-time guarantees.


# 🚦 Priority Inversion in RTOS

## ❗ What is Priority Inversion?
**Priority inversion** occurs when a **low-priority task blocks a high-priority task**, preventing it from executing. This typically happens when the high-priority task tries to acquire a **semaphore** that is currently held by a lower-priority task.

---

## 🧠 Example 1: Simple Priority Inversion
- **Task 1** (high priority) is delayed temporarily.
- **Task 2** (lower priority) begins execution and acquires **semaphore S1**.
- When Task 1 resumes, it attempts to acquire S1.
- Since S1 is held by Task 2, **Task 1 is blocked** until Task 2 releases the semaphore.

This is a straightforward case and can be resolved by allowing Task 2 to complete and release the resource.

---

## 🧩 Example 2: Complex Priority Inversion
- **Task 1** (highest priority) and **Task 2** (medium priority) are delayed.
- **Task 3** (lowest priority) starts and acquires **semaphore S1**.
- When Task 1 resumes, it tries to acquire S1 but is blocked by Task 3.
- However, **Task 3 cannot proceed** because Task 2 (medium priority) is now ready and preempts Task 3.

This creates an **indirect block**:
- Task 2 prevents Task 3 from releasing S1.
- Task 1 remains blocked indefinitely, even though it has the highest priority.

This scenario is more dangerous, especially in **real-time or safety-critical systems**, because the duration of the block is unpredictable.

---

## 🛠️ Solutions to Priority Inversion

### 1. 🔒 Disabling Preemption
- Temporarily disables task switching during critical sections.
- Ensures that once a task enters a critical section, it cannot be interrupted.
- Simple but **not scalable**—can reduce system responsiveness.

### 2. 🧬 Priority Inheritance
- Temporarily **raises the priority** of the task holding the semaphore to match the priority of the blocked task.
- Allows the lower-priority task to complete its critical section faster and release the resource.
- Once done, the task’s priority returns to its original level.
- This is a **widely used and effective** strategy in RTOS.

---

## 📌 Summary
Priority inversion can severely impact real-time performance and safety. Understanding how it occurs and applying strategies like **priority inheritance** is essential for designing robust embedded systems.
# 🔄 Priority Inversion Example 1 (ASCII Timeline)

## 🧵 Timeline Overview
```

Time:      0   1   2   3   4   5   6   7   8   9  10
           |---|---|---|---|---|---|---|---|---|---|

Task 1     [Delay] ──> [Normal] ──> [Blocked on S1] ──> [Critical] ──> [Normal] ──> [Delay]

Task 2           [Normal] ──> [Takes S1] ──> [Critical] ──> [Releases S1] ──> [Normal]

Legend:
[Delay]        = Task is delayed
[Normal]       = Non-critical execution
[Blocked on S1]= Task is waiting for semaphore S1
[Critical]     = Critical section (protected by S1)
[Takes S1]     = Semaphore S1 acquired
[Releases S1]  = Semaphore S1 released
```

# 🔄 Priority Inversion Example 2 (ASCII Timeline)

## 🧵 Timeline Overview
```
Time:      0   1   2   3   4   5   6   7   8   9  10
           |---|---|---|---|---|---|---|---|---|---|

Task 1     [Delay] ──> [Blocked on S1] ──> [Can't Execute] ──> [Critical]

Task 2     [Normal] ──> [Delay] ──> [Finish]

Task 3     [Normal] ──> [Takes S1] ──> [Critical] ──> [Releases S1]

Legend:
[Delay]          = Task is delayed
[Normal]         = Non-critical execution
[Blocked on S1]  = Task is waiting for semaphore S1
[Can't Execute]  = Task is blocked by medium-priority task
[Critical]       = Critical section (protected by S1)
[Takes S1]       = Semaphore S1 acquired
[Releases S1]    = Semaphore S1 released
```
###  Explanation

- This example demonstrates a complex priority inversion scenario involving three tasks:
    - Task 1 has the highest priority.
    - Task 2 has medium priority.
    - Task 3 has the lowest priority.

- Sequence of Events:
    - Task 1 is delayed initially.
    - Task 3 begins execution and acquires semaphore S1.
    - Task 2 resumes and preempts Task 3 due to higher priority.
    - Task 1 resumes and tries to acquire S1 but is blocked because Task 3 still holds it.
    - Task 3 cannot finish its critical section because Task 2 keeps preempting it.
    - Task 1 remains blocked for an unpredictable duration.

#### 🔥 Problem:

- Even though Task 1 has the highest priority, it is indirectly blocked by Task 2 because Task 3 (which holds the resource) cannot run. This leads to unbounded blocking time, which is dangerous in real-time systems.
##### 🛠️ Solution Preview
- To prevent this kind of inversion, RTOS designers use strategies like:
    - Priority Inheritance: Temporarily boost Task 3’s priority to match Task 1’s.
    - Disabling Preemption: Prevent interruptions during critical sections.
- These techniques ensure that high-priority tasks are not indefinitely blocked by lower-priority ones.



