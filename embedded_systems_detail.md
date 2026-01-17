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
