# AUTOSAR Partitioning — Detailed Technical Explanation

## 🧩 What **[AUTOSAR Partitioning](ca://s?q=Explain_AUTOSAR_partitioning_in_detail)** Actually Means

In AUTOSAR, **partitioning** means splitting software into isolated execution and memory regions so that components cannot interfere with each other. It is a core mechanism for **safety**, **security**, and **multi‑core scheduling**.

AUTOSAR uses partitions to enforce **Freedom From Interference (FFI)** — a mandatory ISO 26262 requirement. Partitioning ensures that one software component cannot corrupt or disturb another, especially when they have different ASIL levels.

---

## 1. **[Memory Partitioning](ca://s?q=Explain_AUTOSAR_memory_partitioning)**

- Each **OS‑Application** runs in its own protected memory region.  
- The **MPU (Memory Protection Unit)** enforces boundaries.  
- Safety‑critical SWCs (ASIL‑B/C/D) are separated from QM (non‑safety) SWCs.  
- Prevents accidental writes, corruption, or tampering.

**Key idea:** QM code cannot access or overwrite ASIL‑D memory.

---

## 2. **[Execution Partitioning](ca://s?q=Explain_AUTOSAR_execution_partitioning)**

- AUTOSAR maps **Runnables → Tasks → OS‑Applications**.  
- Each partition has its own execution context.  
- Faults in one partition do not crash the entire ECU.

This ensures deterministic scheduling and fault containment.

---

## 3. **[Communication Partitioning](ca://s?q=Explain_AUTOSAR_communication_partitioning)**

- Intra‑ECU communication is controlled by the **RTE**.  
- Inter‑ECU communication uses the **COM stack** with **E2E protection**.  
- Prevents unauthorized or unsafe data exchange.

---

## 4. **[Multi‑Core Partitioning](ca://s?q=Explain_AUTOSAR_multi_core_partitioning)**

- Safety functions can run on one core, QM functions on another.  
- Each core has its own OS‑Applications + MPU regions.  
- Prevents cross‑core interference and improves determinism.

---

## 🔧 Why Partitioning Exists

AUTOSAR partitioning solves three major engineering problems:

### A. **[Safety (ISO 26262)](ca://s?q=Explain_ISO_26262_safety_requirements)**

- Prevents QM code from corrupting ASIL‑D data.  
- Ensures deterministic behavior and fault isolation.

### B. **[Security](ca://s?q=Explain_security_in_AUTOSAR_partitioning)**

- Limits access to memory‑mapped peripherals.  
- Reduces attack surface by isolating modules.

### C. **[Modularity & Maintainability](ca://s?q=Explain_modularity_in_AUTOSAR)**

- SWCs become independent modules.  
- Easier updates, debugging, and certification.

---

## 🏎️ Simple Example (Automotive ECU)

| Component               | ASIL | Partition        | Why                                               |
|------------------------|------|------------------|---------------------------------------------------|
| **[Brake Control SWC](ca://s?q=Explain_brake_control_SWC)** | D    | Safety partition | Must be isolated from infotainment                |
| **[Infotainment CAN Gateway](ca://s?q=Explain_CAN_gateway_QM)** | QM   | QM partition     | Non‑critical, cannot corrupt safety data          |
| **[Watchdog Manager](ca://s?q=Explain_watchdog_manager_in_AUTOSAR)** | C    | Safety partition | Supervises safety tasks                           |

The **MPU** ensures QM code cannot write into safety memory regions.

---

## 🧠 Visual Mental Model

Think of AUTOSAR partitioning like **containers inside an ECU**:

- Each container has its own memory.  
- Each container has its own tasks.  
- Containers communicate only through controlled interfaces.  
- A crash in one container does not crash others.

# Classic AUTOSAR vs Adaptive AUTOSAR Partitioning  

---

## 1. Overview  
Classic AUTOSAR uses **static, compile‑time partitioning** enforced by an **MPU**, while Adaptive AUTOSAR uses **dynamic, process‑based partitioning** enforced by an **MMU** and POSIX/Linux mechanisms.

---

## 2. Comparison Table

| Topic | Classic AUTOSAR Partitioning | Adaptive AUTOSAR Partitioning |
|------|------------------------------|-------------------------------|
| **Execution Model** | Tasks + OS‑Applications | POSIX Processes + Threads |
| **Memory Isolation** | Static MPU regions | MMU, virtual memory, dynamic allocation |
| **Scheduling** | Deterministic, static | Dynamic, Linux/POSIX |
| **Communication** | RTE, COM stack, signals | ara::com, SOME/IP, DDS |
| **Fault Containment** | OS‑Application isolation | Process isolation + supervisor restart |
| **Safety Certification** | ASIL‑D capable | ASIL‑B/C capable |
| **Update Model** | Static firmware | Dynamic OTA updates |
| **Use Cases** | Body, chassis, powertrain ECUs | ADAS, IVI, HPC, domain controllers |

---

## 3. Memory Partitioning

### Classic AUTOSAR
- Uses **MPU** (no virtual memory).  
- Memory regions are **fixed at compile time**.  
- Each OS‑Application has a static memory map.  
- Violations → protection fault → safe state.

### Adaptive AUTOSAR
- Uses **MMU** with full virtual memory.  
- Each Adaptive Application runs in its own **process address space**.  
- Isolation similar to Linux containers.  
- Violations → process crash, not ECU crash.

---

## 4. Execution Partitioning

### Classic
- Runnables → Tasks → OS‑Applications.  
- No dynamic loading.  
- Hard real‑time behavior.  
- Designed for deterministic microcontrollers.

### Adaptive
- Processes + threads.  
- Dynamic loading of applications.  
- Service discovery + dynamic binding.  
- Designed for high‑performance compute.

---

## 5. Communication Partitioning

### Classic
- RTE enforces strict interfaces.  
- COM stack with E2E protection.  
- Static configuration (signals, PDUs).

### Adaptive
- ara::com (SOME/IP, DDS).  
- Dynamic service discovery.  
- Rich IPC: shared memory, sockets, queues.

---

## 6. Fault Isolation

### Classic
- OS‑Application fault → system protection hook.  
- ECU may enter safe state.  
- Strict fault model for ASIL‑D.

### Adaptive
- Process fault → Adaptive Execution Manager restarts it.  
- Other processes unaffected.  
- Microservices‑like isolation.

---

## 7. Multi‑Core Partitioning

### Classic
- Static core assignment.  
- No migration.  
- Each core has its own OS‑Applications.

### Adaptive
- Linux scheduler handles cores.  
- Processes can migrate across cores.  
- NUMA‑aware scheduling possible.
NUMA (Non‑Uniform Memory Access) systems have multiple memory nodes. Accessing local memory is fast; accessing remote memory is slower. NUMA‑aware scheduling tries to:
- Keep threads on the same NUMA node as their allocated memory
- Reduce remote memory traffic
- Improve cache locality
- Increase throughput and lower latency

---

## 8. Security Partitioning

### Classic
- MPU + static memory regions.  
- Limited security primitives.  
- No user/kernel mode separation.

### Adaptive
- Full Linux security stack:  
  - SELinux/AppArmor  
  - Namespaces  
  - cgroups  
  - seccomp  
- Strong isolation and sandboxing.
- Note : SELinux and AppArmor are Linux Security Modules (LSMs) that enforce Mandatory Access Control (MAC) to confine applications and limit what they can access — even if they get compromised.They solve the same problem (restricting processes), but they work very differently.
---

## 9. Typical Use Cases

### Classic AUTOSAR
- Window lifter  
- Door control  
- Powertrain  
- Chassis  
- Airbag ECU  
- Battery management  

### Adaptive AUTOSAR
- ADAS perception  
- Sensor fusion  
- IVI / cockpit domain  
- Autonomous driving stack  
- Central compute / HPC

---

## 10. Summary

- **Classic partitioning** = static, deterministic, safety‑certified isolation for microcontrollers.  
- **Adaptive partitioning** = dynamic, process‑based isolation for high‑performance automotive compute.

They complement each other in modern vehicle architectures.

---

# Classic AUTOSAR OS‑Applications  
A Classic AUTOSAR **OS‑Application** is the core isolation and partitioning unit in the AUTOSAR Classic Platform. It groups tasks, ISRs, and memory into a protected container enforced by the **MPU** to guarantee safety and prevent interference.

---

## 1. What is an OS‑Application?
An **OS‑Application** is a logical container inside the AUTOSAR OS that bundles:
- Tasks  
- ISRs  
- Trusted functions  
- Memory regions  
- Access rights  

It is the smallest unit of **safety isolation** and **fault containment** in Classic AUTOSAR.

---

## 2. Why OS‑Applications Exist
OS‑Applications implement ISO 26262 **Freedom From Interference (FFI)** by ensuring:
- Memory isolation  
- Timing isolation  
- Controlled communication  
- Fault containment  

They prevent QM code from corrupting ASIL‑D code.

---

## 3. Key Properties of OS‑Applications

### 3.1 Memory Isolation (MPU)
Each OS‑Application has:
- Its own memory region  
- Configured MPU boundaries  
- Restricted access to other applications  

If a task tries to access unauthorized memory → **MPU fault → protection hook**.

---

### 3.2 Execution Isolation
Each OS‑Application contains:
- A set of tasks  
- A set of ISRs  
- Trusted functions  

A fault inside one OS‑Application does **not** crash the entire ECU.

---

### 3.3 Access Rights
OS‑Applications define:
- Which tasks can access which memory  
- Which tasks can call which trusted functions  
- Which tasks can access hardware drivers  

This prevents accidental or malicious interference.

---

### 3.4 Fault Containment
If an OS‑Application violates rules:
- OS triggers **ProtectionHook()**  
- ECU may enter safe state  
- Fault is contained within the application  

---

## 4. Structure of an OS‑Application
+-------------------------------+
OS-Application A
Tasks: T1, T2
ISRs: ISR1
Trusted Functions: TF1
Memory Region: MR_A

+-------------------------------+
+-------------------------------+
OS-Application B
Tasks: T3
ISRs: ISR2
Trusted Functions: TF2, TF3
Memory Region: MR_B

+-------------------------------+

Each OS‑Application is isolated by the **MPU**.

---

## 5. Types of OS‑Applications

### 5.1 Non‑Trusted OS‑Application
- Cannot access hardware directly  
- Must use trusted functions  
- Used for QM or lower ASIL software  

### 5.2 Trusted OS‑Application
- Can access hardware drivers  
- Can call privileged operations  
- Used for ASIL‑C/D software  

---

## 6. Mapping SWCs → Runnables → Tasks → OS‑Applications

SWC → Runnable → Task → OS‑Application → Core


This mapping determines:
- Safety level  
- Memory protection  
- Scheduling  
- Communication rules  

---

## 7. Multi‑Core Behavior
On multi‑core ECUs:
- Each core has its own OS‑Applications  
- No migration between cores  
- Each core has its own MPU configuration  

This ensures deterministic behavior.

---

## 8. Example: Brake Control ECU

| Component | ASIL | OS‑Application | Notes |
|----------|------|----------------|-------|
| Brake Pressure Control | D | SafetyApp | Trusted, isolated |
| Wheel Speed Processing | C | SafetyApp | Trusted |
| Diagnostics | QM | QMApp | Non‑trusted |
| CAN Gateway | QM | QMApp | Non‑trusted |

SafetyApp and QMApp cannot interfere with each other.

---

## 9. Summary
- OS‑Applications are the **partitioning mechanism** in Classic AUTOSAR.  
- They enforce **memory**, **execution**, and **fault isolation**.  
- They are essential for **ISO 26262 compliance**.  
- They provide deterministic, safety‑certified behavior on MCUs.

---

