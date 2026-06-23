# Pipeline Hazards in CPU Architecture

Pipelining increases instruction throughput by allowing multiple instructions to be processed 
simultaneously in different stages of the CPU. However, this parallelism introduces situations 
where instructions interfere with each other. These situations are called **pipeline hazards**.

Pipeline hazards reduce performance, cause stalls, and require additional hardware mechanisms 
such as forwarding, hazard detection units, and branch prediction.

This chapter explains all major hazard types and how modern CPUs resolve them.

---

# 1. What Is a Pipeline Hazard?

A pipeline hazard occurs when the next instruction in the pipeline cannot execute in the next 
clock cycle. This forces the pipeline to stall or insert bubbles.

Hazards fall into three major categories:

1. **Data Hazards**  
2. **Control Hazards**  
3. **Structural Hazards**

---

# 2. Data Hazards

Data hazards occur when instructions depend on each other’s data.

There are three types:

- **RAW (Read After Write)**  
- **WAR (Write After Read)**  
- **WAW (Write After Write)**  

## 2.1 RAW – Read After Write (True Dependency)

This is the most common hazard.

Instruction 1: R1 = R2 + R3
Instruction 2: R4 = R1 + R5   ← needs R1 before Instruction 1 writes it


Instruction 2 needs the result of Instruction 1, but the value is not yet written.

### How CPUs fix RAW hazards
- **Forwarding / Bypassing**  
- **Pipeline stalls**  
- **Register renaming (OoO CPUs)**  

RAW hazards cannot be eliminated by compilers alone — they are true dependencies.

---

## 2.2 WAR – Write After Read (Anti‑Dependency)

Occurs when a later instruction writes a register before an earlier instruction reads it.

Instruction 1: R4 = R1 + R2   ← needs R1
Instruction 2: R1 = R3 + R5   ← overwrites R1 too early


### How CPUs fix WAR hazards
- **Register renaming** (OoO CPUs)
- **In‑order execution** (Cortex‑M, Cortex‑R) avoids WAR hazards entirely

---

## 2.3 WAW – Write After Write (Output Dependency)

Occurs when two instructions write to the same register.

Instruction 1: R1 = R2 + R3
Instruction 2: R1 = R4 + R5   ← must not write before Instruction 1


### How CPUs fix WAW hazards
- **Register renaming**  
- **In‑order pipelines** avoid WAW hazards

---

# 3. Structural Hazards

A structural hazard occurs when two instructions need the same hardware resource at the same time.

Examples:
- single memory port for both instruction and data  
- single ALU shared by multiple operations  
- single write port to register file  

### Fixes
- **Harvard architecture** (separate I/D buses)  
- **duplicated hardware units**  
- **stalling** when resources are busy  

Cortex‑M and Cortex‑R use Harvard architecture to avoid many structural hazards.

---

# 4. Control Hazards (Branch Hazards)

Control hazards occur when the pipeline does not know which instruction to fetch next.

Example:

BEQ R1, R2, LABEL


The CPU must wait until the branch condition is evaluated.

### Fixes
- **branch prediction**  
- **speculative execution**  
- **branch delay slots (older RISC)**  
- **flush + restart** (simple pipelines)  

Cortex‑A uses advanced branch predictors; Cortex‑M uses simpler logic.

---

# 5. Pipeline Stalls and Bubbles

A **stall** is when the pipeline stops advancing for one or more cycles.

A **bubble** is an empty pipeline slot inserted to resolve a hazard.

Example RAW hazard stall:

Cycle:   IF   ID   EX   MEM  WB
Instr1:  IF   ID   EX   MEM  WB
Instr2:       IF   ID   STALL EX   MEM  WB


Stalls reduce throughput, so CPUs try to avoid them using forwarding.

---

# 6. Forwarding (Bypassing)

Forwarding allows the CPU to use the result of an instruction **before it is written back**.

Example:

ADD R1, R2, R3   ; result available at end of EX
SUB R4, R1, R5   ; needs R1 in EX stage


Instead of waiting for WB, the CPU forwards the ALU output directly to the next instruction.

### Forwarding Paths
- EX → EX  
- MEM → EX  
- MEM → ID (rare)  

Forwarding dramatically reduces RAW hazards.

---

# 7. Hazard Detection Unit

A hardware block that:
- detects RAW hazards  
- inserts stalls when forwarding is not possible  
- controls pipeline bubbles  
- prevents incorrect execution  

Example: load‑use hazard


LDR R1, [R2]
ADD R3, R1, R4   ← must stall 1 cycle


Load data is only available after MEM stage, so forwarding cannot help.

---

# 8. Branch Prediction and Control Hazard Resolution

Modern CPUs use:
- static prediction (backward = taken, forward = not taken)  
- dynamic prediction (2‑bit saturating counters)  
- global/local history tables  
- branch target buffers (BTB)  

Cortex‑A uses advanced predictors; Cortex‑M uses simple logic.

---

# 9. Summary Table

| Hazard Type | Cause | Example | Fix |
|-------------|-------|---------|-----|
| RAW | True dependency | Read before write | Forwarding, stalls |
| WAR | Anti‑dependency | Write before read | Register renaming |
| WAW | Output dependency | Write before write | Register renaming |
| Structural | Resource conflict | One ALU, one memory port | Duplicate units, stalls |
| Control | Unknown next PC | Branches | Prediction, flush |

---

# Summary

Pipeline hazards are unavoidable in pipelined CPUs, but modern architectures use a combination of 
hardware and compiler techniques to minimize their impact:

- **RAW, WAR, WAW** data hazards  
- **structural hazards** from shared resources  
- **control hazards** from branches  
- **stalls and bubbles** to maintain correctness  
- **forwarding** to reduce RAW stalls  
- **hazard detection units** for automatic stall insertion  
- **branch prediction** to reduce control hazards  

Understanding pipeline hazards is essential for designing efficient CPU pipelines and writing 
performance‑critical embedded software.

