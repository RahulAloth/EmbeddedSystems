# Embedded Functional Safety

Functional safety ensures that embedded systems continue to operate safely even when hardware or software faults occur. Standards such as ISO 26262, IEC 61508, DO‑178C, and DO‑254 define the requirements for detecting, containing, and mitigating faults in safety‑critical applications such as automotive ECUs, industrial controllers, robotics, medical devices, and aerospace systems.

This chapter covers the core hardware safety mechanisms used across the embedded industry.

---

## 1. Lockstep Cores

Lockstep is a CPU‑level redundancy technique used to detect computation faults in real time.

### Concept
Two identical processor cores execute the same instruction stream simultaneously.  
A hardware comparator checks their outputs every cycle (or with a small delay).  
If the outputs differ → fault detected → system transitions to a safe state.

### Why It Is Used
- Detects transient faults (radiation, EMI)  
- Detects permanent faults (stuck‑at, aging)  
- Detects comparator faults  
- Provides high diagnostic coverage for safety standards  

### Variants
- Cycle‑by‑cycle lockstep — strict, immediate comparison  
- Delayed lockstep — checker core runs a few cycles behind to avoid common‑cause faults  
- TMR (Triple Modular Redundancy) — three cores with majority voting  

---

## 2. ECC – Error‑Correcting Code

ECC protects memory and buses from silent data corruption.

### Why Memory Errors Occur
- cosmic radiation  
- electromagnetic interference  
- voltage fluctuations  
- transistor aging  

A single bit can flip from 1 → 0 or 0 → 1 (soft error).

### How ECC Works
ECC stores extra parity bits alongside the data.  
These parity bits are computed using XOR‑based rules so the hardware can detect and correct errors.

Example:  
Data: 10110101  
ECC bits: P1 P2 P3 P4 P5 P6 P7  
Stored as:  
10110101 | P1 P2 P3 P4 P5 P6 P7

### Capabilities (SECDED)
- Single‑bit error correction  
- Double‑bit error detection  
- Prevents corrupted data from entering safety‑critical logic  

---

## 3. BIST – Built‑In Self‑Test

BIST allows hardware to test itself at startup and during runtime.

### Types
- LBIST — tests digital logic (ALUs, pipelines, control logic)  
- MBIST — tests SRAM arrays  

### Purpose
- Detect latent hardware faults  
- Validate safety mechanisms (lockstep, ECC, comparators)  
- Required for high‑integrity systems (ASIL‑D, SIL‑3/4)  

### Where It Is Used
- Automotive microcontrollers  
- Industrial PLCs  
- Safety‑certified SoCs  

---

## 4. Safety Island

A Safety Island is an independent subsystem inside a larger SoC that remains operational even if the main compute fabric fails.

### Typical Components
- Lockstep safety microcontroller  
- Independent power domain  
- Independent clock domain  
- ECC‑protected SRAM  
- Watchdogs and safety monitors  
- Error aggregation and reporting  
- Safe‑state control outputs  

### Purpose
Modern SoCs include large compute blocks (CPU clusters, GPUs, NPUs, DSPs) that are not safety‑certified.  
A Safety Island provides a dedicated ASIL‑D supervisor that monitors the entire chip and enforces safe‑state transitions.

---

## 5. Watchdog Timers

A watchdog timer resets the system if software becomes stuck or behaves unexpectedly.

### Types
- Windowed watchdog — must be serviced within a specific time window  
- Independent watchdog — runs on a separate clock domain  
- Safety watchdog — part of a safety island or safety MCU  

### Purpose
Prevents silent system hangs.

---

## 6. Clock and Voltage Monitors

### Clock Monitors
Detect failures in the system clock:
- drift  
- stoppage  
- frequency out of range  
- jitter  

### Voltage/Power Monitors
Detect:
- undervoltage  
- overvoltage  
- brown‑out conditions  

### Purpose
Ensures the system does not operate under unstable electrical conditions.

---

## 7. Memory Protection Unit (MPU)

The MPU enforces memory access rules to prevent accidental corruption.

### Capabilities
- region‑based access control  
- privilege separation  
- stack/heap protection  

### Purpose
Prevents runaway code from corrupting safety‑critical data.

---

## 8. Error Signaling and Fault Aggregation

Many safety‑certified MCUs include a centralized error aggregation module.

### Purpose
- Collects all hardware error signals  
- Routes them to the safety controller  
- Ensures deterministic safe‑state handling  

---

## 9. Redundant Peripherals

Critical peripherals often have redundancy to avoid single‑point failures.

### Examples
- dual ADCs  
- redundant PWM generators  
- dual CAN/LIN/FlexRay controllers  
- redundant sensor interfaces  

---

## 10. CRC – Cyclic Redundancy Check

CRC protects data integrity during communication and storage.

### Used For
- flash memory integrity  
- communication frames  
- bootloader verification  
- safety‑critical data structures  

---

## 11. Safe Boot / Secure Boot

Ensures the system boots only trusted, verified software.

### Mechanisms
- signature verification  
- hash checks  
- boot ROM integrity checks  

---

## 12. Dual‑Channel Architecture

Two independent channels compute the same function and cross‑check each other.

### Used In
- braking systems  
- steering systems  
- industrial robots  

---

## 13. End‑to‑End Protection (E2E)

Protects data as it moves between software components.

### Includes
- sequence counters  
- CRC  
- timeout monitoring  

---

## 14. Safe State Machines

A state machine designed to transition to a safe state on any unexpected condition.

---

## 15. Redundant Sensors

Critical systems use multiple sensors for the same measurement.

### Examples
- dual‑redundant steering angle sensors  
- triple‑redundant IMUs  
- redundant temperature sensors  

---

## 16. ASIL Decomposition

Splitting a high‑ASIL function into multiple lower‑ASIL components that together achieve the required safety level.

---

## System‑Level Interaction of Safety Mechanisms

| Mechanism | Role | Why It Matters |
|----------|------|----------------|
| Lockstep Cores | Detect CPU computation faults | Prevents incorrect actuator commands |
| ECC | Protects memory & buses | Prevents silent data corruption |
| BIST | Tests hardware at startup/runtime | Catches latent faults |
| Safety Island | Independent safety supervisor | Maintains safe state even if main SoC fails |
| Watchdogs | Detect software hangs | Ensures system responsiveness |
| Clock/Voltage Monitors | Detect electrical instability | Prevents undefined behavior |
| MPU | Enforces memory access rules | Prevents corruption |
| CRC | Ensures data integrity | Detects communication/storage errors |

---

## Chapter Summary

Embedded functional safety relies on a layered approach:

- Lockstep → CPU redundancy  
- ECC → memory integrity  
- BIST → hardware self‑test  
- Safety Island → isolated supervisor  
- Watchdogs → software fault detection  
- Clock/Voltage Monitors → electrical stability  
- MPU → memory protection  
- CRC → data integrity  
- Redundancy → sensors, peripherals, channels  

Together, these mechanisms form the backbone of safety‑critical embedded architectures.
