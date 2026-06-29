# Embedded Systems Architecture

Embedded Systems Architecture describes how computing elements are organized inside devices that interact with the physical world. It defines the structure, components, memory, communication, timing, and execution model of systems such as automotive ECUs, IoT devices, industrial controllers, medical instruments, and consumer electronics.

---

## 1. Core Definition

An embedded system is a specialized computing system designed to perform dedicated functions with constraints on power, timing, reliability, and cost. Its architecture determines how hardware and software cooperate to meet these constraints.

Embedded architecture focuses on:

- Processor type and capabilities  
- Memory hierarchy and access patterns  
- Peripherals and I/O interfaces  
- Real‑time behavior and determinism  
- Power management and safety  
- Communication buses and protocols  

---

## 2. Processor Categories

Embedded systems use different processor types depending on performance and integration needs:

- **Microcontrollers (MCUs)** – CPU + memory + peripherals on one chip; optimized for control and real‑time.  
- **Microprocessors (MPUs)** – CPU only; rely on external memory and run complex OS like Linux.  
- **Digital Signal Processors (DSPs)** – optimized for math‑heavy signal processing.  
- **System‑on‑Chips (SoCs)** – highly integrated devices combining CPU, GPU, accelerators, and controllers.  
- **FPGAs** – reconfigurable hardware for custom logic and deterministic timing.  

Each processor type defines the system’s performance, complexity, and software model.

---

## 3. Memory Architecture

Embedded systems use a structured memory hierarchy:

- **On‑chip SRAM** – fast, deterministic, used for real‑time tasks.  
- **Flash** – stores firmware and configuration.  
- **External DDR** – large, high‑speed memory for MPUs and SoCs.  
- **EEPROM** – small, reliable non‑volatile memory for parameters.  

Memory placement affects timing, reliability, and power consumption.

---

## 4. Peripherals and I/O

Embedded systems interact with the physical world through integrated peripherals:

- **GPIO** – digital input/output control.  
- **ADC/DAC** – analog signal conversion.  
- **Timers/PWM** – precise timing and motor control.  
- **Communication interfaces** – UART, SPI, I²C, CAN, LIN, Ethernet.  

The choice of peripherals determines how the system senses, controls, and communicates.

---

## 5. Real‑Time Execution Model

Real‑time behavior is central to embedded architecture:

- **Hard real‑time** – deadlines must never be missed (airbags, braking).  
- **Soft real‑time** – occasional delays tolerated (infotainment).  

Execution models:

- **Bare‑metal** – direct control, minimal overhead.  
- **RTOS** – deterministic scheduling, tasks, semaphores.  
- **Linux/Android** – rich features, non‑deterministic.  

The execution model is chosen based on timing requirements.

---

## 6. Power and Safety Architecture

Embedded systems often operate under strict constraints:

- **Low‑power modes** – sleep, deep sleep, clock gating.  
- **Safety mechanisms** – watchdogs, lockstep cores, ECC memory.  
- **Functional safety standards** – ISO 26262 (automotive), IEC 61508 (industrial).  

Safety and power architecture ensure reliability in critical environments.

---

## 7. Communication Architecture

Devices communicate using specialized buses:

- **CAN** – robust automotive network.  
- **LIN** – low‑cost sensor/actuator bus.  
- **FlexRay** – high‑speed deterministic communication.  
- **Ethernet** – high‑bandwidth networking.  

Communication architecture defines how data flows between components.

---

## 8. System Integration Models

Embedded systems combine hardware and software into cohesive units:

- **Single‑chip systems (MCUs)** – minimal PCB complexity.  
- **Multi‑chip systems (MPUs + DDR + PMIC)** – complex boards with high performance.  
- **Hybrid systems (SoC + FPGA)** – combine flexibility and speed.  

Integration level affects cost, performance, and design effort.

---

## 9. Typical Application Domains

Embedded architecture powers:

- Automotive ECUs and infotainment  
- Industrial automation and robotics  
- IoT sensors and gateways  
- Medical devices  
- Consumer electronics  

Each domain has unique timing, safety, and communication requirements.

---

## 10. Why Architecture Matters

A well‑designed embedded architecture ensures:

- Deterministic real‑time behavior  
- Reliable operation under constraints  
- Efficient power usage  
- Robust communication  
- Safe and predictable system behavior  

It is the foundation for deeper topics such as MCU architecture, MPU subsystems, RTOS design, and automotive safety systems.

---
