# CPU Pipelining

Pipelining is a hardware technique used in processors to increase instruction throughput.  
Instead of executing one instruction from start to finish before beginning the next, the processor divides instruction execution into several stages. Each stage performs a specific part of the work, and different instructions can occupy different stages at the same time. This allows multiple instructions to be processed in parallel, each at a different step of execution.

In simple terms: **a pipeline lets the CPU work on several instructions at once, with each stage handling one step of the process.**

---

## The Classic 5‑Stage RISC Pipeline

A typical RISC processor uses a well‑structured five‑stage pipeline:

1. **Instruction Fetch (IF)**  
   The next instruction is read from memory.

2. **Instruction Decode & Register Read (ID)**  
   The instruction is decoded, and required registers are loaded.

3. **Execute (EX)**  
   The ALU performs arithmetic, logic, or address calculations.

4. **Memory Access (MEM)**  
   Load/store instructions access data memory.

5. **Write Back (WB)**  
   The result is written back to the destination register.

These stages form a hardware assembly line, where each stage is implemented using digital logic such as multiplexers, adders, comparators, flip‑flops, and control state machines.

---

## How the Pipeline Operates

At every clock cycle, each instruction moves forward to the next stage.  
This keeps all stages busy and increases overall throughput.

Example:
- Cycle 1: Instruction A in IF  
- Cycle 2: Instruction A in ID, Instruction B in IF  
- Cycle 3: Instruction A in EX, Instruction B in ID, Instruction C in IF  

Ideally, one instruction completes every cycle once the pipeline is full.

---

## Pipeline Principle

Each stage is separated by **pipeline registers**.  
When the clock ticks:

- Each pipeline register captures the output of the previous stage.
- The next stage reads stable data from its input register.
- All stages operate in parallel on different instructions.

This is entirely **hardware‑driven** and does not involve software.

---

## Practical Considerations: Stalls and Hazards

In real processors, the pipeline cannot always advance smoothly.  
Some instructions must wait because:

- a required value is not ready yet (data hazard)  
- the next instruction address is uncertain (control hazard)  
- a resource is busy (structural hazard)  

Hardware mechanisms such as forwarding, hazard detection, and branch prediction help reduce these delays.

---

## Simplified Pipelined Datapath

A pipelined RISC datapath typically includes:

- instruction memory  
- register file  
- ALU  
- data memory  
- pipeline registers between each stage  
- control logic for hazard detection and forwarding  

Each block is implemented using digital logic gates and flip‑flops.

---

## Summary

Pipelining is a hardware technique that:

- splits instruction execution into multiple stages  
- allows several instructions to be processed simultaneously  
- increases throughput without increasing clock frequency  
- relies entirely on digital logic, not software  

It is a foundational concept in modern embedded CPU design and is used in nearly all RISC architectures.

