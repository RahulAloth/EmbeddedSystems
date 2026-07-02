# Controller Area Network (CAN) Protocol — Technical Overview

Controller Area Network (CAN) is a robust, real‑time communication protocol used in automotive, industrial, and embedded systems.  
It enables multiple ECUs (Electronic Control Units) to communicate over a shared bus without a central host.

---

# 1. CAN Basics

- **Type:** Multi‑master, message‑based protocol  
- **Medium:** Differential signaling (CAN_H, CAN_L)  
- **Speed:** Up to 1 Mbit/s (Classical CAN), up to 5–8 Mbit/s (CAN FD data phase)  
- **Topology:** Bus with termination at both ends (120 Ω)

CAN is designed for **high reliability**, **fault tolerance**, and **real‑time communication**.

---

# 2. CAN Frame Structure

A CAN frame contains:

1. **Start of Frame (SOF)**  
2. **Arbitration Field**  
   - Identifier (11‑bit or 29‑bit)  
   - RTR bit  
3. **Control Field**  
   - DLC (Data Length Code)  
4. **Data Field**  
   - 0–8 bytes (Classical CAN)  
   - 0–64 bytes (CAN FD)  
5. **CRC Field**  
6. **ACK Field**  
7. **End of Frame (EOF)**

---

# 3. CAN Identifier Types

## Standard CAN (11‑bit ID)
- Range: **0x000 – 0x7FF**
- Used in most automotive ECUs

## Extended CAN (29‑bit ID)
- Range: **0x00000000 – 0x1FFFFFFF**
- Used for diagnostics, J1939, and some OEM networks

---

# 4. Arbitration (CSMA/CR)

CAN uses **Carrier Sense Multiple Access with Collision Resolution**.

- All nodes listen to the bus  
- If the bus is idle, any node may transmit  
- If two nodes transmit simultaneously:
  - The node with **lower ID value** (higher priority) wins  
  - The other node stops and retries later

This makes CAN **event‑triggered** and **non‑deterministic**.

---

# 5. Bit‑Stuffing

To maintain synchronization:

- After **five consecutive identical bits**, CAN inserts a **stuff bit**  
- This increases frame length unpredictably  
- Worst‑case overhead: **20–30%**

---

# 6. Error Handling

CAN has five error types:

- **Bit Error**  
- **Stuff Error**  
- **CRC Error**  
- **Form Error**  
- **ACK Error**

Nodes maintain error counters:

- **Transmit Error Counter (TEC)**  
- **Receive Error Counter (REC)**

Error states:

- **Error Active**  
- **Error Passive**  
- **Bus Off** (node disconnected)

---

# 7. CAN Physical Layer

- Differential signaling  
- Dominant bit = 0 (CAN_H high, CAN_L low)  
- Recessive bit = 1 (both lines idle)  
- Termination: **120 Ω** at both ends  
- Typical voltage levels:  
  - CAN_H ≈ 3.5 V  
  - CAN_L ≈ 1.5 V

---

# 8. CAN FD (Flexible Data Rate)

Enhancements over Classical CAN:

- **Data phase bit‑rate:** up to 5–8 Mbit/s  
- **Payload:** up to 64 bytes  
- **Improved CRC**  
- **Faster communication for diagnostics and ADAS**

Limitations:

- Arbitration still at **1 Mbit/s**  
- Latency still **non‑deterministic**

---

# 9. CAN Message Types

- **Data Frame** — carries data  
- **Remote Frame** — requests data  
- **Error Frame** — signals error  
- **Overload Frame** — adds delay between frames

---

# 10. CAN Advantages

- High reliability  
- Real‑time capability  
- Fault tolerance  
- Low cost  
- Simple wiring  
- Widely supported in automotive ECUs

---

# 11. CAN Limitations

- Limited bandwidth (1 Mbit/s classical, ~5 Mbit/s FD)  
- Non‑deterministic latency  
- Arbitration delays  
- Not suitable for high‑bandwidth sensors (camera, radar)

---

# 12. Typical Automotive CAN IDs (Examples)

| Function | CAN ID | Type |
|----------|--------|------|
| Engine RPM | 0x0C0 | Standard |
| Vehicle Speed | 0x0AA | Standard |
| Steering Angle | 0x0B0 | Standard |
| ABS Status | 0x1A0 | Standard |
| Diagnostics (UDS) | 0x7E0 (Req), 0x7E8 (Resp) | Standard |
| J1939 Engine Data | 0x0CF00400 | Extended |

---

# 13. CAN in Automotive Networks

Used in:

- Powertrain ECUs  
- Body control modules  
- Instrument clusters  
- HVAC  
- Airbags  
- Diagnostics (UDS)  
- Gateway ECUs

Not used for:

- ADAS sensors  
- High‑bandwidth camera streams  
- IVI video/audio transport

---

# Summary

CAN is a **robust, event‑triggered, multi‑master bus** designed for real‑time automotive communication.  
It provides reliability and simplicity but has bandwidth and latency limitations, which led to CAN FD and Automotive Ethernet for modern systems.
# CAN Data Frame Structure (ASCII Diagram)
```markdown

+-------------------------------------------------------------------------------------------+
|                                      CAN DATA FRAME                                       |
+-------------------------------------------------------------------------------------------+

  Start of Frame (SOF)
  ---------------------
  [0]

  Arbitration Field
  -----------------
  Standard ID (11-bit):
  [ID10 ID9 ID8 ID7 ID6 ID5 ID4 ID3 ID2 ID1 ID0]

  Extended ID (29-bit):
  [ID28 ... ID18] SRR IDE [ID17 ... ID0]

  RTR (Remote Transmission Request)
  ---------------------------------
  [RTR]

  Control Field
  -------------
  IDE   — Identifier Extension bit  
  r0    — Reserved bit  
  DLC   — Data Length Code (0–8 for Classical CAN, 0–64 for CAN FD)

  [IDE r0 DLC3 DLC2 DLC1 DLC0]

  Data Field
  ----------
  Classical CAN: 0–8 bytes  
  CAN FD: 0–64 bytes

  [D0 D1 D2 D3 D4 D5 D6 D7]   <-- Example 8-byte payload

  CRC Field
  ---------
  [CRC15 ... CRC0] CRC Delimiter

  ACK Field
  ---------
  ACK Slot (dominant = received OK)  
  ACK Delimiter

  [ACK ACK_DELIM]

  End of Frame (EOF)
  ------------------
  [1 1 1 1 1 1 1]   <-- 7 recessive bits

+-------------------------------------------------------------------------------------------+
|                               FULL CAN FRAME (LINEAR VIEW)                                |
+-------------------------------------------------------------------------------------------+

SOF | Arbitration | Control | Data | CRC | ACK | EOF

[0] | [ID + RTR] | [IDE + DLC] | [D0..Dn] | [CRC] | [ACK] | [1111111]


```
