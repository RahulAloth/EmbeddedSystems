# Local Interconnect Network (LIN) Protocol — Technical Overview

LIN (Local Interconnect Network) is a low‑cost, low‑speed serial communication protocol used in automotive body electronics.  
It complements CAN by handling simple, non‑critical functions.

---

# 1. LIN Basics

- **Speed:** Up to 20 kbit/s  
- **Topology:** Single‑master, multiple‑slave  
- **Medium:** Single wire (UART‑based)  
- **Deterministic:** Yes (time‑triggered schedule table)  
- **Use cases:** Door modules, seat control, mirrors, sunroof, HVAC flaps, sensors

LIN is designed to be **cheap**, **simple**, and **predictable**.

---

# 2. LIN Frame Structure (ASCII Diagram)
```markdown

+-------------------------------------------------------------------------------------------+
|                                      LIN FRAME                                            |
+-------------------------------------------------------------------------------------------+

Break Field
-----------
[0x00]  <-- At least 13 dominant bits (forces sync)

Sync Field
----------
[0x55]  <-- 01010101 (used for baud rate calibration)

Identifier Field (ID)
---------------------
[ID0 ID1 ID2 ID3 ID4 ID5 P0 P1]
- 6-bit ID
- 2 parity bits (P0, P1)

Data Field
----------
[D0 D1 D2 D3 D4 D5 D6 D7]  <-- 0–8 bytes

Checksum Field
--------------
[CHK]
- Classic checksum: sum of data bytes
- Enhanced checksum: includes ID + data

+-------------------------------------------------------------------------------------------+
|                               FULL LIN FRAME (LINEAR VIEW)                                |
+-------------------------------------------------------------------------------------------+

BREAK | SYNC | ID | DATA | CHECKSUM

[00]  | [55] | [ID + Parity] | [D0..Dn] | [CHK]


```
