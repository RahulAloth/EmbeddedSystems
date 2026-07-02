# Unified Diagnostic Services (UDS) — Service IDs (SIDs)

UDS (ISO 14229) is the diagnostic protocol used in automotive ECUs.  
Each service is identified by a **Service ID (SID)**.

---

## 1. Diagnostic Session Control — **0x10**
Switch ECU diagnostic modes.

- 0x10 01 — Default Session  
- 0x10 02 — Programming Session  
- 0x10 03 — Extended Diagnostic Session  

**Response SID:** 0x50

---

## 2. ECU Reset — **0x11**
Reset the ECU.

- 0x11 01 — Hard Reset  
- 0x11 02 — Key Off/On Reset  
- 0x11 03 — Soft Reset  

**Response SID:** 0x51

---

## 3. Security Access — **0x27**
Unlock protected ECU functions.

- 0x27 01 — Request Seed  
- 0x27 02 — Send Key  
- 0x27 03/04/05… — Higher security levels  

**Response SID:** 0x67

---

## 4. Communication Control — **0x28**
Enable/disable ECU communication.

- 0x28 00 — Enable Rx/Tx  
- 0x28 01 — Disable Rx  
- 0x28 02 — Disable Tx  

**Response SID:** 0x68

---

## 5. Tester Present — **0x3E**
Keep ECU awake during diagnostics.

- 0x3E 00 — Standard sub‑function  

**Response SID:** 0x7E

---

## 6. Read Data By Identifier (DID) — **0x22**
Read ECU data.

Common DID examples:
- 0xF190 — VIN  
- 0xF187 — ECU Serial Number  
- 0xF18C — Software Version  
- 0xF1A0 — Bootloader Version  
- 0xF1D0 — Vehicle Manufacturer Data  

**Response SID:** 0x62

---

## 7. Write Data By Identifier — **0x2E**
Write ECU configuration/coding.

Examples:
- 0x2E F1A0 — Write Bootloader Config  
- 0x2E F1D0 — Write Vehicle Config  

**Response SID:** 0x6E

---

## 8. Read Memory By Address — **0x23**
Read raw memory (RAM/Flash).

Format:
- 0x23 [AddressLength] [Address] [Size]

**Response SID:** 0x63

---

## 9. Write Memory By Address — **0x3D**
Write raw memory (development use).

**Response SID:** 0x7D

---

## 10. Routine Control — **0x31**
Run internal ECU routines.

- 0x31 01 xx — Start Routine  
- 0x31 02 xx — Stop Routine  
- 0x31 03 xx — Request Routine Results  

Common routines:
- 0x31 01 FFFF — Erase Flash  
- 0x31 01 FD00 — Check Programming Preconditions  
- 0x31 01 0203 — Clear DTC Routine  

**Response SID:** 0x71

---

## 11. Request Download — **0x34**
Start flashing.

**Response SID:** 0x74

---

## 12. Transfer Data — **0x36**
Send flash blocks.

**Response SID:** 0x76

---

## 13. Request Transfer Exit — **0x37**
Finish flashing.

**Response SID:** 0x77

---

## 14. Clear DTCs — **0x14**
Clear diagnostic trouble codes.

- 0x14 FF FF FF — Clear all DTCs  
- 0x14 xx xx xx — Clear specific DTC group  

**Response SID:** 0x54

---

## 15. Read DTC Information — **0x19**
Read diagnostic trouble codes.

Examples:
- 0x19 02 — Read DTC by status mask  
- 0x19 0A — Read DTC snapshot  
- 0x19 0B — Read DTC extended data  

**Response SID:** 0x59

---

## 16. Negative Response — **0x7F**
ECU rejects a request.

Common NRC codes:
- 0x10 — General Reject  
- 0x11 — Service Not Supported  
- 0x12 — Sub‑function Not Supported  
- 0x13 — Incorrect Message Length  
- 0x22 — Conditions Not Correct  
- 0x33 — Security Access Denied  
- 0x78 — Response Pending  

---

# Summary Table

| Service | SID | Purpose |
|--------|-----|---------|
| Diagnostic Session Control | 0x10 | Switch ECU mode |
| ECU Reset | 0x11 | Reset ECU |
| Security Access | 0x27 | Unlock protected functions |
| Read Data By ID | 0x22 | Read VIN, SW version, etc. |
| Write Data By ID | 0x2E | Coding/config |
| Routine Control | 0x31 | Run internal routines |
| Request Download | 0x34 | Start flashing |
| Transfer Data | 0x36 | Flash blocks |
| Request Transfer Exit | 0x37 | Finish flashing |
| Clear DTCs | 0x14 | Clear faults |
| Read DTC Info | 0x19 | Read faults |
| Tester Present | 0x3E | Keep session alive |
| Negative Response | 0x7F | Error handling |

