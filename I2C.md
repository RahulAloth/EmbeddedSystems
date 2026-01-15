# I²C (Inter‑Integrated Circuit)  
### Interface & Protocol Guide for Embedded Systems

I²C is one of the most widely used serial communication protocols in embedded systems.  
It connects microcontrollers and peripherals using only **two wires**, making it ideal for sensors, EEPROMs, ADCs, displays, and power-management ICs.

---

## 1. Physical Layer

I²C uses two open‑drain lines:

| Line | Meaning |
|------|---------|
| **SCL** | Serial Clock (driven by master) |
| **SDA** | Serial Data (bidirectional) |

Both lines require **pull‑up resistors** because devices only pull the line low.  
Pull‑up sizing affects rise time and bus speed.

---

## 2. Roles on the Bus

- **Master**  
  - Generates clock  
  - Initiates communication  
  - Sends START/STOP conditions  

- **Slave**  
  - Responds to master  
  - Has a unique 7‑bit or 10‑bit address  

Multiple masters and multiple slaves are allowed (multi‑master bus).

---

## 3. I²C Speed Modes

| Mode | Max Speed |
|------|-----------|
| Standard | 100 kHz |
| Fast | 400 kHz |
| Fast Mode Plus | 1 MHz |
| High‑Speed | 3.4 MHz |
| Ultra‑Fast | 5 MHz (unidirectional) |

Higher speeds require tighter timing and lower bus capacitance.

---

## 4. Bus Conditions

### START Condition
SDA goes **low** while SCL is **high**.

### STOP Condition
SDA goes **high** while SCL is **high**.

These transitions define packet boundaries on the bus.

---

## 5. Address + R/W Bit

A typical 7‑bit address frame:

```
| 7-bit Address | R/W | ACK |
```

- **R/W = 0** → Write  
- **R/W = 1** → Read  
- Slave must pull SDA low for **ACK**.

Reserved addresses and special-purpose addresses are defined in the I²C spec.

---

## 6. Data Transfer

Each byte is transferred MSB first:

```
| D7 | D6 | D5 | D4 | D3 | D2 | D1 | D0 | ACK |
```

- Master controls SCL  
- Data changes only when SCL is low  
- Receiver must ACK each byte

---

## 7. Clock Stretching

Slaves may **hold SCL low** to delay the master if they need more time.  
This is called **clock stretching** and is part of the official spec.

---

## 8. Arbitration (Multi‑Master)

If two masters start transmitting simultaneously:

- Both monitor SDA  
- The one that sees a mismatch (tries to send 1 but sees 0) **loses arbitration**  
- The winning master continues without corruption  

This ensures collision‑free communication.

---

## 9. Pull‑Up Resistor Calculation

Pull‑ups must satisfy:

- Rise‑time requirements  
- Bus capacitance limits  
- Current limits of devices  

TI provides formulas and examples for calculating optimal values.

---

## 10. Typical I²C Transaction Examples

### Write Transaction
- START
- ADDR + W
- ACK
- REGISTER
- ACK
- DATA
- ACK
- STOP

### Read Transaction (with repeated START)
- START
- ADDR + W
- ACK
- REGISTER
- ACK
- REPEATED START
- ADDR + R
- ACK
- DATA
- NACK
- STOP

Repeated START avoids releasing the bus between operations.

---

## 11. Common I²C Issues

- Missing pull‑ups  
- Incorrect address (7‑bit vs 8‑bit confusion)  
- Bus stuck low (device holding SDA/SCL)  
- Noise causing false START/STOP  
- Clock stretching not supported by master  

---

## 12. I²C in Embedded C (Example)

### Write a byte to a register
```c
void i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t data)
{
    i2c_start();
    i2c_send(addr << 1 | 0);  // write
    i2c_send(reg);
    i2c_send(data);
    i2c_stop();
}
```
- Read a byte from a register
```
uint8_t i2c_read_reg(uint8_t addr, uint8_t reg)
{
    uint8_t val;

    i2c_start();
    i2c_send(addr << 1 | 0);  // write
    i2c_send(reg);

    i2c_start();              // repeated start
    i2c_send(addr << 1 | 1);  // read
    val = i2c_recv_nack();
    i2c_stop();

    return val;
}
```
# 13. When to Use I²C
- Use I²C when you need:
    - Low pin count (2 wires)
    - Multiple devices on one bus
    - Moderate speed
    - Simple wiring
- Avoid I²C when you need:
    - High throughput (use SPI)
    - Long cable lengths
    - Very low latency
