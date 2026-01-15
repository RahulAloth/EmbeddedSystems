# UART (Universal Asynchronous Receiver/Transmitter)  
### Interface & Protocol Guide for Embedded Systems

UART is one of the simplest and most widely used serial communication interfaces.  
It is asynchronous (no clock line), full‑duplex, and ideal for debugging, logging, and communicating with modules like GPS, GSM, BLE, Wi‑Fi, and microcontrollers.

---

## 1. Physical Layer

UART uses **two data lines**:

| Line | Direction | Description |
|------|-----------|-------------|
| **TX** | Output | Transmitter pin |
| **RX** | Input | Receiver pin |

Optional lines (depending on hardware):

- RTS / CTS → Hardware flow control  
- DTR / DSR / DCD → Legacy modem control signals  

---

## 2. UART Frame Format

A UART frame contains:
```
Start Bit | Data Bits | Parity Bit (optional) | Stop Bit(s)
```

### Common configurations
- **8N1** → 8 data bits, No parity, 1 stop bit  
- **8E1** → 8 data bits, Even parity, 1 stop bit  
- **7E1**, **7O1**, etc.

### Start Bit
- Always **low**  
- Signals the beginning of a frame

### Stop Bit(s)
- Always **high**  
- Ensures line returns to idle state

---

## 3. Baud Rate

Baud rate = bits per second.

Common values:

- 9600  
- 19200  
- 38400  
- 57600  
- 115200  
- 921600  

Both devices **must match** baud rate and frame format.

---

## 4. Asynchronous Nature

UART has **no clock line**.  
Instead, both sides must agree on:

- Baud rate  
- Frame format  
- Voltage levels  

The receiver samples the incoming bits using its own clock.

---

## 5. Voltage Levels

Two common electrical standards:

### **TTL UART (0–3.3V or 0–5V)**
- Used inside microcontrollers  
- Logic‑level signals  
- Directly compatible with GPIO

### **RS‑232 UART (±12V)**
- Legacy PC serial ports  
- Requires level shifter (MAX232)

Never connect RS‑232 directly to MCU pins.

---

## 6. UART Communication Examples

### Transmitting a byte
```
Idle (High)
Start (Low)
D0 D1 D2 D3 D4 D5 D6 D7
Stop (High)
```

### Full‑duplex
- TX and RX operate independently  
- Can send and receive simultaneously  

---

## 7. Flow Control

### Software Flow Control (XON/XOFF)
- Uses special characters  
- No extra wires  
- Less reliable for binary data

### Hardware Flow Control (RTS/CTS)
- RTS = Ready to Send  
- CTS = Clear to Send  
- Prevents buffer overrun  
- Used in high‑speed links

---

## 8. Common UART Issues

- Mismatched baud rate  
- Wrong frame format (8N1 vs 7E1)  
- Noise on long cables  
- Missing ground reference  
- Buffer overruns  
- Blocking reads causing deadlocks  

---

## 9. UART in Embedded C (Minimal Driver)

### Transmit a byte
```c
void uart_write_byte(uint8_t data)
{
    while (!(UART->STATUS & UART_TX_READY));
    UART->TX = data;
}
```
### Receive a byte
```
uint8_t uart_read_byte(void)
{
    while (!(UART->STATUS & UART_RX_READY));
    return UART->RX;
}

```
### Transmit a string
```
void uart_write_string(const char *s)
{
    while (*s)
        uart_write_byte(*s++);
}
```
## 10. UART Register‑Level Driver (Hardware‑Agnostic)

Initialization
```
void uart_init(uint32_t baud)
{
    UART->BAUD = baud;
    UART->CTRL = UART_ENABLE | UART_TX_EN | UART_RX_EN;
}

```
### Blocking send/receive
```
bool uart_send(uint8_t data)
{
    while (!(UART->STATUS & UART_TX_READY));
    UART->TX = data;
    return true;
}

bool uart_recv(uint8_t *out)
{
    if (!out) return false;
    while (!(UART->STATUS & UART_RX_READY));
    *out = UART->RX;
    return true;
}
```
# 11. When to Use UART
- Use UART when you need:
    - Debugging output
    - Simple MCU‑to‑MCU communication
    - Talking to modules (GPS, GSM, ESP8266, HC‑05, etc.)
    - Low‑pin‑count serial link

- Avoid UART when you need:
    - High throughput (use SPI)
    - Multi‑drop bus (use RS‑485 or CAN)
    - Synchronous timing (use SPI/I²C)
# Summary

UART is simple, robust, and ideal for debugging and low‑speed communication.
Understanding its frame structure, baud rate, and electrical levels is essential for embedded engineering.
