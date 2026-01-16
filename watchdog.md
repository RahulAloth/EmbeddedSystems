# Watchdog Timer (WDT)
### Study Notes for Embedded Systems

A **Watchdog Timer (WDT)** is a hardware safety mechanism that automatically resets a microcontroller if the software becomes stuck, unresponsive, or behaves abnormally.  
It ensures system reliability, especially in safety‑critical or unattended applications.

---

## 1. What Is a Watchdog?

A watchdog is a hardware timer that continuously counts down.  
Your software must periodically **refresh** (also called *kick*, *feed*, or *service*) the watchdog before it expires.

If the watchdog is **not** refreshed in time:

- The timer expires  
- A **system reset** is triggered  
- The MCU restarts cleanly  

This prevents the system from staying frozen indefinitely.

---

## 2. Why Watchdogs Are Needed

Embedded systems can fail due to:

- Infinite loops  
- Deadlocks  
- Stack overflows  
- Memory corruption  
- Peripheral lockups  
- EMI / electrical noise  
- Software bugs  

A watchdog ensures **automatic recovery** without human intervention.

---

## 3. How a Watchdog Works

1. Watchdog timer starts counting down  
2. Software must periodically refresh it  
3. If refresh is missed → watchdog expires  
4. MCU resets and restarts the program  

This mechanism guarantees that the system never stays stuck.

---

## 4. Types of Watchdogs

### 4.1 Standard Watchdog
- Must be refreshed before timeout  
- Simple and widely used  

### 4.2 Windowed Watchdog
- Must be refreshed **within a specific time window**  
- Refreshing too early or too late triggers a reset  
- Detects runaway loops that refresh too fast  

### 4.3 Independent Watchdog
- Runs from its own clock source  
- Works even if CPU clock fails  
- Used in high‑reliability systems  

---

## 5. Watchdog Timeout

The **timeout period** is the maximum allowed time between refreshes.

Examples:
- 10 ms  
- 100 ms  
- 1 second  

Timeout must be long enough to avoid false resets but short enough to catch real failures.

---

## 6. Typical Firmware Flow

```c
int main(void)
{
    WDT_Init(100);   // 100 ms timeout

    while (1)
    {
        do_tasks();
        WDT_Service();   // Refresh watchdog
    }
}
```
If do_tasks() hangs → watchdog resets the MCU.

## 7. What Happens After a Watchdog Reset?

Most MCUs:
    - Set a WDT reset flag
    - Restart the program from the beginning
    - Allow firmware to detect the cause of reset
## Best Practices
    - Refresh the watchdog only when system is healthy
    - Never refresh inside interrupts (hides bugs)
    - Use windowed watchdog for safety‑critical systems
    - Log watchdog resets for debugging
    - Choose timeout based on worst‑case execution time
    - Use independent watchdog if available
