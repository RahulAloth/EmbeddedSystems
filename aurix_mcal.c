/*
# Example: GPIO Port Configuration on AURIX™ TC49xx

This example configures **Port P10.2** as a **push‑pull output** and toggles it in a loop.  
The code demonstrates how to manipulate:

- IOCR (Input/Output Control Register)
- OUT (Output Register)
- OMR (Output Modification Register)

---

## 1. C Example: Configure P10.2 as Output and Toggle

```c
*/

#include "IfxPort.h"
#include "IfxScuWdt.h"

void init_GPIO(void)
{
    /* Disable watchdogs (typical in AURIX init code) */
    IfxScuWdt_disableCpuWatchdog(IfxScuWdt_getCpuWatchdogPassword());
    IfxScuWdt_disableSafetyWatchdog(IfxScuWdt_getSafetyWatchdogPassword());

    /* Configure P10.2 as push‑pull output */
    IfxPort_setPinMode(&MODULE_P10, 2, IfxPort_Mode_outputPushPullGeneral);
}

void toggle_GPIO(void)
{
    while (1)
    {
        /* Set pin high */
        IfxPort_setPinHigh(&MODULE_P10, 2);

        for (volatile int i = 0; i < 100000; i++);

        /* Set pin low */
        IfxPort_setPinLow(&MODULE_P10, 2);

        for (volatile int i = 0; i < 100000; i++);
    }
}

int main(void)
{
    init_GPIO();
    toggle_GPIO();
    return 0;
}

// 2. Register‑Level C Example (No iLLD)
#define P10_IOCR0   (*(volatile unsigned int*)0xF003B410)
#define P10_OUT     (*(volatile unsigned int*)0xF003B404)
#define P10_OMR     (*(volatile unsigned int*)0xF003B408)

/* IOCR field for P10.2 is bits [23:20] */
#define PC2_OUTPUT_PUSH_PULL   (0x10U << 20)

void init_GPIO_registers(void)
{
    /* Clear PC2 bits */
    P10_IOCR0 &= ~(0x1FU << 20);

    /* Set PC2 = 0b10000 (push‑pull output) */
    P10_IOCR0 |= PC2_OUTPUT_PUSH_PULL;
}

void toggle_GPIO_registers(void)
{
    while (1)
    {
        /* Set P10.2 high: OMR bit 2 = 1 */
        P10_OMR = (1U << 2);

        for (volatile int i = 0; i < 100000; i++);

        /* Set P10.2 low: OMR bit (2 + 16) = 1 */
        P10_OMR = (1U << (2 + 16));

        for (volatile int i = 0; i < 100000; i++);
    }
}

