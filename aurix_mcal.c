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
