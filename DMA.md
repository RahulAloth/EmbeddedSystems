# DMA (Direct Memory Access)
### Study Notes for Embedded Systems

DMA stands for **Direct Memory Access**, a hardware feature that allows peripherals or memory controllers to transfer data **without CPU involvement**.  
It dramatically improves performance, reduces CPU load, and enables real‑time data movement in embedded systems.

---

## 1. What DMA Does

Normally, the CPU must copy data:

- Peripheral → CPU → Memory
- Memory → CPU → Peripheral


With DMA:

- Peripheral → DMA → Memory
- Memory → DMA → Peripheral


The CPU is **bypassed**, freeing it to do other tasks.

---

## 2. Why DMA Is Important

DMA is used when:

- Data transfers are large  
- Transfers must be fast  
- CPU must remain free for other tasks  
- Real‑time performance is required  

Examples:
- ADC sampling → memory buffer  
- UART/SPI/I2C → memory  
- Memory‑to‑memory copies  
- Audio streaming  
- Camera/sensor data  
- Motor control feedback loops  

---

## 3. How DMA Works (High‑Level)

1. CPU configures DMA:
   - Source address  
   - Destination address  
   - Transfer size  
   - Transfer type (byte/halfword/word)  
   - Trigger source (ADC, UART, SPI, timer, etc.)

2. CPU starts DMA channel

3. DMA hardware moves data **autonomously**

4. DMA raises an interrupt when:
   - Transfer completes  
   - Half‑transfer occurs  
   - Error happens  

---

## 4. DMA Transfer Types

### 4.1 Peripheral → Memory  
Example: ADC continuously writing samples into a buffer.

### 4.2 Memory → Peripheral  
Example: Sending a buffer over UART or SPI.

### 4.3 Memory → Memory  
Example: Fast block copy or filling RAM.

### 4.4 Peripheral → Peripheral  
Rare, but possible on some MCUs.

---

## 5. DMA Trigger Sources

DMA can be triggered by:

- ADC end‑of‑conversion  
- UART RX/TX events  
- SPI RX/TX  
- Timer events  
- PWM events  
- Software trigger  

Each DMA channel is usually mapped to specific peripherals.

---

## 6. DMA Modes

### 6.1 Normal Mode  
- Transfer stops after N bytes  
- Interrupt fires once

### 6.2 Circular Mode  
- Buffer restarts automatically  
- Ideal for continuous ADC sampling or audio streams

### 6.3 Double Buffer Mode (Ping‑Pong)  
- Two alternating buffers  
- CPU processes one buffer while DMA fills the other  
- Zero data loss

---

## 7. DMA Advantages

- **CPU offloading**  
- **Higher throughput**  
- **Lower latency**  
- **Deterministic timing**  
- **Reduced power consumption**  
- **Efficient continuous data streaming**

---

## 8. DMA Limitations

- Complex configuration  
- Memory alignment requirements  
- Limited number of channels  
- Possible bus contention  
- Must avoid overwriting active buffers  
- Debugging can be harder  

---

## 9. DMA in Embedded C (Generic Example)

### Configure DMA for ADC → Memory

```c
DMA->SRC = (uint32_t)&ADC->DR;      // ADC data register
DMA->DST = (uint32_t)adc_buffer;    // memory buffer
DMA->COUNT = BUFFER_SIZE;

DMA->CTRL = DMA_ENABLE | DMA_PERIPH_TO_MEM | DMA_CIRCULAR;

DMA->START = 1;
```
## DMA interrupt handler
```c
void DMA_IRQHandler(void)
{
    if (DMA->STATUS & DMA_TC)       // transfer complete
    {
        process_buffer();
        DMA->STATUS = DMA_TC;       // clear flag
    }
}
```
## 10. DMA Use Cases
    - ADC continuous sampling
    - Audio input/output
    - Camera/video capture
    - UART high‑speed logging
    - SPI flash read/write
    - Memory block transfers
    - PWM waveform generation
    - Motor control (FOC)

