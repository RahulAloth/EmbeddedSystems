# ADC (Analog‑to‑Digital Converter)  
### Master‑Level Study Notes for Embedded Systems

An **ADC** converts a continuous analog voltage into a discrete digital number.  
It is the bridge between the physical world (sensors) and digital processing (microcontrollers).

---

## 1. What an ADC Does

An ADC performs three fundamental operations:

### 1.1 Sampling  
Captures the analog signal at fixed time intervals.  
Sampling frequency \( f_s \) must satisfy Nyquist:



\[
f_s \ge 2 \cdot f_{max}
\]



### 1.2 Quantization  
Maps each sampled voltage to one of many discrete levels.

### 1.3 Encoding  
Outputs a binary number representing the quantized level.

---

## 2. Resolution

Resolution defines how many digital steps the ADC can output.

| Resolution | Levels | Example LSB (3.3V ref) |
|-----------|--------|-------------------------|
| 8‑bit | 256 | 12.9 mV |
| 10‑bit | 1024 | 3.22 mV |
| 12‑bit | 4096 | 0.8 mV |
| 16‑bit | 65536 | 0.05 mV |

LSB size:



\[
LSB = \frac{V_{ref}}{2^N}
\]



Higher resolution → finer voltage measurement.

---

## 3. Sampling Rate

Defines how fast the ADC takes measurements.

Typical values:
- Slow sensors: 1–10 kHz  
- Motor control: 10–100 kHz  
- Audio: 44.1 kHz  
- High‑speed ADCs: MHz range  

Nyquist rule applies.

---

## 4. ADC Input Types

### 4.1 Single‑Ended  
- One input pin (AINx)  
- Measured relative to GND  
- Most common in MCUs  

### 4.2 Differential  
- Two inputs: AIN+ and AIN−  
- Measures the difference  
- Better noise immunity  
- Used in precision systems  

---

## 5. Reference Voltage (Vref)

The ADC compares input voltage to a reference.

Types:
- **Internal Vref** (1.2V, 2.5V)  
- **External Vref** (high precision)  
- **Vdd as Vref** (least accurate)

Accuracy depends heavily on Vref stability.

---

## 6. ADC Conversion Formula

For an N‑bit ADC:



\[
V_{in} = \frac{ADC\_value}{2^N - 1} \cdot V_{ref}
\]



Example:  
12‑bit ADC, reading = 2048, Vref = 3.3V



\[
V_{in} \approx 1.65V
\]



---

## 7. ADC Architectures

### 7.1 SAR (Successive Approximation Register)
- Most common in microcontrollers  
- Medium speed (100 ksps – few Msps)  
- Good accuracy  

### 7.2 Sigma‑Delta (ΔΣ)
- Very high resolution (16–24 bits)  
- Low speed  
- Used in audio and precision sensors  

### 7.3 Flash ADC
- Extremely fast (GHz)  
- Used in oscilloscopes and RF  

### 7.4 Pipeline ADC
- High speed + good resolution  
- Used in high‑performance data acquisition  

---

## 8. ADC Errors & Non‑Idealities

Real ADCs are imperfect. Key error sources:

- **Offset error**  
- **Gain error**  
- **INL (Integral Non‑Linearity)**  
- **DNL (Differential Non‑Linearity)**  
- **Noise**  
- **Sampling capacitor droop**  
- **Input impedance mismatch**

Calibration and filtering help reduce these.

---

## 9. Practical ADC Hardware Considerations

- Use **RC low‑pass filters** on inputs  
- Keep analog traces short  
- Separate analog and digital grounds  
- Use shielded cables for sensors  
- Ensure sensor output impedance is low enough  
- Avoid switching noise during sampling (PWM edges)  

---

## 10. Oversampling & Averaging

Oversampling improves effective resolution.

Example:  
Oversampling by 4× → +1 bit  
Oversampling by 16× → +2 bits

Averaging reduces noise and stabilizes readings.

---

## 11. Basic ADC Workflow in Embedded C

### 11.1 Configure ADC
- Select channel  
- Set sampling time  
- Choose Vref  
- Enable ADC  

### 11.2 Start conversion
```c
ADC->CR |= ADC_CR_START;
