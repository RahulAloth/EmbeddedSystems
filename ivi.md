# IVI Architecture (SoC, GPU, Display Pipeline, Audio, Connectivity)

An In‑Vehicle Infotainment (IVI) system is a **heterogeneous compute platform** combining a high‑performance SoC, GPU/accelerators, multimedia pipelines, audio DSPs, and connectivity modules. Below is a clean, engineering‑grade breakdown of the full IVI architecture.

---

## 1. System-on-Chip (SoC)

The IVI SoC is the central compute element. It typically includes:

- **CPU clusters** (Arm Cortex‑A, Cortex‑R for safety islands)  
- **GPU** for rendering UI, maps, animations  
- **NPU / AI accelerators** for voice, gesture, ML workloads  
- **ISP** for camera ingestion (rear camera, driver monitoring)  
- **Video codecs** (H.264/H.265/AV1)  
- **Hardware security modules**  
- **Memory controllers** (LPDDR4/5)

### Responsibilities
- Runs **Android Automotive OS**, QNX, Linux, or AUTOSAR Adaptive  
- Manages UI, apps, navigation, media, connectivity  
- Handles safety‑critical display paths (rear camera, cluster)

---

## 2. GPU & Rendering Pipeline

The GPU handles all graphical rendering:

### GPU Responsibilities
- **UI composition** (SurfaceFlinger / Wayland compositor)  
- **OpenGL ES / Vulkan rendering**  
- **Map rendering** (vector maps, 3D navigation)  
- **Animations & transitions**  
- **Cluster rendering** (speedometer, ADAS visualization)

### Rendering Flow
```
App → Rendering API (OpenGL/Vulkan) → GPU → Composition Engine → Display Controller
```


GPU also supports:
- Hardware overlays  
- Multi‑display pipelines  
- Low‑latency camera rendering (rear‑view)

---

## 3. Display Pipeline

The display subsystem includes:

### Components
- **Display Controller (DC)**  
- **Timing Controller (TCON)**  
- **LVDS / eDP / MIPI‑DSI interfaces**  
- **Backlight controller**  
- **Touch controller (I2C)**

### Display Pipeline Flow
```
GPU Framebuffer
↓
Hardware Composer / SurfaceFlinger
↓
Display Controller (scaling, blending)
↓
MIPI‑DSI / eDP / LVDS
↓
LCD / OLED Panel
```

### Features
- Multi‑layer composition  
- Hardware scaling  
- HDR tone‑mapping  
- Safety‑critical bypass path (camera → display without GPU)

---

## 4. Audio Architecture

IVI audio uses a combination of SoC DSPs and external amplifiers.

### Components
- **Audio DSP** (for mixing, EQ, ANC)  
- **I2S / TDM interfaces**  
- **External amplifier (Class‑D)**  
- **Microphone array**  
- **Voice processing engine** (beamforming, echo cancellation)

### Audio Pipeline
```
App (Media/Navigation/Voice)
↓
Audio HAL / ALSA / AAudio
↓
Audio DSP (mixing, EQ, effects)
↓
I2S/TDM
↓
Amplifier → Speakers
```

### Features
- Multi‑zone audio  
- ANC (Active Noise Cancellation)  
- Voice assistant wake‑word detection  
- Bluetooth A2DP/SBC/AAC/LDAC decoding

---

## 5. Connectivity Subsystem

IVI systems integrate multiple connectivity modules:

### External Connectivity
- **Wi‑Fi 6/7**  
- **Bluetooth 5.x**  
- **LTE/5G modem**  
- **GNSS (GPS/GLONASS/Galileo)**  
- **UWB** (for digital keys)

### In‑Vehicle Connectivity
- **Ethernet AVB/TSN** (camera, ADAS data)  
- **CAN / CAN FD** (vehicle signals)  
- **LIN** (HVAC, buttons)  
- **FlexRay** (legacy)  
- **MOST** (older IVI systems)

### Connectivity Flow
```
Modem/Wi-Fi/Bluetooth
↓
Connectivity HAL / Network Stack
↓
Apps (Maps, OTA, Streaming)
```

---

## 6. Full IVI Architecture Diagram (Text)
```
+-----------------------------+
|           IVI SoC           |
|-----------------------------|
| CPU Clusters                |
| GPU / NPU / ISP             |
| Video Codecs                |
| Audio DSP                   |
| Security Module             |
+--------------+--------------+
|
+----------------------+----------------------+
|                      |                      |
Display Pipeline        Audio Pipeline        Connectivity
|                      |                      |
MIPI-DSI/eDP/LVDS       I2S/TDM → Amp         Wi-Fi/BT/LTE/5G
|                      |                      |
LCD/OLED Panel         Speakers/Mics         Vehicle Networks

```


---

## 7. Typical IVI Software Stack

### OS Layer
- Android Automotive OS  
- Linux + Wayland  
- QNX + Qt  
- AUTOSAR Adaptive (for safety‑critical IVI)

### Middleware
- Audio HAL  
- Camera HAL  
- Display HAL  
- Connectivity HAL  
- Vehicle HAL (CAN/Ethernet signals)

### Applications
- Navigation  
- Media  
- Voice assistant  
- Vehicle settings  
- ADAS visualization  
- Rear camera view

---

## 8. Summary

- **SoC**: Central compute with CPU/GPU/NPU/ISP  
- **GPU**: Renders UI, maps, cluster graphics  
- **Display pipeline**: Composes frames and drives LCD/OLED  
- **Audio pipeline**: DSP‑based mixing, effects, multi‑zone  
- **Connectivity**: Wi‑Fi/BT/LTE/5G + CAN/Ethernet for vehicle integration  

This architecture enables modern IVI systems with rich graphics, multi‑zone audio, connectivity, and safety‑critical camera rendering.

---

