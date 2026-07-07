# Camera Pipeline Overview  
A modern camera pipeline in automotive/robotics consists of four major stages:

1. **MIPI‑CSI** – High‑speed interface carrying RAW pixel data from the sensor  
2. **ISP (Image Signal Processor)** – Converts RAW → usable image (denoise, demosaic, color)  
3. **HDR Pipeline** – Merges multiple exposures or uses DOL (Digital Overlap) frames  
4. **Dewarp / Geometric Correction** – Corrects lens distortion (fisheye, wide‑angle)

This pipeline feeds ADAS perception, SLAM, robotics vision, and neural networks.

---

## 1. MIPI‑CSI (Camera Serial Interface)

### Purpose  
MIPI‑CSI is the physical + protocol layer connecting the camera sensor to the SoC.

### Key Concepts  
- **CSI‑2**: Most common version  
- **Lanes**: 1, 2, 4, or 8 data lanes  
- **RAW Bayer formats**: RAW8, RAW10, RAW12, RAW14  
- **D‑PHY / C‑PHY**: Physical layer signaling  
- **Packetized transmission**: Short packets (metadata) + long packets (pixel data)

### What it delivers  
- Pure **RAW Bayer** frames  
- No processing  
- High bandwidth (up to multiple Gbps per lane)

---

## 2. ISP (Image Signal Processor)

### Purpose  
The ISP converts RAW sensor data into a usable image for perception or display.

### Core ISP Stages  
- **Black level correction**  
- **Lens shading correction (LSC)**  
- **Demosaicing** (Bayer → RGB)  
- **Noise reduction** (temporal + spatial)  
- **Color correction matrix (CCM)**  
- **Gamma correction**  
- **Auto‑exposure (AE)**  
- **Auto‑white‑balance (AWB)**  
- **Auto‑focus (AF)**  
- **Sharpening**  
- **Tone mapping**

### Output formats  
- RGB888  
- YUV420 / NV12  
- RAW (after ISP tuning)  
- HDR‑merged frames

ISPs in automotive SoCs (Renesas R‑Car, TI TDAx, NVIDIA Orin) include hardware blocks for real‑time processing at 60–120 FPS.

---

## 3. HDR Pipeline (High Dynamic Range)

### Why HDR is needed  
Automotive cameras face extreme lighting: tunnels, sunlight, reflections, night scenes.

### HDR Techniques  
#### A. **Multi‑Exposure HDR**
Sensor captures:
- Short exposure (highlights)  
- Medium exposure  
- Long exposure (shadows)

ISP merges them into one HDR frame.

#### B. **DOL HDR (Digital Overlap)**
Sensor outputs multiple exposures **in the same frame**:
- DOL‑2 (two exposures)  
- DOL‑3 (three exposures)

Advantages:
- Lower motion artifacts  
- Better temporal alignment

#### C. **Local Tone Mapping**
Compresses dynamic range while preserving contrast.

### HDR Output  
- 16‑bit linear HDR  
- 12‑bit compressed HDR  
- YUV HDR for perception pipelines

---

## 4. Dewarp / Geometric Correction

### Why dewarp is needed  
Wide‑angle and fisheye lenses distort geometry:
- Straight lines appear curved  
- Objects appear stretched near edges

### Dewarp Techniques  
- **Radial distortion correction**  
- **Tangential distortion correction**  
- **Fisheye → rectilinear projection**  
- **Equirectangular projection** (for 360° cameras)  
- **Perspective correction**  
- **Homography transforms**

### Dewarp Output  
- Rectified image  
- Undistorted image for perception  
- Multiple virtual camera views (surround view)

---

## Full Pipeline Diagram
```
Camera Sensor
│
▼
[MIPI-CSI Interface]
│ RAW Bayer
▼
[ISP Pipeline]
│ Demosaic, NR, CCM, Gamma
▼
[HDR Merge]
│ Multi-exposure / DOL HDR
▼
[Dewarp Engine]
│ Geometric correction
▼
[ADAS / Robotics Perception]
│ CNNs, SLAM, Object Detection
```

---

## Automotive Example (Surround View)

| Stage | Purpose | Output |
|------|---------|--------|
| MIPI‑CSI | Capture RAW frames | RAW10 |
| ISP | Demosaic + NR + CCM | RGB/YUV |
| HDR | Merge exposures | HDR12 |
| Dewarp | Fisheye → rectilinear | Rectified image |
| Stitching | Combine 4 cameras | Bird’s‑eye view |
| Perception | Detect lanes, objects | Metadata |

---

## Summary  
- **MIPI‑CSI** brings RAW sensor data into the SoC.  
- **ISP** converts RAW → usable image with color, noise reduction, and tuning.  
- **HDR** handles extreme lighting using multi‑exposure or DOL frames.  
- **Dewarp** corrects lens distortion for perception algorithms.  
- This pipeline is foundational for ADAS, robotics, and autonomous systems.

# Universal Guide: Connecting Any CSI‑2 Camera to Renesas R‑Car Gen5 as an example.

This document explains how to connect **any** MIPI CSI‑2 camera sensor to **any** R‑Car Gen5 SoC (V4H, V4M, V3U, S4) as an example to connect Camera with an Embedded SoC.
It is sensor‑agnostic and applies to all RAW/YUV CSI‑2 cameras.

---

# 1. R‑Car Gen5 Camera Architecture (Generic)

Every R‑Car Gen5 SoC uses this pipeline:
```
Camera Sensor → CSI‑IF → VIN → Memory → (optional ISP/VSP)
```


### CSI‑IF (MIPI Receiver)
- Receives CSI‑2 packets  
- Handles LP/HS switching  
- Performs lane merging  
- Reports ECC/CRC errors  
- Forwards frames to VIN

### VIN (Video Input Engine)
- Converts CSI‑2 RAW/YUV formats  
- Cropping and scaling  
- DMA to memory  
- Exposes `/dev/videoX` via V4L2

---

# 2. Hardware Requirements (Applies to ANY Sensor)

### Mandatory Signals
- **MIPI CSI‑2 Clock Lane:** CLK+, CLK−  
- **MIPI CSI‑2 Data Lanes:** D0+/D0−, D1+/D1−, … (1–4 lanes depending on sensor)  
- **I²C:** SDA, SCL (sensor control)  
- **Power Rails:** AVDD, DVDD, IOVDD  
- **Reset:** XCLR / RESET pin  
- **Clock Input:** XCLK (usually 24 MHz)

### Universal Electrical Rules
- Lane polarity must match  
- Lane count must match sensor output  
- Termination resistors must follow MIPI spec  
- Power‑up sequence must follow sensor datasheet  
- XCLK must be stable before sensor init

---

# 3. Universal Device Tree Template

This template works for **any CSI‑2 sensor**.  
You only change:
- `compatible`  
- `reg`  
- `data-lanes`  
- `bus-width`  
- clocks/reset GPIOs  

```dts
&i2c2 {
    camera@3c {
        compatible = "<your-sensor-compatible>";
        reg = <0x3c>;

        clocks = <&cpg CPG_MOD <clock-id>>;
        clock-names = "xclk";

        reset-gpios = <&gpioX Y GPIO_ACTIVE_LOW>;

        port {
            camera_ep: endpoint {
                bus-width = <number_of_lanes>;
                data-lanes = <1 2>;   // Example: 2 lanes
                remote-endpoint = <&csi0_ep>;
            };
        };
    };
};

&csi0 {
    port {
        csi0_ep: endpoint {
            remote-endpoint = <&camera_ep>;
        };
    };
};

&vin0 {
    port {
        vin0_ep: endpoint {
            remote-endpoint = <&csi0_ep>;
        };
    };
};
```
This is the minimum required media graph:

```
camera_ep ↔ csi0_ep ↔ vin0_ep

```

# Kernel Requirements (Sensor‑Agnostic)

## Required Drivers
- Sensor driver (any V4L2 subdevice driver)
- R‑Car CSI‑IF driver
- R‑Car VIN driver
- Media Controller framework
- V4L2 subdevice API

## Kernel Config Flags
```bash
CONFIG_VIDEO_RCAR_CSI2=y
CONFIG_VIDEO_RCAR_VIN=y
CONFIG_MEDIA_CONTROLLER=y
CONFIG_V4L2_SUBDEV_API=y
```

---

# Universal Bring‑Up Checklist

## Step 1 — Power & I²C
- Sensor powers up
- I²C responds (`i2cdetect`)
- XCLK is present (oscilloscope recommended)

## Step 2 — CSI‑2 Link
- LP mode active at boot
- HS mode active when streaming
- No ECC/CRC errors in CSI‑IF logs
- Lane count matches Device Tree

## Step 3 — VIN
- VIN detects pixel format
- VIN DMA buffers allocated
- Media graph is complete (`media-ctl -p`)

---

# Universal Streaming Commands

## List Media Graph
```bash
media-ctl -p
```

## Configure Format (example RAW10)
```bash
media-ctl -V '"vin0":0 [fmt:SRGGB10_1X10/1920x1080]'
```

## Start Streaming
```bash
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=100
```

## Capture a Frame
```bash
v4l2-ctl -d /dev/video0 --stream-to=frame.raw --stream-count=1
```

---

# Universal Debugging Guide

## CSI‑IF Errors
- ECC/CRC errors → lane polarity or timing issue  
- No HS mode → sensor not configured correctly  
- No clock lane activity → sensor not outputting CSI‑2  

## VIN Errors
- Wrong pixel format → fix sensor driver  
- DMA errors → incorrect DT memory region  
- No `/dev/video0` → media graph incomplete  

## Media Graph Issues
- Missing links → incorrect `remote-endpoint`  
- Wrong bus-width → mismatch with sensor  
- Wrong lane count → sensor output mismatch  

---

# Summary

To connect **any CSI‑2 camera** to **R‑Car Gen5**:

1. Wire CSI lanes + I²C + power + reset + XCLK  
2. Add sensor + CSI‑IF + VIN nodes in Device Tree  
3. Ensure correct lane count and polarity  
4. Verify CSI‑2 LP/HS mode  
5. Stream using V4L2 tools  

This process is identical for all CSI‑2 sensors.

---



