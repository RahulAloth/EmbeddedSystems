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

