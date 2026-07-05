# E/E Architecture (Domain Controllers, Zonal Architecture, Power Distribution)

Modern vehicles are transitioning from legacy distributed ECUs to centralized compute architectures. This evolution improves software complexity management, reduces wiring harness weight, and enables high‑bandwidth ADAS/IVI features.
E/E architecture means the complete Electrical + Electronic architecture of a vehicle — the full system that defines how power, data, compute, sensors, and actuators are organized, connected, and controlled. It is the blueprint of the vehicle’s electronics.
E/E architecture is the overall design of all electronic systems in a vehicle, including ECUs, wiring harnesses, power distribution, communication networks, domain/zonal controllers, and central compute. It determines how the car’s electronics are built, connected, powered, and controlled.

---

## 1. Evolution of E/E Architecture

### 1. Distributed Architecture (Legacy)
- 70–150 ECUs scattered across the vehicle  
- Each ECU handles one function (window lifter, door, HVAC, ABS)  
- CAN/LIN dominated  
- Heavy wiring harness  
- Difficult OTA updates  

### 2. Domain Architecture (Current Generation)
- ECUs grouped into **functional domains**  
- High‑performance domain controllers  
- Ethernet backbone  
- Partial centralization  

### 3. Zonal Architecture (Next Generation)
- Vehicle divided into **zones** (front‑left, front‑right, rear‑left, rear‑right)  
- Each zone has a **zonal controller**  
- Central compute (HPC) runs software functions  
- Smart power distribution integrated into zones  
- Dramatically reduced wiring harness complexity  

---

## 2. Domain Controllers

Domain controllers consolidate ECUs by function:

### Typical Domains
- **ADAS Domain**  
- **IVI / Cockpit Domain**  
- **Body Domain**  
- **Powertrain Domain**  
- **Chassis Domain**  

### Responsibilities
- Run domain‑specific software  
- Provide compute for multiple ECUs  
- Gateway between CAN/Ethernet networks  
- Support OTA updates  
- Enable functional safety partitioning  

### Example Architecture
```
[ADAS Domain Controller]
- Sensor fusion
- Perception
- Planning

[IVI Domain Controller]
- Android Automotive
- Navigation
- Media

[Body Domain Controller]
- Doors
- HVAC
- Lighting
```


---

## 3. Zonal Architecture

Zonal architecture replaces functional domains with **geographical zones**.

### Zones
- Front‑Left Zone  
- Front‑Right Zone  
- Rear‑Left Zone  
- Rear‑Right Zone  
- Central Zone (HPC)

### Zonal Controller Responsibilities
- Local I/O (LIN, CAN, GPIO, PWM)  
- Local power distribution  
- Sensor/actuator aggregation  
- Ethernet uplink to HPC  
- Diagnostics and health monitoring  

### Benefits
- 30–50% reduction in wiring harness weight  
- Simplified manufacturing  
- Centralized software execution  
- Easier OTA updates  
- Improved reliability  

### Zonal Architecture Diagram
```
+---------------------------+
|        Central HPC        |
|  (ADAS + IVI + Body SW)   |
+-------------+-------------+
|
Ethernet Backbone
|
+---------+   +---------+   +---------+   +---------+
| Zone FL |   | Zone FR |   | Zone RL |   | Zone RR |
+---------+   +---------+   +---------+   +---------+
|             |             |             |
Local I/O     Local I/O     Local I/O     Local I/O
Power Dist.   Power Dist.   Power Dist.   Power Dist.
```

---

## 4. Power Distribution (Smart PDUs)

In zonal architecture, power distribution becomes **smart and software‑defined**.

### Components
- Solid‑state switches (MOSFET‑based)  
- Current sensing  
- Load diagnostics  
- Electronic fuses (eFuses)  
- Power gating  
- Thermal monitoring  

### Responsibilities
- Provide power to local actuators/sensors  
- Protect circuits (over‑current, short‑circuit)  
- Report diagnostics to HPC  
- Enable software‑controlled power routing  

### Power Distribution Flow
```
Battery → Power Distribution Unit (PDU)
↓
Zonal Controller
↓
Local Loads (motors, lights, sensors)
```

---

## 5. Communication Backbone

### Legacy
- CAN  
- LIN  
- FlexRay  

### Domain Architecture
- Automotive Ethernet (100BASE‑T1 / 1000BASE‑T1)  
- CAN FD  
- SOME/IP  
- TSN for deterministic traffic  

### Zonal Architecture
- High‑speed Ethernet backbone  
- TSN for ADAS and safety‑critical traffic  
- CAN/LIN only inside zones  

---

## 6. Central Compute (HPC)

The HPC replaces many domain controllers.

### Responsibilities
- Runs ADAS, IVI, body, and chassis software  
- Hosts containerized or Adaptive AUTOSAR applications  
- Provides OTA updates  
- Manages zonal controllers  
- Executes service‑oriented architecture (SOA)

### HPC Architecture
```

CPU + GPU + NPU + ISP
↓
Hypervisor / Adaptive AUTOSAR / Linux
↓
Service-Oriented Middleware
↓
Vehicle Functions (Apps)

```


---

## 7. Summary

- **Domain controllers** consolidate ECUs by function.  
- **Zonal architecture** consolidates ECUs by physical location.  
- **HPC** centralizes compute for ADAS, IVI, and body functions.  
- **Smart power distribution** integrates into zonal controllers.  
- **Ethernet backbone** replaces CAN‑centric networks.  
- This architecture enables OTA, software‑defined vehicles, and reduced wiring complexity.

---
```
            +-----------------------------+
            |        Central HPC          |
            |  (ADAS + IVI + Body SW)     |
            +--------------+--------------+
                           |
                    Ethernet Backbone
                           |
    +---------+    +---------+    +---------+    +---------+
    | Zone FL |    | Zone FR |    | Zone RL |    | Zone RR |
    +---------+    +---------+    +---------+    +---------+
       |              |              |              |
   Sensors/Actuators Sensors/Actuators Sensors/Actuators Sensors/Actuators
   Power Distribution Power Distribution Power Distribution Power Distribution

```



