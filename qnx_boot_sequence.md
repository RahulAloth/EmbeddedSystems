# QNX Boot Sequence  

This document explains how a QNX Neutrino system boots—from CPU reset to the point where your drivers and applications start running.  
It follows the exact sequence used in embedded boards, SoCs, and automotive ECUs.

---

## 1. High‑Level Boot Flow

```
Start → CPU → BIOS / ROM Monitor → IPL → Startup → procnto → Boot Script → Drivers / Processes → Done
```

The boot sequence has two major phases:

- **Board / SoC vendor phase**  
  Hardware‑specific initialization (BIOS, ROM monitor, IPL)

- **QNX phase**  
  Startup code, microkernel, process manager, boot script, system processes

---

## 2. Boot Components Explained

### 2.1 CPU Reset
When the system powers on or resets:
- The CPU begins execution at a predefined reset vector.
- It loads the first instructions from ROM, flash, or bootloader memory.

---

## 3. Vendor‑Specific Boot Components

### 3.1 BIOS and Extensions (x86)
On x86 systems:
- BIOS initializes hardware (memory controller, PCI, timers).
- Option ROMs may run (e.g., network boot, storage controllers).
- Control eventually passes to the QNX IPL.

### 3.2 ROM Monitor (ARM / Embedded)
On embedded boards:
- A ROM monitor or bootloader (U‑Boot, RedBoot, proprietary loader) runs.
- Provides basic commands, flash access, and boot configuration.
- Loads and executes the QNX IPL.

---

## 4. QNX Boot Components

### 4.1 IPL (Initial Program Loader)
The **IPL** is the first QNX‑specific code executed.

Responsibilities:
- Configure chip selects  
- Initialize DRAM  
- Set up basic memory mapping  
- Load the **startup** image into RAM  
- Jump to the startup code

The IPL is extremely hardware‑specific.

---

### 4.2 Startup Code
The **startup** code prepares the environment for the QNX kernel.

Responsibilities:
- Initialize CPU/MMU  
- Set up interrupt controllers  
- Configure clocks and timers  
- Prepare memory regions  
- Build system page (syspage) describing hardware  
- Load and launch **procnto**

Startup is the bridge between hardware and the OS.

---

### 4.3 procnto (Process Manager + Microkernel)
`procnto` is the first real process in QNX (PID 1).  
It contains two tightly bound components:

#### Microkernel
- Thread scheduling  
- IPC (message passing)  
- Interrupt handling  
- Timer services  

#### Process Manager
- Process creation  
- Memory management  
- Path resolution  
- File descriptor management  

Together, they form the core of the QNX runtime.

---

## 5. Boot Script

After `procnto` starts, it executes the **boot script** embedded in the image.

The boot script typically launches:
- Disk drivers (`devb-*`)  
- Network stack (`io-sock`)  
- Character drivers (`devc-*`)  
- Bus managers (`pci-server`, `io-usb-*`)  
- System daemons (`cron`, `sshd`, `qconn`)  
- Your application processes  

Example boot script snippet:
```sh
[+session] devb-eide &
io-sock &
pci-server &
devc-ser8250 &
my_app &
```

The boot script defines the initial runtime environment.

---

## 6. End of Boot Sequence

Once the boot script finishes:
- All essential drivers are running  
- System daemons are active  
- Your applications are launched  
- The system is fully operational  

At this point, QNX is ready for real‑time scheduling, IPC, networking, and user interaction.

---

## 7. Summary

- **IPL** initializes RAM and jumps to startup.  
- **Startup** configures hardware and prepares the kernel environment.  
- **procnto** launches the microkernel + process manager.  
- **Boot script** starts drivers, daemons, and your processes.  
- The system becomes fully operational.

This sequence ensures deterministic, modular, and reliable startup—critical for embedded and automotive systems.

