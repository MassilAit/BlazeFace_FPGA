# BlazeFace FPGA Acceleration

This project implements a fully quantized and hardware-accelerated version of the BlazeFace neural network on the PYNQ-Z2 FPGA board. It includes model quantization, HLS hardware accelerators, a Vivado block design, C++ inference on the PYNQ, and Python code to drive everything from a PC.

---

## 📁 Project Structure

| Folder        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `hls/`        | Vitis HLS projects for accelerating neural network operations (ReLU, DWConv, PWConv). |
| `vivado/`     | Vivado block design project containing the top-level system and hardware integration. |
| `pynq_cpp/`   | C++ code running on the PYNQ board for managing inference and serial communication. |
| `host_python/`| Python code running on the host PC for sending frames and reading results over serial. |
| `qunatization/`| Scripts and utilities to quantize the BlazeFace model and export its parameters. |

---
