import serial
import numpy as np
import torch
from quantization_params import *
from model_util import dequantize


def intit_serial_com():
    PORT   = 'COM11'  
    BAUD   = 3000000    

    return serial.Serial(PORT, BAUD, timeout=10)

def send_frame(ser: serial.Serial, frame: np.ndarray) -> int:

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(frame)}")
    
    data = frame.astype(np.uint8).tobytes()
    total = len(data)

    written = ser.write(data)
    if written != total:
        raise RuntimeError(f"Only wrote {written}/{total} bytes")
    return written


def read_output(ser: serial.Serial) -> tuple[torch.Tensor, torch.Tensor]:

    # sizes
    N_R, C_R = 896, 16
    N_C = 1
    N_OUT = N_R * C_R + N_R * N_C  # 15232

    # 1) read exactly N_OUT bytes
    data = bytearray()
    while len(data) < N_OUT:
        chunk = ser.read(N_OUT - len(data))
        if not chunk:
            break
        data.extend(chunk)
    if len(data) != N_OUT:
        raise RuntimeError(f"Expected {N_OUT} bytes, got {len(data)}")

    
    arr = np.frombuffer(data, dtype=np.uint8)

    
    r_np = dequantize(arr[: N_R * C_R].reshape(N_R, C_R), r_zero, r_scale)
    c_np = dequantize(arr[N_R * C_R :].reshape(N_R, N_C), c_zero, c_scale)

    # 3) to torch and add batch dim 
    c = torch.from_numpy(c_np).unsqueeze(0)  
    r = torch.from_numpy(r_np).unsqueeze(0)  

    return [r, c]