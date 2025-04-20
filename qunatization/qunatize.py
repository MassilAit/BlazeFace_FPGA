from model_q import QBlazeFace
import torch
import json
from util import save_state_dict

# 1. Load model:
qnet=QBlazeFace()
qnet.to('cpu')
qnet.load_weights("blazeface.pth")


# 2. Observer on dummy inputs ([-1, 1])

qnet.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(qnet, inplace=True) 

with torch.no_grad():
    for _ in range(10):
        dummy_input = 2 * torch.rand(1, 3, 128, 128) - 1
        qnet(dummy_input)


# 3. Runing PTQ 

torch.quantization.convert(qnet, inplace=True)


# 4. Save result

save_state_dict(qnet, "qblazeface_weights.json")





