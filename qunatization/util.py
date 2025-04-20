import json

# Function to process quantized tensors
def process_quantized_tensor(tensor):
    if tensor.is_quantized:
        return {
            "values": tensor.int_repr().tolist(),  # Extract raw int values
            "scale": tensor.q_scale(),
            "zero_point": tensor.q_zero_point()
        }
    else:
        return tensor.detach().cpu().numpy().tolist()  # Normal tensor

def save_state_dict(model, path):
    state_dict = {k: process_quantized_tensor(v) for k, v in model.state_dict().items()}

    # Save to JSON file
    with open(path, "w") as f:
        json.dump(state_dict, f, indent=4)