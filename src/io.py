import os
import torch
import json

def save_model_weights(model, dir, filename):
    """
    Saves the model weights to a specified directory.
    """
    save_path = os.path.join(dir, filename)

    # Ensure the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 2. Save the state dictionary of the model
    # The .state_dict() method extracts all learned parameters (weights and biases)
    torch.save(model.model.state_dict(), save_path)

    print(f"\nModel weights successfully saved to: {save_path}")

def load_model_weights(dir, filename):
    """
    Loads the model weights from a specified directory.
    """
    load_path = os.path.join(dir, filename)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No weights file found at: {load_path}")

    # Load the state dictionary into the model
    state_dict = torch.load(load_path, weights_only=True)

    print(f"\nModel weights successfully loaded from: {load_path}")

    return state_dict

def save_json(json_dicts, file_names, output_dir="./annotated_output"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (json_dict, file_name) in enumerate(zip(json_dicts, file_names)):
        print(f"Saving JSON file: {i+1}/{len(json_dicts)}")
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(json_dict, f, indent=2)