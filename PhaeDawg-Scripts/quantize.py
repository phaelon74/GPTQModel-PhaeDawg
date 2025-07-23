import os
import yaml
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
import torch

def main():
    # Load configuration from YAML file
    with open("PhaeDawg-Scripts/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set CUDA_VISIBLE_DEVICES for multi-GPU support
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_devices"])
    
    # Verify that GPUs are available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

    # --- Quantization Configuration ---
    quantization_config = config["quantization_config"]
    quant_config = QuantizeConfig(
        bits=quantization_config["bits"],
        group_size=quantization_config["group_size"],
        sym=quantization_config["sym"],
        desc_act=quantization_config["desc_act"],
        v2=quantization_config["v2"],
        damp_percent=quantization_config["damp_percent"]
    )

    # --- Calibration Dataset ---
    calibration_config = config["calibration"]
    calibration_dataset = load_dataset(
        calibration_config["dataset"],
        data_files=calibration_config["subset"],
        split="train"
    ).select(range(calibration_config["num_samples"]))["text"]

    # --- Model Loading ---
    model_id = config["model_id"]
    print(f"Loading model: {model_id}")
    model = GPTQModel.load(model_id, quant_config)

    # --- Quantization ---
    print("Starting quantization...")
    model.quantize(
        calibration_dataset,
        batch_size=calibration_config["batch_size"]
    )
    print("Quantization complete.")

    # --- Saving the Quantized Model ---
    output_dir = config["output_dir"]
    print(f"Saving quantized model to: {output_dir}")
    model.save(output_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    main() 