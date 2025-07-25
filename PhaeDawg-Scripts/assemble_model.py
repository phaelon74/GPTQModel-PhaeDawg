import os
import yaml
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.model import find_modules, get_module_by_name_prefix, pack_model
import torch
import glob

def assemble_quantized_model(config):
    """
    Assembles the final quantized model from the temporary layer files saved to disk.
    """
    # --- Environment and Config ---
    output_dir = config["output_dir"]
    quant_layers_dir = os.path.join(output_dir, "quant_layers")
    
    if not os.path.isdir(quant_layers_dir):
        raise FileNotFoundError(f"Quantized layer directory not found: {quant_layers_dir}. Please run the quantize.py script first.")

    model_id = config["model_id"]
    quant_config = QuantizeConfig(**config["quantization_config"])

    # --- Load Base Model ---
    print(f"Loading base model: {model_id}")
    # Load on CPU to minimize VRAM usage during assembly
    model = GPTQModel.load(model_id, quant_config, device_map=None)
    
    # --- Load Quantized Layers from Disk ---
    print("Loading quantized layer data from disk...")
    quantizers = {}
    
    layer_files = glob.glob(os.path.join(quant_layers_dir, "*.pt"))
    if not layer_files:
        raise FileNotFoundError(f"No quantized layer files found in {quant_layers_dir}.")

    for file_path in tqdm.tqdm(layer_files, desc="Loading layer files"):
        layer_data = torch.load(file_path, map_location='cpu')
        
        # Reconstruct the module name from the file path
        file_name = os.path.basename(file_path)
        parts = file_name.replace('.pt', '').split('_')
        layer_index = parts[1]
        module_name = "_".join(parts[2:])
        
        # The quantizers dict needs a specific structure
        full_module_name = f"{model.layers_node}.{layer_index}.{module_name}"
        
        # We need a mock quantizer object with the scale and zero tensors
        mock_quantizer = torch.nn.Module()
        mock_quantizer.scale = layer_data['scale']
        mock_quantizer.zero = layer_data['zero']
        
        quantizers[full_module_name] = (
            mock_quantizer,
            layer_data['scale'],
            layer_data['zero'],
            layer_data['g_idx']
        )
        # Also, we need to manually set the quantized weight data
        # This is a bit of a hack, as we are working around the library's in-memory design
        module_to_update = get_module_by_name_prefix(model.model, full_module_name)
        if module_to_update:
            module_to_update.weight.data = layer_data['weight']

    # --- Pack the Model ---
    print("Packing model...")
    pack_model(
        model=model.model,
        quantizers=quantizers,
        bits=quant_config.bits,
        group_size=quant_config.group_size,
        backend=config.get("backend", "AUTO"),
        desc_act=quant_config.desc_act,
        format=quant_config.format,
        quant_method=quant_config.quant_method,
        pack_dtype=quant_config.pack_dtype
    )
    
    # --- Save Final Model ---
    print(f"Saving final assembled model to: {output_dir}")
    model.save(output_dir)
    print("Model assembly and saving complete.")

def main():
    with open("PhaeDawg-Scripts/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    assemble_quantized_model(config)

if __name__ == "__main__":
    import tqdm # Add tqdm for progress bars
    main() 