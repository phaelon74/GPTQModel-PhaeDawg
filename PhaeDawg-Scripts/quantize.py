import os
import yaml
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.model import find_modules, get_module_by_name_prefix
from gptqmodel.quantization.gptq import GPTQ
import torch
import glob
import tqdm

# --- Utility Functions ---
def get_layer_modules(model_def):
    """Gets the list of modules to be quantized in each layer."""
    if model_def.dynamic_expert_index is not None:
        from gptqmodel.models.base import get_moe_layer_modules
        num_experts = getattr(model_def.model.config, model_def.dynamic_expert_index)
        return get_moe_layer_modules(layer_modules=model_def.layer_modules, num_experts=num_experts)
    return model_def.layer_modules

def get_layers(model_def):
    """Gets the layers of the model."""
    return get_module_by_name_prefix(model_def.model, model_def.layers_node)

def create_quant_dir(output_dir):
    """Creates the directory to store temporary quantization files."""
    quant_dir = os.path.join(output_dir, "quant_layers")
    os.makedirs(quant_dir, exist_ok=True)
    return quant_dir

# --- Main Functions ---
def quantize_and_save_layers(config):
    """
    Performs layer-by-layer quantization and saves each quantized layer to disk.
    This avoids holding the entire quantized model in RAM.
    """
    # --- Environment Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_devices"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    output_dir = config["output_dir"]
    quant_layers_dir = create_quant_dir(output_dir)

    # --- Quantization and Calibration Config ---
    quant_config = QuantizeConfig(**config["quantization_config"])
    calibration_config = config["calibration"]

    # --- Model and Tokenizer ---
    model_id = config["model_id"]
    print(f"Loading model: {model_id}")
    # Load the model on CPU to start, layers will be moved to GPU one by one
    model = GPTQModel.load(model_id, quant_config, device_map=None)
    
    # --- Calibration Dataset ---
    print("Preparing calibration dataset...")
    calibration_dataset = model.prepare_dataset(
        calibration_dataset=load_dataset(
            calibration_config["dataset"],
            data_files=calibration_config["subset"],
            split="train"
        ).select(range(calibration_config["num_samples"]))["text"],
        batch_size=calibration_config["batch_size"]
    )

    # --- Layer-by-Layer Quantization Loop ---
    print("Starting layer-by-layer quantization...")
    
    layers = get_layers(model)
    layer_modules_list = get_layer_modules(model)
    
    # Get model inputs by running a forward pass with a hook
    layer_inputs = []
    attention_masks = []
    position_ids = []

    def store_input_hook(_, args, kwargs):
        layer_inputs.append([arg.to('cpu') for arg in args])
        attention_masks.append(kwargs.get("attention_mask").to('cpu') if kwargs.get("attention_mask") is not None else None)
        position_ids.append(kwargs.get("position_ids").to('cpu') if kwargs.get("position_ids") is not None else None)
        raise ValueError("Stopping forward pass after capturing inputs.")

    handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    
    # Move base modules to the first GPU for the forward pass
    for module_name in model.base_modules:
        module = get_module_by_name_prefix(model.model, module_name)
        if module is not None:
            module.to(f"cuda:0")

    for example in tqdm.tqdm(calibration_dataset, desc="Capturing model inputs"):
        try:
            model.model(**{k: v.to(f"cuda:0") for k, v in example.items()})
        except ValueError:
            pass
    
    handle.remove()

    # Move base modules back to CPU
    for module_name in model.base_modules:
        module = get_module_by_name_prefix(model.model, module_name)
        if module is not None:
            module.to('cpu')
    
    torch.cuda.empty_cache()

    # Main quantization loop
    for i in tqdm.tqdm(range(len(layers)), desc="Quantizing Layers"):
        layer = layers[i]
        layer.to(f"cuda:0")
        
        # Find all linear modules in the current layer
        full_modules = find_modules(layer)
        
        for module_names in layer_modules_list:
            subset = {name: full_modules[name] for name in module_names if name in full_modules}
            
            gptq_handlers = {}
            for name, mod in subset.items():
                gptq_handlers[name] = GPTQ(mod, quant_config)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_handlers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name, mod in subset.items():
                handles.append(mod.register_forward_hook(add_batch(name)))

            # Run forward pass for calibration on the current layer
            for j in range(len(calibration_dataset)):
                layer_input_args = [arg.to(f"cuda:0") for arg in layer_inputs[j]]
                layer_attention_mask = attention_masks[j].to(f"cuda:0") if attention_masks[j] is not None else None
                layer_position_ids = position_ids[j].to(f"cuda:0") if position_ids[j] is not None else None
                
                layer(
                    *layer_input_args, 
                    attention_mask=layer_attention_mask, 
                    position_ids=layer_position_ids
                )

            for h in handles:
                h.remove()
            
            # Quantize and save each module
            for name, gptq_handler in gptq_handlers.items():
                print(f"Quantizing layer {i}, module {name}...")
                quantized_weight, scale, zero, g_idx, _, _, _, _ = gptq_handler.quantize()
                
                # Save tensors to disk immediately
                layer_quant_data = {
                    'weight': quantized_weight.cpu(),
                    'scale': scale.cpu(),
                    'zero': zero.cpu(),
                    'g_idx': g_idx.cpu()
                }
                
                module_path = os.path.join(quant_layers_dir, f"layer_{i}_{name}.pt")
                torch.save(layer_quant_data, module_path)

                gptq_handler.free() # Free up memory
        
        layer.to('cpu')
        torch.cuda.empty_cache()

    print("Layer-by-layer quantization complete.")
    print(f"Temporary quantized files saved in: {quant_layers_dir}")

def main():
    with open("PhaeDawg-Scripts/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    quantize_and_save_layers(config)

if __name__ == "__main__":
    main() 