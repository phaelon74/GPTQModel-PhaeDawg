# GPTQModel Quantization Script

This document provides instructions for setting up and running the `quantize.py` script to perform 8-bit, GPTQv2 quantization on large language models using multiple GPUs.

## 1. Prerequisites

Before you begin, ensure your system meets the following requirements:

-   **Operating System:** A modern Linux distribution (e.g., Ubuntu 20.04+).
-   **NVIDIA GPUs:** At least one CUDA-compatible NVIDIA GPU. This script is designed for multi-GPU systems.
-   **NVIDIA Drivers:** The appropriate NVIDIA drivers for your GPUs must be installed.
-   **CUDA Toolkit:** The NVIDIA CUDA Toolkit must be installed. The version should be compatible with the version of PyTorch you intend to use (e.g., CUDA 11.8 or 12.1).

## 2. Installation (From a Bare-Metal Machine)

These steps will guide you through setting up the Python environment and installing all necessary dependencies.

### Step 1: Clone the Repository

First, clone the `GPTQModel-PhaeDawg` repository to your local machine.

```bash
git clone <repository_url>
cd GPTQModel-PhaeDawg
```

### Step 2: Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment to avoid conflicts with system-wide packages.

```bash
# Install venv if you don't have it
# sudo apt-get install python3-venv

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 3: Install PyTorch with CUDA Support

This is a critical step. You must install a version of PyTorch that is compiled for your specific CUDA toolkit version.

1.  Visit the official [PyTorch website](https://pytorch.org/get-started/locally/).
2.  Use the interactive tool to select the correct configuration (e.g., Stable, Linux, Pip, Python, your CUDA version).
3.  Copy the generated command and run it in your terminal. It will look something like this:

    ```bash
    # Example for CUDA 12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

### Step 4: Install All Other Dependencies

Now, install the remaining dependencies required by the project and the quantization script.

```bash
# Install dependencies from the main requirements file and the script-specific ones
pip install -r requirements.txt pyyaml datasets
```

### Step 5: Install the GPTQModel Package

Finally, install the `gptqmodel` package itself in "editable" mode. This allows the Python interpreter to find the package source code within the project directory.

```bash
# Run this command from the root of the GPTQModel-PhaeDawg directory
pip install -e .
```

If you encounter an error about `torch` not being found during this step, you may need to create a `pyproject.toml` file in the root directory.

## 3. Configuration

All settings for the quantization process are controlled by the `config.yaml` file located in the `PhaeDawg-Scripts` directory.

-   `model_id`: The identifier of the model on the Hugging Face Hub or a path to a local model directory.
-   `output_dir`: The directory where the quantized model will be saved.
-   `gpu_devices`: A comma-separated list of GPU IDs to use (e.g., `"0,1,2,3"`).
-   `quantization_config`:
    -   `bits`: Set to `8` for 8-bit (FP8) quantization.
    -   `v2`: Must be `true` to use the GPTQv2 algorithm.
-   `calibration`:
    -   `num_samples`: **This is a critical parameter for managing memory.** For very large models (100B+ parameters), the memory required for caching activations during quantization can be substantial. If you encounter an `OutOfMemoryError`, you must **reduce this value** significantly (e.g., to 64, 32, or even lower) until the process fits within your GPU's VRAM.

## 4. Usage

Once the installation is complete and the `config.yaml` file is configured, you can run the script.

```bash
# Navigate to the script's directory
cd PhaeDawg-Scripts

# Run the quantization script
python quantize.py
```

The script will print progress updates to the console. Upon successful completion, the quantized model will be available in the directory specified by `output_dir` in your configuration file. 