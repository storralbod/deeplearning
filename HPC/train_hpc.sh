#!/bin/bash
# =============================================================================
# Script: train_hpc.sh
# Description: LSF job script to set up Conda environment and train on GPUs.
# =============================================================================

# =============================================================================
# LSF Job Options
# =============================================================================

### Specify the queue based on the GPU type you need
# Available queues based on your cluster setup:
# - gpua100: Tesla A100 PCIE 40 GB & 80 GB
# - gpuv100: Tesla V100 16 GB, 32 GB
# - gpua10: Tesla A10 PCIE 24 GB
# - gpua40: Tesla A40 48 GB with NVLink
# - gpuamd: AMD Radeon Instinct MI25 16 GB
# Note: Retired queues are not included here.

# Example: Using Tesla V100 32 GB GPUs
# Uncomment the desired queue below and its corresponding module loads.

### ADJUST THIS BASED ON WHAT GPU TO USE
#BSUB -q gpua10


### Set the job name
### ADD HERE THE NAME OF THE JOB
#BSUB -J Teitur-Training

### Request the number of CPU cores
# For GPUs with multiple GPUs per node, adjust -n accordingly
# Example: For 1 GPU with 4 CPU cores
#BSUB -n 4

### Select GPU resources
# Example: Requesting 1 GPU in exclusive process mode
# For multiple GPUs, adjust 'num' accordingly
#BSUB -gpu "num=1:mode=exclusive_process"

### Set walltime limit: hh:mm
# Maximum 24 hours for GPU queues
#BSUB -W 1:00

### Request memory per core
# Adjust based on your application's requirements
# Example: 10GB per core
#BSUB -R "rusage[mem=10GB]"

### Set email notifications (optional)
### REPLAE WITH YOUR OWN EMAIL
#BSUB -u your-EMAIL-Here

### Enable email notifications at job start and completion
#BSUB -B
#BSUB -N

### Specify output and error files
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# =============================================================================
# Environment Setup
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# =============================================================================
# Module Loading Based on Selected Queue
# =============================================================================

# Ensure that only one queue is uncommented
selected_queues=($(echo "$LSB_QUEUE"))

# Since LSF directives are processed before the script runs, we need another way
# to ensure only one queue is selected. We'll count the number of uncommented queues.

# þARFT Að BREYTA train_hpc.sh Í ÞaÐ nafn sem ÞÚ gefur Þessu
queue_count=$(grep -c '^#BSUB -q ' train_hpc.sh)

if [ "$queue_count" -ne 1 ]; then
    echo "Error: Please uncomment exactly one queue in the job script."
    exit 1
fi

# Load modules based on the selected queue
# You need to manually uncomment the queue and its corresponding module loads below

# =============================================================================
# Uncomment the module load commands corresponding to your selected queue
# =============================================================================

# For gpua100: Tesla A100 PCIE 40 GB & 80 GB
# If you selected gpua100, uncomment the following lines:
# module load cuda/12.6.2
# module load intel/2024.2.mpi
# module load mpi/5.0.5-gcc-14.2.0-binutils-2.43 

# For gpuv100: Tesla V100 16 GB, 32 GB
# If you selected gpuv100, uncomment the following lines:
# module load cuda/11.6
# module load intel/2024.2.187.mpi
# module load openmpi/5.0.3-gcc-12.4.0

# For gpua10: Tesla A10 PCIE 24 GB
# If you selected gpua10, uncomment the following lines:
module load cuda/12.6.2
module load intel/2024.2.187.mpi
module load openmpi/5.0.3-gcc-12.4.0

# For gpua40: Tesla A40 48 GB with NVLink
# If you selected gpua40, uncomment the following lines:
# module load cuda/12.6.2
# module load intel/2024.2.187.mpi
# module load openmpi/5.0.3-gcc-12.4.0

# For gpuamd: AMD Radeon Instinct MI25 16 GB
# If you selected gpuamd, uncomment the following lines:
# module load cuda/12.6.2
# module load intel/2024.2.187.mpi
# module load openmpi/5.0.3-gcc-12.4.0

# =============================================================================
# Conda Environment Setup
# =============================================================================


#You need to change this part of the script to match your conda environment
# Define variables
ENV_NAME="deeplearn"
ENV_YML="environment.yml"
REQUIREMENTS_TXT="requirements.txt"
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Miniconda
install_miniconda() {
    echo "Downloading Miniconda installer..."
    wget "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

    echo "Installing Miniconda to $MINICONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"

    echo "Removing Miniconda installer..."
    rm "$MINICONDA_INSTALLER"

    echo "Miniconda installation completed successfully."
}

# Function to initialize Conda
initialize_conda() {
    echo "Initializing Conda..."
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
}

# Function to create or update the Conda environment
setup_conda_environment() {
    if [[ ! -f "$ENV_YML" ]]; then
        echo "Error: Environment file '$ENV_YML' not found in the current directory."
        exit 1
    fi

    # Check if the environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Conda environment '${ENV_NAME}' already exists. Updating environment..."
        conda env update -n "${ENV_NAME}" -f "$ENV_YML" --prune
    else
        echo "Creating Conda environment '${ENV_NAME}'..."
        conda env create -f "$ENV_YML"
    fi
}

# Function to activate the Conda environment
activate_conda_environment() {
    echo "Activating Conda environment '${ENV_NAME}'..."
    conda activate "${ENV_NAME}"
}

# Function to install additional Python packages
install_additional_packages() {
    if [[ -f "$REQUIREMENTS_TXT" ]]; then
        echo "Installing additional Python packages from '$REQUIREMENTS_TXT'..."
        pip install --no-cache-dir -r "$REQUIREMENTS_TXT"
    else
        echo "No additional Python packages to install."
    fi
}

# =============================================================================
# Main Script Execution
# =============================================================================

echo "=== Starting HPC Job Setup ==="

# Check if Conda is already installed
if command_exists conda; then
    echo "Conda is already installed."
    CONDA_BASE=$(conda info --base)
    echo "Conda base directory: $CONDA_BASE"
else
    # Check if Miniconda directory exists and has Conda
    if [[ -d "$MINICONDA_DIR" && -x "$MINICONDA_DIR/bin/conda" ]]; then
        echo "Conda found in '$MINICONDA_DIR'. Adding to PATH..."
        export PATH="$MINICONDA_DIR/bin:$PATH"
    else
        echo "Conda not found. Proceeding to install Miniconda..."
        install_miniconda
        CONDA_BASE="$MINICONDA_DIR"
    fi
fi

# Initialize Conda if not already initialized
if ! command_exists conda; then
    initialize_conda
fi

# Source Conda to make sure it's available in the current shell
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Setup Conda environment
setup_conda_environment

# Activate the Conda environment
activate_conda_environment

# Install additional Python packages if required
install_additional_packages

# =============================================================================
# Training Execution
# =============================================================================

echo "=== Starting Training ==="

# Navigate to the directory containing train.py if not already there
# Uncomment and modify the following line if necessary
# cd /path/to/your/project

# Execute the training script

python3 model/Demo.py --data data/trainingdata.csv


echo "=== Training Completed ==="

# =============================================================================
# Final Steps
# =============================================================================

# Optionally, deactivate the Conda environment
conda deactivate

echo "=== HPC Job Script Completed ==="

