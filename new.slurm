#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=MiniProject
#SBATCH --time=0-08:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --output=output.txt
#SBATCH --error=output.err
#SBATCH --mail-user=mariulie@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Miniconda3/24.7.1-0

# Correct activation using conda
eval "$(conda shell.bash hook)"
conda activate miniProj

# Debug check
echo "Using Python at: $(which python)"
python --version

# Run the script
python /cluster/home/mariulie/Documents/TDT4265/3D/UNet-segmentation-MRI/main.py

