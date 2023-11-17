#!/bin/bash

#SBATCH --job-name=test_ejk
#SBATCH --partition=gpuv100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00


# Activate the virtual environment
source /zhome/87/1/168411/dl-venv/bin/activate

# Change to the directory containing the Python script
cd /zhome/87/1/168411/Deep-learning-project

# Run the Python script
python test2.py