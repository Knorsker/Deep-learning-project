#!/bin/bash

#SBATCH --job-name=test_ejk
#SBATCH --partition=gpuv100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00


### -- set the email address -# please uncomment the following line and put in your e-mail address, # if you want to receive e-mail notifications on a non-default address 
#BSUB -u s20s214704@student.dtu.dk 
### -- send notification at start -
#BSUB -B 
### -- send notification at completion -
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -
### -- -o and -e mean append, -oo and -eo mean overwrite -
#BSUB -o %J.out
#BSUB -e %J.err


# Activate the virtual environment
source /zhome/87/1/168411/dl-venv/bin/activate

# Change to the directory containing the Python script
cd /zhome/87/1/168411/Deep-learning-project

# Run the Python script
python test2.py