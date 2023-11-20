#!/bin/bash

### General options
### -- specify queue -
#BSUB -q gpuv100
#BSUB -gpu "num=1"
### -- set the job Name -
#BSUB -J test_ejk
### -- ask for number of cores (default: 1) - 4 cores per gpu
#BSUB -n 4
### -- specify that the cores must be on the same host -
#BSUB -R "span[hosts=1]" 
### -- specify that we need x GB of memory per core/slot -
#BSUB -R "rusage[mem=6GB]" 
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -
#BSUB -M 6GB
### -- set walltime limit: hh:mm -
#BSUB -W 00:30 


### -- set the email address -# please uncomment the following line and put in your e-mail address, # if you want to receive e-mail notifications on a non-default address 
#BSUB -u s214704@student.dtu.dk 
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

# Your data is assumed to be in a directory on the cluster, adjust the path accordingly
data_directory=/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p225

# Your Python script execution command with the correct data path
python3 test.py &> /zhome/87/1/168411/Deep-learning-project/test.txt

