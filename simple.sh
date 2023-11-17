#!/bin/sh
### General options
### -- specify queue -
#BSUB -q gpuv100
#BSUB -gpu "num=1"
### -- set the job Name -
#BSUB -J test2
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
# here follow the commands you want to execute
source source dl-venv/bin/activate
python3 test2.py &> output.txt