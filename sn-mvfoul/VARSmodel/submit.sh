#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J views
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
# BSUB -R "select[gpu32gb]"
#BSUB -B
#BSUB -N
#BSUB -u s194572@dtu.dk
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

. /work3/s194572/miniconda3/etc/profile.d/conda.sh
conda activate VDTU

python3 train_decoder.py