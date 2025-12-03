#!/bin/bash
#PBS -l elapstim_req=01:00:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log_train.log

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./torch_generate_mfimage.py\
	--outdir ./data/dataset_250911/train/\
	--ndata 11250\
    --noise\
    --offset 1250