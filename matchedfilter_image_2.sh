#!/bin/bash
#PBS -l elapstim_req=00:30:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log_validate.log

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --bind `pwd` dl4longcbc.sif ./generate_matched_filter_image_separately.py\
	--outdir ./data/dataset_250911/noise/\
	--ndata 100\
    --noise\
    --max_workers 2

