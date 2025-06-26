#!/bin/bash
#PBS -l elapstim_req=00:30:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -o mfimage.out
#PBS -e mfimage.out

INJECTION_FILE=data/mdc/ds1/injection.hdf
FOREGROUND_FILE=data/mdc/ds1/foreground.hdf

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
apptainer exec --bind `pwd` dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
	--outdir=data/dataset_250625/train/cbc\
	--foreground=$FOREGROUND_FILE\
	--injection=$INJECTION_FILE\

