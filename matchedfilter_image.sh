#!/bin/bash
#PBS -l elapstim_req=24:00:00
#PBS -l cpunum_job=48
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_cbc/
#PBS -t 0-3

INJECTION_FILE=data/mdc/ds1/injection.hdf
FOREGROUND_FILE=data/mdc/ds1/foreground.hdf
i=$PBS_ARRAY_INDEX
NSTART=$((12 + 29 * i))
NEND=$((12 + 29 * (i + 1)))

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
apptainer exec --bind `pwd` dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
	--outdir=data/dataset_250625/train/cbc\
	--foreground=$FOREGROUND_FILE\
	--injection=$INJECTION_FILE\
    --nstart=$NSTART\
    --nend=$NEND

