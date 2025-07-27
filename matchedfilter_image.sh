#!/bin/bash
#PBS -l elapstim_req=02:00:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log
#PBS -t 0-3

INJECTION_FILE=data/largesnr/ds1_val/injection.hdf
FOREGROUND_FILE=data/largesnr/ds1_val/foreground.hdf
NSTART=$(($PBS_SUBREQNO * 3))
NEND=$(($PBS_SUBREQNO * 3 + 3))

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
apptainer exec --bind `pwd` dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
	--outdir=data/dataset_250716/validate/noise\
	--foreground=$FOREGROUND_FILE\
	--injection=$INJECTION_FILE\
    --nstart=$NSTART\
    --nend=$NEND\
    --offevent

