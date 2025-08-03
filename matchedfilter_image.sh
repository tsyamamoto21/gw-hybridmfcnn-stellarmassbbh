#!/bin/bash
#PBS -l elapstim_req=02:00:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log
#PBS -t 0-3

# INJECTION_FILE=data/largesnr/ds1_test_cbc/injection.hdf
# FOREGROUND_FILE=data/largesnr/ds1_test_cbc/foreground.hdf
# NSTART=$(($PBS_SUBREQNO * 3))
# NEND=$(($PBS_SUBREQNO * 3 + 3))

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
# apptainer exec --bind /home,/mnt dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
# 	--outdir=data/dataset_250729/test/cbc\
# 	--foreground=$FOREGROUND_FILE\
# 	--injection=$INJECTION_FILE\
apptainer exec --bind `pwd` dl4longcbc.sif ./generate_matched_filter_image.py\
	--outdir ./data/250802_dataset/train\
	--ndata 100000\
	--config config/dataset.ini\
	--starttime 1284083203\
