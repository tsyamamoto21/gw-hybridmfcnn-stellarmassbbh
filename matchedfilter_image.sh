#!/bin/bash
#PBS -l elapstim_req=01:30:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log_val
#PBS -t 0-4

# INJECTION_FILE=data/largesnr/ds1_test_cbc/injection.hdf
# FOREGROUND_FILE=data/largesnr/ds1_test_cbc/foreground.hdf
NSTART=$(($PBS_SUBREQNO * 2))
# NEND=$(($PBS_SUBREQNO * 3 + 3))
GPSSTART=1284083203

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
# apptainer exec --bind /home,/mnt dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
# 	--outdir=data/dataset_250729/test/cbc\
# 	--foreground=$FOREGROUND_FILE\
# 	--injection=$INJECTION_FILE\
apptainer exec --bind `pwd` dl4longcbc.sif ./generate_matched_filter_image.py\
	--outdir ./data/dataset_250803/validate\
	--ndata 2000\
	--config config/dataset.ini\
	--starttime $(($GPSSTART + 24 * 2000 * $PBS_SUBREQNO))\
    --offset $NSTART\
    --noiseonly

