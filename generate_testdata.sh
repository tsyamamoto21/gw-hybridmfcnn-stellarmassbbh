#!/bin/bash
#PBS -l elapstim_req=00:10:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_cbc/log_test
#PBS -t 0-4

# INJECTION_FILE=data/largesnr/ds1_test_cbc/injection.hdf
# FOREGROUND_FILE=data/largesnr/ds1_test_cbc/foreground.hdf
# NSTART=$(($PBS_SUBREQNO * 2))
# NEND=$(($PBS_SUBREQNO * 3 + 3))
GPSSTART=1284169603

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
# apptainer exec --bind /home,/mnt gw_hybridmfcnn.sif ./use_mdc_generate_matchedfilter_image.py\
# 	--outdir=data/dataset_250729/test/cbc\
# 	--foreground=$FOREGROUND_FILE\
# 	--injection=$INJECTION_FILE\
apptainer exec --bind `pwd` gw_hybridmfcnn.sif ./generate_matched_filter_image.py\
    --outdir ./data/dataset_250911/test/\
    --ndata 2000\
    --config config/dataset.ini\
    --starttime $(($GPSSTART + $PBS_SUBREQNO * 24 * 2000))\
    --offset $(($PBS_SUBREQNO * 2))\
    --parameteronly
