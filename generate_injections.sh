#!/bin/bash
#PBS -l elapstim_req=20:00:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -o log/generate_injections.out
#PBS -e log/generate_injections.out

DATASETTYPE=1
SEEDNUMBER=512
OUTPUTDIR=data/mdc/ds1_val/
OUTPUT_INJECTION_FILE=$OUTPUTDIR/injection.hdf
OUTPUT_FOREGROUND_FILE=$OUTPUTDIR/foreground.hdf
DURATION=2592000

set -x
module load cuda/12.1.0

cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc/generate_data.py\
	-d $DATASETTYPE\
	-i $OUTPUT_INJECTION_FILE\
	-f $OUTPUT_FOREGROUND_FILE\
	-s $SEEDNUMBER\
	--duration $DURATION\
	--verbose
cp generate_injections.sh $OUTPUTDIR
