#!/bin/bash
#PBS -l elapstim_req=00:30:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -o log/generate_injections.out
#PBS -e log/generate_injections.out

DATASETTYPE=4
SEEDNUMBER=2514409456
OUTPUTDIR=data/mdc/ds4_largesnr/
OUTPUT_INJECTION_FILE=$OUTPUTDIR/injection.hdf
OUTPUT_FOREGROUND_FILE=$OUTPUTDIR/foreground.hdf
OUTPUT_BACKGROUND_FILE=$OUTPUTDIR/background.hdf
DURATION=25920

set -x
#module load cuda/12.1.0

cd $PBS_O_WORKDIR
apptainer exec --nv --bind /home,/mnt dl4longcbc.sif ./mdc/generate_data.py\
	-d $DATASETTYPE\
    -i $OUTPUT_INJECTION_FILE\
    -f $OUTPUT_FOREGROUND_FILE\
    -b $OUTPUT_BACKGROUND_FILE\
	-s $SEEDNUMBER\
	--duration $DURATION\
	--verbose
cp generate_injections.sh $OUTPUTDIR
