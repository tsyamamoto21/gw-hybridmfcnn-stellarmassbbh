#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -o log/mdc_main.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR

MODELDIRECTORY=data/model/model_250912/smearing_ksize5-5_channels64_relu/
OUTPUTDIRECTORY=$MODELDIRECTORY/ds1_demo/
mkdir $OUTPUTDIRECTORY

# Process background data
time apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc_main.py\
    -i data/mdc/ds1_demo/background.hdf\
    -o $OUTPUTDIRECTORY/bg.hdf\
    --modeldir $MODELDIRECTORY

# Process foreground data
time apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc_main.py\
    -i data/mdc/ds1_demo/foreground.hdf\
    -o $OUTPUTDIR/fg.hdf\
    --modeldir $MODELDIRECTORY

