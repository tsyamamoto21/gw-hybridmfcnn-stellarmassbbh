#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=04:00:00
#PBS -j o
#PBS -o log/mdc_main.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR

MODELDIRECTORY=data/model/model_250912/smearing_ksize5-5_channels64_relu/
MDCDATATYPE=ds1
OUTPUTDIRECTORY=$MODELDIRECTORY/$MDCDATATYPE/
mkdir $OUTPUTDIRECTORY

# Process background data
time apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc_main.py\
   -i data/mdc/$MDCDATATYPE/background.hdf\
   -o $OUTPUTDIRECTORY/bg.hdf\
   --modeldir $MODELDIRECTORY

# Process foreground data
time apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc_main.py\
   -i data/mdc/$MDCDATATYPE/foreground.hdf\
   -o $OUTPUTDIRECTORY/fg.hdf\
   --modeldir $MODELDIRECTORY

# Evaluate the results
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc/evaluate.py\
    --injection-file data/mdc/$MDCDATATYPE/injection.hdf\
    --foreground-events $OUTPUTDIRECTORY/fg.hdf\
    --foreground-files data/mdc/$MDCDATATYPE/foreground.hdf\
    --background-events $OUTPUTDIRECTORY/bg.hdf\
    --output-file $OUTPUTDIRECTORY/eval.hdf\
