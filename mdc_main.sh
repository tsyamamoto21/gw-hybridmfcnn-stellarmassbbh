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
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./mdc_main.py\
    -i data/mdc/ds1/foreground.hdf\
    -o data/mdc/testoutput.hdf

