#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -o log/neuralnet/train3.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./train.py --dirname smearingkernel2_ksize5-5_channels64_relu
