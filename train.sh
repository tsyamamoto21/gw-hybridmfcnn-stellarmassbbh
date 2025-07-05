#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=12:00:00
#PBS -j o
#PBS -o log/neuralnet/train.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
./train.py config/config_train.yaml
