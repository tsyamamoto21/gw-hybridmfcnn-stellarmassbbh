#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=00:30:00
#PBS -j o
#PBS -o log/neuralnet/test.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./test.py\
    --outdir=data/model/model_250912/20250917_235830/test_noise/\
    --modeldir=data/model/model_250912/20250917_235830/\
    --datadir=data/dataset_250911/test/\
    --ndata=10000\
    --batchsize=100
