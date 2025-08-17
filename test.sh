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
    --outdir=data/model/own_dataset_generator/20250817_183006/test_cbc/\
    --modeldir=data/model/own_dataset_generator/20250817_183006/\
    --datadir=data/dataset_250803/test/\
    --ndata=10000\
    --batchsize=500
