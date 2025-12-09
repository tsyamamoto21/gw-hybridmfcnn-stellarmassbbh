#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=00:15:00
#PBS -j o
#PBS -o log/neuralnet/test.log

#------- Program execution -------
set -x
EXPERIMENTNAME=251112_retrain
MODELNAME=smearing_ksize5-5_channels64_relu_3
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` gw_hybridmfcnn.sif ./test.py\
    --outdir=data/model/$EXPERIMENTNAME/$MODELNAME/test_cbc/\
    --modeldir=data/model/$EXPERIMENTNAME/$MODELNAME/\
    --datadir=data/dataset_250911/test/\
    --ndata=10000\
    --batchsize=100\
    --cbc

apptainer exec --nv --bind `pwd` gw_hybridmfcnn.sif ./test.py\
    --outdir=data/model/$EXPERIMENTNAME/$MODELNAME/test_noise/\
    --modeldir=data/model/$EXPERIMENTNAME/$MODELNAME/\
    --datadir=data/dataset_250911/test/\
    --ndata=10000\
    --batchsize=100\
    --noise
