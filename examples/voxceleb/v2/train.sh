###!/bin/bash

##./run.sh> exp/2pooling_constrainSamePho0.0007_diff0.00001/log 2>&1&


#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N 0.0001
#PBS -q volta_gpu
#PBS -l select=1:ncpus=10:mem=20gb:ngpus=1
#PBS -l walltime=72:00:00

cd $PBS_O_WORKDIR;

mkdir -p /scratch/ma_yi
rsync -hav /hpctmp/ma_yi/dataset_vox1 /scratch/ma_yi/

image="/app1/common/singularity-img/3.0.0/pytorch_2.0_cuda_12.0_cudnn8-devel_u22.04.sif"

singularity exec $image bash << EOF >  stdout.$PBS_JOBID 2> error.$PBS_JOBID
PYTHONPATH=$PYTHONPATH:/home/svu/ma_yi/volta_pypkg/lib/python3.8/site-packages
export PYTHONPATH

./run.sh
EOF
