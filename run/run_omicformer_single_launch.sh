#!/bin/bash
#DSUB -n job_omicformer
#DSUB -A root.project.P24Z10200N0985
#DSUB -R 'cpu=24;gpu=1;mem=100000'
#DSUB -eo tmp/%J.%I.err.log
#DSUB -oo tmp/%J.%I.out.log

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
# module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh



JOB_ID=${BATCH_JOB_ID}
NODEFILE=tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
HOST=`hostname`
flock -x ${NODEFILE} -c "echo ${HOST} >> ${NODEFILE}"
MASTER_IP=`head -n 1 ${NODEFILE}`
echo $MASTER_IP
# rm $NODEFILE

torchrun --nproc_per_node=1 --master_port=30342 run_omicformer_permutation_multiinstance.py
# torchrun --nproc_per_node=2 run_omicformer_train.py
# torchrun --nproc_per_node=3 run_omicformer_prediction.py
# dsub -s run_omicformer_single_launch.sh
# nohup python run_omicformer_train.py > train.log &
# nohup CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 run_omicformer_train.py > train.log &

# nohup torchrun --nproc_per_node=1 run_omicformer_permutation_multiinstance.py > permut.log &
