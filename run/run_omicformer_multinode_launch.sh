#!/bin/bash
#DSUB -n job_omicformer
#DSUB -A root.project.P24Z10200N0985
#DSUB -R 'cpu=20;gpu=4;mem=120000'
#DSUB -N 4
#DSUB -eo tmp/%J.%I.err.log
#DSUB -oo tmp/%J.%I.out.log

## Set scripts
RANK_SCRIPT="run_omicformer_multinode_script.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/weiyilin/project/DNALLM/omiclm/run"

## Set NNODES
NNODES=4

## Create nodefile
JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
#cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/tmp/${JOB_ID}.nodefile
# cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > tmp/CCS_ALLOC_FILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}

# dsub -s run_omicformer_multinode_launch.sh

# nohup torchrun --nproc_per_node 1 run_omicformer_train.py > train.log &