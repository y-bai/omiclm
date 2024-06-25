#!/bin/bash
#DSUB -n job_prep
#DSUB -A root.project.P24Z10200N0985
#DSUB -R 'cpu=50;gpu=0;mem=100000'
#DSUB -eo %J.%I.err.log
#DSUB -oo %J.%I.out.log

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
# module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
HOST=`hostname`
flock -x ${NODEFILE} -c "echo ${HOST} >> ${NODEFILE}"
MASTER_IP=`head -n 1 ${NODEFILE}`
echo $MASTER_IP
rm $NODEFILE

# 1. Tokenize seq data 
# python tokenize_seq.py

# 2. Convert tokenized HF seq dataset into json file, which is used for training with huge data
python hf_dataset_to_json.py

# 3. embed seq data (optional)
# NOTE: this would generate too many and large embedding files because there is (501, 256) for each seq with 501 length.
# So, we embed the seqs on the fly instead of calculating embeddings in advance like scRNA being done.
# torchrun --nproc_per_node 4 embed_seq.py

# 4. embed scRNA data
# torchrun --nproc_per_node 1 embed_scrna.py

# dsub -s run_seq_prep.sh

# nohup torchrun --nproc_per_node 1 embed_scrna.py > train.log &
