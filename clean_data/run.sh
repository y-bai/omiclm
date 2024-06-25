#!/bin/bash
# dsub -A root.project.P24Z10200N0985 -R 'cpu=84;gpu=0;mem=100000' -eo %J.%I.err.log -oo %J.%I.out.log -s run.sh

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
# module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

# bash clean_data.sh /home/share/huadjyin/home/weiyilin/project/DNALLM/processed_data/DNA_training_input.tsv DNA_training_input.csv 80 sample,pos,seq,peak_value
bash clean_data.sh /home/share/huadjyin/home/weiyilin/project/DNALLM/processed_data/DNA_test_input.tsv DNA_test_input.csv 80 sample,pos,seq,peak_value