# User environment PATH
# PATH="$HOME/.local/bin:$HOME/bin:$PATH"
# export PATH

# load modules
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
## module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
## end load modules
## add conda 3 into PATH
#export PATH="/home/HPCBase/tools/anaconda3/bin:$PATH"

conda activate bioenv39104  # flash_attn=1.0.4

##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

##Config nnodes node_rank master_addr
NNODES=$1
HOSTFILE=$2
HOST=`hostname`
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
echo $MASTER_IP

HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1

DISTRIBUTED_ARGS="
    --nproc_per_node 4 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_IP \
    --master_port 30342 
"
echo $DISTRIBUTED_ARGS

echo "
torchrun.sh ---------------
NNODES=${NNODES}, 
HOST=${HOST},
HOSTFILE=${HOSTFILE}, 
MASTER_IP=${MASTER_IP}, 
HOST_RANK=${HOST_RANK}, 
NODE_RANK=${NODE_RANK}
---------------------------"

##Start torchrun
# nvidia-smi 
torchrun --nproc_per_node=4 --master_port=30342 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} embed_seq.py

# dsub -s run_seq_prep_multinode_launch.sh
