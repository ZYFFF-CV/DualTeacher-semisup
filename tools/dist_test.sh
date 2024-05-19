
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-27500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PERCENT=$1
GPUS=1 
config_dir="../code/noisedet-mmrot/configs"
config_name=""
work_dir="../workdirs/noisedet/baseline/${config_name}/${PERCENT}"
CONFIG="${config_dir}/baseline/${config_name}.py"


CHECKPOINT="${work_dir}/latest.pth"
RESULT_OUT_PATH="${work_dir}/valset_results.pkl"

CONFIG_analy="${config_dir}/soft_teacher/for_data_analy.py"

# Get evaluation results on test set
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --out $RESULT_OUT_PATH\
    ${@:4}


