set -x

PERCENT=$1
GPUS=4 
PORT=${PORT:-29500}

config_name=".."


work_dir="../workdirs/noisedet/${config_name}/${PERCENT}"
config_dir="../code/noisedet-mmrot/configs"

config="${config_dir}/soft_teacher/${config_name}.py"
ckpt="${work_dir}/latest.pth"

eval_work_dir="${work_dir}/eval"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $config \
        --launcher pytorch \
        --cfg-options percent=${PERCENT} ${@:5} \
        --work-dir $work_dir 

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $config \
    $ckpt \
    --launcher pytorch \
    --out "${work_dir}/valset_results.pkl"\
    ${@:4}