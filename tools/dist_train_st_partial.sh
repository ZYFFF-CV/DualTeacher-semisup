set -x

PERCENT=$1
GPUS=8 
PORT=${PORT:-29500}
config_name=".."
work_dir="../workdirs/noisedet/${config_name}/${PERCENT}"
config_dir="../code/noisedet-mmrot/configs"
config="${config_dir}/soft_teacher/${config_name}.py"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $config \
        --launcher pytorch \
        --cfg-options percent=${PERCENT} ${@:5} \
        --work-dir $work_dir 
        #fold=${FOLD} 
        