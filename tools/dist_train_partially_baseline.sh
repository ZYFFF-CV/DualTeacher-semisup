set -x

PERCENT=$1
GPUS=8 
PORT=${PORT:-29501}

config_name=".."



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


work_dir="../workdirs/noisedet/baseline/${config_name}/${PERCENT}-2"
config_dir="../code/noisedet-mmrot/configs"

config="${config_dir}/baseline/${config_name}.py"

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $config \
        --launcher pytorch \
        --cfg-options percent=${PERCENT} ${@:5} \
        --work-dir $work_dir \
        --seed 12345