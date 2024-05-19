
PERCENT=10
config_name=""

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
work_dir="../workdirs/noisedet/baseline/${config_name}/10"

ckpt_path_in_name="latest"
ckpt_path_out_name="published_${ckpt_path_in_name}"

python  $(dirname "$0")/model_converters/publish_model.py \
    "${work_dir}/${ckpt_path_in_name}.pth" \
    "${work_dir}/${ckpt_path_out_name}.pth"\