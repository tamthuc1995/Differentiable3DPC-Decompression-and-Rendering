export CUDA_VISIBLE_DEVICES="0"

root_path=/home/dotamthuc/Works/Projects/Compress3DGS/Experiments
path_dataset=$root_path/data/mip-nerf-360
path_output_base=$root_path/output/SVRaster/mip-nerf-360

declare -a scene_list=(
    # bicycle
    # bonsai
    # counter
    # flowers
    # garden
    # kitchen
    # room
    stump
    treehill
)

for ((i = 0; i < ${#scene_list[@]}; i++)); do
    path_source="$path_dataset"/${scene_list[i]}
    echo path_source=$path_source


    # Rate00
    path_output="$path_output_base"/${scene_list[i]}
    echo path_output=$path_output
    python -u ./svraster/train.py \
        --eval \
        --cfg_files ./svraster/cfg/mipnerf360.yaml \
        --source_path "$path_source" \
        --model_path "$path_output" \
        --res_width 1600 \
        --n_iter 30_000 

    python ./svraster/render.py $path_output --skip_train --eval_fps
    python ./svraster/render.py $path_output --skip_train 
    python ./svraster/eval.py $path_output 

done