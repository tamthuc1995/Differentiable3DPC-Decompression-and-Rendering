export CUDA_VISIBLE_DEVICES="0"

root_path=/home/dotamthuc/Works/Projects/Compress3DGS/Experiments
path_dataset=$root_path/data/mip-nerf-360
path_model_base=$root_path/output/SVRaster/mip-nerf-360
path_bitstream_base=$root_path/output/SVRasterCompressed-DirColorv/mip-nerf-360

declare -a scene_list=(
    # bicycle
    # bonsai
    # counter
    # flowers
    # garden
    # kitchen
    # room
    stump
    # treehill
)


declare -a color_coefs_stepsize=(
    # 0.25
    0.5
    # 1
    # 2
    # 4
    # 8
    # 16
)

declare -a config_names=(
    # config00
    config01
    # config02
    # config03
    # config04
    # config05
    # config06
)


for ((i = 0; i < ${#scene_list[@]}; i++)); do
    path_model=$path_model_base/${scene_list[i]}
    echo path_model=$path_model

    path_output=$path_bitstream_base/${scene_list[i]}
    echo path_output=$path_output

    for ((j = 0; j < ${#config_names[@]}; j++)); do
        echo color_coefs_stepsize=${color_coefs_stepsize[j]}
        echo suffix=${config_names[j]}

        python ./svraster/encode_bitstream.py $path_model \
            --model_bitstreams_path $path_output \
            --color_coefs_stepsize ${color_coefs_stepsize[j]} \
            --suffix ${config_names[j]} \
            --use_directional_color_transform \
            --debug

        # python ./svraster/render_from_decode.py --model_bitstreams_path $path_output --suffix ${config_names[j]} --skip_train

    done
done

