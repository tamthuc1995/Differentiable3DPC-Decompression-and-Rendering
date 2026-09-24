export CUDA_VISIBLE_DEVICES="0"

root_path=/home/dotamthuc/Works/Projects/Compress3DGS/Experiments
path_dataset=$root_path/data/mip-nerf-360

path_model_base=$root_path/output/SVCompress/SVRasterTiny/mip-nerf-360

path_bitstream_base=$root_path/output/SVCompress/SVRasterTinyQ2Compressed/mip-nerf-360

path_bitstream_dircolor_base=$root_path/output/SVCompress/SVRasterTinyQ2Compressed-DirColorv/mip-nerf-360

path_bitstream_noshstransform_base=$root_path/output/SVCompress/SVRasterTinyQ2Compressed-NoShsTransform/mip-nerf-360

declare -a scene_list=(
    bicycle
    # bonsai
    # counter
    # flowers
    # garden
    # kitchen
    # room
    # stump
    # treehill
)


declare -a color_coefs_stepsize=(
    0.1
    # 0.25
    # 0.5
    # 1
    # 2
    # 4
    # 8
    # 16
)

declare -a config_names=(
    configRoot
    # config00
    # config01
    # config02
    # config03
    # config04
    # config05
    # config06
)


for ((i = 0; i < ${#scene_list[@]}; i++)); do
    path_source="$path_dataset"/${scene_list[i]}
    echo path_source=$path_source

    path_model="$path_model_base"/${scene_list[i]}
    echo path_model=$path_model
    python -u ./svraster/train.py \
        --eval \
        --cfg_files ./svraster/cfg/mipnerf360_tunning.yaml \
        --source_path $path_source \
        --model_path $path_model \
        --res_width 1600 

    python ./svraster/render.py $path_model --skip_train --eval_fps
    python ./svraster/render.py $path_model --skip_train 
    python ./svraster/eval.py $path_model 


    path_bitstream=$path_bitstream_base/${scene_list[i]}
    echo path_bitstream=$path_bitstream
    for ((j = 0; j < ${#config_names[@]}; j++)); do
        echo color_coefs_stepsize=${color_coefs_stepsize[j]}
        echo suffix=${config_names[j]}

        python ./svraster/encode_bitstream.py $path_model \
            --model_bitstreams_path $path_bitstream \
            --color_coefs_stepsize ${color_coefs_stepsize[j]} \
            --suffix ${config_names[j]} \
            --debug

        python ./svraster/render_from_decode.py --model_bitstreams_path $path_bitstream --suffix ${config_names[j]} --skip_train
    done


    path_bitstream_dircolor=$path_bitstream_dircolor_base/${scene_list[i]}
    echo path_bitstream_dircolor=$path_bitstream_dircolor
    for ((j = 0; j < ${#config_names[@]}; j++)); do
        echo color_coefs_stepsize=${color_coefs_stepsize[j]}
        echo suffix=${config_names[j]}

        python ./svraster/encode_bitstream.py $path_model \
            --model_bitstreams_path $path_bitstream_dircolor \
            --color_coefs_stepsize ${color_coefs_stepsize[j]} \
            --suffix ${config_names[j]} \
            --use_directional_color_transform \
            --debug

        python ./svraster/render_from_decode.py --model_bitstreams_path $path_bitstream_dircolor --suffix ${config_names[j]} --skip_train
    done

    path_bitstream_noshstransform=$path_bitstream_noshstransform_base/${scene_list[i]}
    echo path_bitstream_noshstransform=$path_bitstream_noshstransform
    for ((j = 0; j < ${#config_names[@]}; j++)); do
        echo color_coefs_stepsize=${color_coefs_stepsize[j]}
        echo suffix=${config_names[j]}

        python ./svraster/encode_bitstream.py $path_model \
            --model_bitstreams_path $path_bitstream_noshstransform \
            --color_coefs_stepsize ${color_coefs_stepsize[j]} \
            --suffix ${config_names[j]} \
            --dont_do_shs_transform \
            --debug

        python ./svraster/render_from_decode.py --model_bitstreams_path $path_bitstream_noshstransform --suffix ${config_names[j]} --skip_train
    done

done

