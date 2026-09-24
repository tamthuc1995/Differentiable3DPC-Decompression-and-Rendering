export CUDA_VISIBLE_DEVICES="0"

root_path=/home/dotamthuc/Works/Projects/Compress3DGS/Experiments
path_dataset=$root_path/data/nerf_synthetic

path_model_base=$root_path/output/SVCompress_tuningV03/SVRasterSmall/nerf_synthetic

path_bitstream_base=$root_path/output/SVCompress_tuningV03/SVRasterSmallCompressed/nerf_synthetic

path_bitstream_dircolor_base=$root_path/output/SVCompress_tuningV03/SVRasterSmallCompressed-DirColorv/nerf_synthetic

path_bitstream_noshstransform_base=$root_path/output/SVCompress_tuningV03/SVRasterSmallCompressed-NoShsTransform/nerf_synthetic

declare -a scene_list=(
    chair
    drums
    ficus
    hotdog
    lego
    materials
    mic
    ship
)


declare -a color_coefs_stepsize=(
    0.25
    0.5
    1
    2
    4
    8
    16
)

declare -a config_names=(
    config00
    config01
    config02
    config03
    config04
    config05
    config06
)


for ((i = 0; i < ${#scene_list[@]}; i++)); do
    path_source="$path_dataset"/${scene_list[i]}
    echo path_source=$path_source

    path_model="$path_model_base"/${scene_list[i]}
    echo path_model=$path_model
    python -u ./svraster/train.py \
        --eval \
        --cfg_files ./svraster/cfg/synthetic_nerf_small.yaml \
        --source_path $path_source \
        --model_path $path_model 

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
            --geo_params_stepsize 2.0 \
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
            --geo_params_stepsize 2.0 \
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
            --geo_params_stepsize 2.0 \
            --color_coefs_stepsize ${color_coefs_stepsize[j]} \
            --suffix ${config_names[j]} \
            --dont_do_shs_transform \
            --debug

        python ./svraster/render_from_decode.py --model_bitstreams_path $path_bitstream_noshstransform --suffix ${config_names[j]} --skip_train
    done

done

