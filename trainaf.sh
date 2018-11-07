for kf in 10 
do
    for db in wikihow
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --start_fold 7 \
        --use_act_rate 1 \
        --action_rate 0.04 \
        --agent_mode af \
        --epochs 5 \
        --optimizer adadelta \
        --priority 1 \
        --positive_rate 0.8 \
        --gpu_rate 0.24 \
        --actionDB $db \
        --k_fold $kf \
        --result_dir "k_fold"$kf"_test_exc"
    done
done
