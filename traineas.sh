for gn in 7 9 5
do
    for fn in 64 128 32
    do
        for dp in 0.25 0.5 0.75 
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
            --actionDB cooking \
            --dqn_mode cnn2 \
            --agent_mode eas \
            --gpu_rate 0.20 \
            --k_fold 5 \
            --gram_num $gn \
            --filter_num $fn \
            --dropout $dp \
            --result_dir "bn0_gn"$gn"_dp"$dp"flatten_fn"$fn
        done
    done
done
