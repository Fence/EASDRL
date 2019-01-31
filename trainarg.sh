CUDA_VISIBLE_DEVICES=0 python main.py \
    --domain cooking \
    --agent_mode arg \
    --gpu_fraction 0.20 \
    --k_fold 10 \
    --result_dir example_result