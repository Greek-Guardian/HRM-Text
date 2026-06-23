# 方式 A：只跑 MMLU 

cd /root/HRM/HRM-Text 

python -m evaluation.main \
    ckpt_path="/root/HRM/HRM-Text/checkpoints/Sampled HLM-torch/HierarchicalReasoningModel armored-vulture" \
    config="evaluation/config/hrm_benchmarking.yaml" \
    run_only="[MMLU]"

# 方式 B：跑全部基准（含 MMLU） 

python -m evaluation.main \
    ckpt_path="<your_ckpt_path>" \
    config="evaluation/config/hrm_benchmarking.yaml"

# 方式 C：CLI 覆盖参数  

python -m evaluation.main \
    ckpt_path="<your_ckpt>" \
    config="evaluation/config/hrm_benchmarking.yaml" \
    run_only="[MMLU]" \
    generation_config.batch_size=8 \
    generation_config.temperature=0.0 