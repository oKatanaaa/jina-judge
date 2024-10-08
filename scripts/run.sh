config=$1

# Set this env var, otherwise training OOMs on RTX3090
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup jinaj-train $config > logs/$config.log 2>&1 &