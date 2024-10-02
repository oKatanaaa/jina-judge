config=$1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup jinaj-train $config > logs/$config.log 2>&1 &