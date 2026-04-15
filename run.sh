#!/bin/bash
#JSUB -q gpu
#JSUB -m 4TeslaV100
#JSUB -gpgpu 1
#JSUB -J mar_base_eval
#JSUB -e /home/24031111031/mar/job_output/err.%J.mar_eval.txt
#JSUB -o /home/24031111031/mar/job_output/out.%J.mar_eval.txt
#JSUB -cwd /home/24031111031/mar

set -x

mkdir -p /home/24031111031/mar/job_output
cd /home/24031111031/mar

export CUDA_HOME=/apps/software/cuda/11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/24031111031/.conda/envs/mar_copy/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "PWD=$(pwd)"
echo "Start time: $(date)"
echo "Python: $(/home/24031111031/.conda/envs/mar_copy/bin/python -V)"

/home/24031111031/.conda/envs/mar_copy/bin/python main_mar.py \
    --evaluate \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --resume /home/24031111031/pretrained_models/mar/mar_base \
    --vae_path /home/24031111031/pretrained_models/vae/kl16.ckpt \
    --num_images 1000 \
    --eval_bsz 64 \
    --num_iter 256 \
    --num_sampling_steps 100 \
    --cfg 2.9 \
    --cfg_schedule linear \
    --temperature 1.0 \
    --data_path /home/24031111031/mar \
    --dist_url env://

echo "End time: $(date)"