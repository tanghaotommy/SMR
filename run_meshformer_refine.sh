#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=PoseMeshFormer

#SBATCH --output=/checkpoint/haotang/slurm_logs/PoseMeshFormer/%x_%j.out

#SBATCH --error=/checkpoint/haotang/slurm_logs/PoseMeshFormer/%x_%j.err

# pixar, learnaccel, learnfair, scavenge

#SBATCH --partition=eht

#SBATCH --nodes=1

#SBATCH --mem=120G

#SBATCH --gres=gpu:8

#SBATCH --cpus-per-task=10

#SBATCH --time=23:59:00

#SBATCH --mail-user=haotang@fb.com

#SBATCH --mail-type=begin,end,fail,requeue # mail once the job finishes

#SBATCH --signal=USR1@300

#SBATCH --open-mode=append

#SBATCH --comment="PoseMeshFormer"

#SBATCH --requeue

source /private/home/haotang/.bashrc
# module load anaconda3
conda activate smr
export CUDA_HOME=/public/apps/cuda/11.3/


cd /checkpoint/haotang/dev/SMR/

export DATA_ROOT=/checkpoint/haotang/data/co3d_v2_normalized_128
# export DATA_ROOT=/checkpoint/haotang/data/shapenet_multiview_10
# export DATA_ROOT=/checkpoint/haotang/data/CUB_Data
export CUDA_LAUNCH_BLOCKING=1.
srun  python train_multiview.py --batchSize 12 \
                    --n_gpu_per_node 8 \
                    --lr 0.0001 \
                    --niter 9000 \
                    --dataset co3d_seq \
                    --dataroot $DATA_ROOT \
                    --template_path ./template/sphere.obj \
                    --norm_layer InstanceNorm2d \
                    --outf ./log/MeshPoseFormer/bicycle_pose_ddp_instance_norm_original_pose_only \
                    --azi_scope 360 \
                    --elev_range '0~30' \
                    --dist_range '2~6' \
                    --lambda_gan 0.0001 \
                    --lambda_reg 0.1 \
                    --lambda_data 1.0 \
                    --lambda_ic 0.1 \
                    --lambda_lc 0.001 \
                    --visualization_epoch 10 \
                    --model MeshFormer \
                    --refine_pose \
                    --ddp 
                    # --pretrained_path ./log/MeshPoseFormer/pretrained_bs32 \
                    # --outf ./log/MeshPoseFormer/toyplane_pose_refine_pretrained_bs32_ddp \
