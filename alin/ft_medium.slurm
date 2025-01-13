#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 12:00:00
#SBATCH -p gpu-devel
#SBATCH -J FT
#SBATCH --mem=0
#SBATCH --gpu-bind=closest

source ~/.bashrc
micromamba activate janus
module load amd-modules Rust

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
#export LD_LIBRARY_PATH=~/playground/micromamba/micromamba/envs/janus-3/lib/python3.11/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH

OMP_NUM_THREADS=8  mace_run_train --config  ft_medium.yml
