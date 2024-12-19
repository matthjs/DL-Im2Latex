#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=64:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=50gb
#SBATCH --job-name=im2latex

module load CUDA/12.1.1
module load IPython/8.14.0-GCCcore-12.3.0
module load typing-extensions/4.10.0-GCCcore-13.2.0

source $HOME/venvs/im2latex/bin/activate

python main.py