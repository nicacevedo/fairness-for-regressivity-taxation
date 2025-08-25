#!/bin/bash
#SBATCH --job-name=CV_Test0 # job name
#SBATCH --partition=sched_mit_sloan_gpu_r8 # partition
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=1 # cpu
#SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=80GB
#SBATCH --output=logs/CV_Test0.out
#SBATCH --error=logs/CV_Test0.err
#SBATCH -t 0-24:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set

conda init
conda activate fariness_env
python pipeline/01_train_v2.py

