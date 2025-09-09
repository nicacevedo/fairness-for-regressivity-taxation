#!/bin/bash
#SBATCH --job-name=Res_1 # job name
#SBATCH --partition=sched_mit_sloan_gpu_r8 # partition # ou_sloan_gpu #
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=1 # cpu
#SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=80GB
#SBATCH --output=logs/Res_1.out
#SBATCH --error=logs/Res_1.err
#SBATCH -t 0-24:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set

conda init
conda activate fariness_env
# python pipeline/01_main_result_computer_with_sizes.py
python pipeline/01_main_result_computer.py