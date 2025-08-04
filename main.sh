#!/bin/bash
#SBATCH --job-name=Test # job name
#SBATCH --partition=sched_mit_sloan_gpu_r8 # partition
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=1 # cpu
# SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=64GB
#SBATCH --output=Test.out
#SBATCH --error=Test.err
#SBATCH -t 0-01:00:00 # time format is day-hours:minutes:seconds
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set


conda activate fariness_env
python preliminaries.py
