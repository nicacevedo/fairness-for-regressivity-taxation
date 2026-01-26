#!/bin/bash
#SBATCH --job-name=CV_new
#SBATCH --partition=mit_normal # partition # ou_sloan_gpu #
#SBATCH --ntasks 1 # number of tasks
#SBATCH --cpus-per-task=64 # cpu
# SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# SBATCH --mem-per-cpu=64GB # memory per cpu
#SBATCH --mem=164GB
#SBATCH --output=temp/logs/CV_new_2026.out
#SBATCH --error=temp/logs/CV_new_2026.err
#SBATCH -t 0-12:00:00 # time format is day-hours:minutes:seconds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu
# SBATCH --array=1-22%10set
mkdir -p temp/logs

# Prevent hidden nested threading (critical when you do outer parallelism)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: keep LightGBM from trying GPU (if any modules set it)
export CUDA_VISIBLE_DEVICES=""

# Activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate fairness_env

python pipeline/2026_cross_validation.py



# OLD ONE
# #!/bin/bash
# #SBATCH --job-name=CV_Robust_5 # job name
# #SBATCH --partition=sched_mit_sloan_gpu_r8 # partition # ou_sloan_gpu #
# #SBATCH --ntasks 1 # number of tasks
# #SBATCH --cpus-per-task=1 # cpu
# #SBATCH --gres=gpu:1  # THIS ONE WAS THE ONE MISSING
# # SBATCH --mem-per-cpu=64GB # memory per cpu
# #SBATCH --mem=80GB
# #SBATCH --output=logs/CV_Robust_5.out
# #SBATCH --error=logs/CV_Robust_5.err
# #SBATCH -t 0-24:00:00 # time format is day-hours:minutes:seconds
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=nacevedo@mit.edu
# # SBATCH --array=1-22%10set

# conda init
# conda activate fariness_env
# python pipeline/01_train_v2.py