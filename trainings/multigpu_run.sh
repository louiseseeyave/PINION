#!/bin/bash

# set the number of nodes
# #SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00
# #SBATCH --time=02:59:00

# set name of job
#SBATCH --job-name=Train-AI4Reion

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=l.seeyave@sussex.ac.uk

# choose which partition to use
#SBATCH --partition=big
# #SBATCH --partition=devel

# store logs
#SBATCH --output logs/multigpu_train-slurm-%j.out
#SBATCH --error logs/multigpu_train-slurm-%j.err

# run the application

module purge
module load python/3.8.6

source ~/venvs/pyenv/bin/activate
echo "python environment loaded, starting training..."
python3 train.py &> logs/multigpu_train-slurm-$SLURM_JOB_ID.stdout
# python3 generate_propagation_mask.py
