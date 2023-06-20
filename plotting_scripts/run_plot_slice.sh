#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:15:00

# set name of job
#SBATCH --job-name=Compare-AI4Reion

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# choose which partition to use
#SBATCH --partition=devel

# store logs
#SBATCH --output logs/slurm-%j.out
#SBATCH --error logs/slurm-%j.err

# run the application

module purge
module load python/3.8.6

source ~/venvs/pyenv/bin/activate
echo "python environment loaded, starting training..."
python3 plot_slice.py &> logs/slice-slurm-$SLURM_JOB_ID.stdout
