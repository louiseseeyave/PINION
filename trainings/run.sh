#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6:00:00

# set name of job
#SBATCH --job-name=Train-AI4Reion

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=l.seeyave@sussex.ac.uk

# choose which partition to use
#SBATCH --partition=small

# store logs
#SBATCH --output logs/slurm-%j.out
#SBATCH --error logs/slurm-%j.err

# run the application

module purge
module load python/3.8.6

source ~/venvs/pyenv/bin/activate
python3 training-PFD-NP.py &> logs/slurm-$SLURM_JOB_ID.stdout
