#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=36:00:00
# #SBATCH --time=02:59:00

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
# #SBATCH --partition=devel

# store logs
#SBATCH --output logs/slurm-%j.out
#SBATCH --error logs/slurm-%j.err

# run the application

module purge
module load python/3.8.6

source ~/venvs/pyenv/bin/activate
echo "python environment loaded, starting training..."
# python3 test.py &> logs/slurm-$SLURM_JOB_ID.stdout
# python3 training-PFD-NP.py &> logs/slurm-$SLURM_JOB_ID.stdout
python3 train.py &> logs/slurm-$SLURM_JOB_ID.stdout
# python3 compare.py &> logs/slurm-$SLURM_JOB_ID.stdout
# python3 generate_propagation_mask.py
