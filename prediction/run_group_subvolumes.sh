#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:15:00

# set name of job
#SBATCH --job-name=Group-AI4EoR

# set number of GPUs (use CPU for this)
# SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
# #SBATCH --mail-user=l.seeyave@sussex.ac.uk

# choose which partition to use
#SBATCH --partition=devel

# store logs
#SBATCH --output logs/grp-slurm-%j.stdout
#SBATCH --error logs/grp-slurm-%j.stdout

# run the application

echo STARTING AT `date`
echo "Setting up enviroment to use MPI..."

# Set how you want to split the cube.
CUBESIZE=50
FULLCUBESIZE=250

# Here we load the different (you will have to change this, this is CSCS specific)
# module load daint-gpu
# export CRAY_CUDA_MPS=1

echo "Loading modules..."
module purge
module load python/3.8.6

echo "Activating pyenv..."
# Activate your python environnement here
source ~/venvs/pyenv/bin/activate

echo "Executing script..."
# Execute the script for subvolume $TASK_ID
python3 group_subvolumes.py $CUBESIZE $FULLCUBESIZE # &> logs/slurm-${SLURM_ARRAY_JOB_ID}_${TASK_ID}.stdout

echo FINISHED AT `date`
