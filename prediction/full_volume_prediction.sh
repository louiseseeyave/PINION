#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:30:00

# set name of job
#SBATCH --job-name=Pred-AI4Reion

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=l.seeyave@sussex.ac.uk

# choose which partition to use
#SBATCH --partition=devel

# store logs
#SBATCH --output logs/slurm-%j.stdout
#SBATCH --error logs/slurm-%j.stdout

#SBATCH --array=0-1000%100

# run the application

echo STARTING AT `date`
echo "Setting up enviroment to use MPI..."

# Set how you want to split the cube.
CUBESIZE=30
NDOMAINS=1000

# Here we load the different (you will have to change this, this is CSCS specific)
# module load daint-gpu
# export CRAY_CUDA_MPS=1

echo "Loading modules..."
module purge
module load python/3.8.6

echo "Activating pyenv..."
# Activate your python environnement here
source ~/venvs/pyenv/bin/activate

echo "PYTHONPATH..."
# The path to the root the project to load the python modules
# export PYTHONPATH="/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

echo "Defining task number..."
# Define the task number
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
echo "Task ${TASK_ID}/${NDOMAINS}"

echo "Executing script..."
# Execute the script for subvolume $TASK_ID
srun python subvolume_prediction.py $CUBESIZE $TASK_ID # &> logs/slurm-${SLURM_ARRAY_JOB_ID}_${TASK_ID}.stdout

echo FINISHED AT `date`
