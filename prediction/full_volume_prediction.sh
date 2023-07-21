#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
# #SBATCH --time=00:45:00
#SBATCH --time=36:00:00

# set name of job
#SBATCH --job-name=Pred-AI4Reion

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=l.seeyave@sussex.ac.uk

# choose which partition to use
# #SBATCH --partition=devel
#SBATCH --partition=small

# store logs
#SBATCH --output logs/slurm-%j.stdout
#SBATCH --error logs/slurm-%j.stdout

# #SBATCH --array=0-1000%100
#SBATCH --array=0-20

# run the application

echo STARTING AT `date`
echo "Setting up enviroment to use MPI..."

# Set how you want to split the cube.
# CUBESIZE=50 # number of cells along an axis
# NDOMAINS=125 # basically (250/50)^3
CUBESIZE=25 # number of cells along an axis
NDOMAINS=1000 # basically (250/50)^3

# Here we load the different (you will have to change this, this is CSCS specific)
# module load daint-gpu
# export CRAY_CUDA_MPS=1

echo "Loading modules..."
module purge
module load python/3.8.6

echo "Activating pyenv..."
# Activate your python environnement here
source ~/venvs/pyenv/bin/activate

# echo "PYTHONPATH..."
# The path to the root the project to load the python modules
# export PYTHONPATH="/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

# echo "Defining task number..."
# Define the task number
# TASK_ID=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
# echo "Task ${TASK_ID}/${NDOMAINS}"

NBATCH=({0..50..1})
# NBATCH=({0..7..1})
# NBATCH=({49..50..1})
echo ${NBATCH[@]}

echo "Executing script..."
# Execute the script for subvolume $TASK_ID
# srun python subvolume_prediction.py $CUBESIZE $TASK_ID # &> logs/slurm-${SLURM_ARRAY_JOB_ID}_${TASK_ID}.stdout

for ii in ${NBATCH[@]}
  do
    # BATCH=$((${SLURM_ARRAY_TASK_ID} * 7))
    # TASK_ID=$((${BATCH} + ${ii} - 1))
    # echo "Task ID: ${TASK_ID}"
    BATCH=$((${SLURM_ARRAY_TASK_ID} * 50))  
    TASK_ID=$((${BATCH} + ${ii} - 1))
    # TASK_ID=${SLURM_ARRAY_TASK_ID}
    echo "Task ID: ${TASK_ID}"
    # python3 subvolume_prediction.py $CUBESIZE $TASK_ID
    python3 subvolume_predict.py $CUBESIZE $TASK_ID
done

# TASK_ID=${SLURM_ARRAY_TASK_ID}
# echo "Task ID: ${TASK_ID}"
# python3 subvolume_prediction.py $CUBESIZE $TASK_ID
# python3 subvolume_predict.py $CUBESIZE $TASK_ID
    
echo FINISHED AT `date`

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
