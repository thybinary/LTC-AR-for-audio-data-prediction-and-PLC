#!/bin/bash
#SBATCH --time=00:59:00          # Request 1 hour of runtime
#SBATCH --account=st-mthorogo-1-gpu    # Specify your allocation code
#SBATCH --nodes=1                # Request 6 node
#SBATCH --ntasks-per-node=1            # Request 6 task
#SBATCH --mem=64G                # Request 128 GB of memory
#SBATCH --cpus-per-task=12	# request 6 cpus per task
#SBATCH --gpus-per-node=4
#SBATCH --job-name=cfc  # Specify the job name
#SBATCH --output=slurm-%j.out      # Specify the output file
#SBATCH --error=slurm-%j.err       # Specify the error file
#SBATCH --mail-user=<yjoshi03@student.ubc.ca> # Email address for job notifications
#SBATCH --mail-type=ALL          # Receive email notifications for all job events

module load gcc
module load cuda
 
# Load virtualenv
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
 
# Activate virtualenv
source /arc/project/st-mthorogo-1/CfC/bin/activate

# Add your commands below, e.g.:
torchrun CfC_training.py

# This is the allocation code (usually defined by PI)
ALLOC=st-mthorogo-1
export ALLOC

# This is the path to project
PROJECT_PATH=/arc/project/st-mthorogo-1/CfC/yjoshi03
export PROJECT_PATH

# This is the path to scratcg
SCRATCH_PATH=/scratch/st-mthorogo-1/yjoshi03
export SCRATCH_PATH

#gpu info
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355  # Use an open port

 