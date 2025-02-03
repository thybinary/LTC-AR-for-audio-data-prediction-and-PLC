#!/bin/bash
#SBATCH --time=10:00:00          # Request 10 hours of runtime
#SBATCH --account=st-mthorogo-1    # Specify your allocation code
#SBATCH --nodes=3                # Request 16 node
#SBATCH --ntasks=16             # Request 16 task
#SBATCH --mem=128G                # Request 512 GB of memory
#SBATCH --cpus-per-task=6         # request 6 cpus per task
#SBATCH --job-name=cfc  # Specify the job name
#SBATCH --output=slurm-%j.out      # Specify the output file
#SBATCH --error=slurm-%j.err       # Specify the error file
#SBATCH --mail-user=<yjoshi03@student.ubc.ca> # Email address for job notifications
#SBATCH --mail-type=ALL          # Receive email notifications for all job events
  
 
# Load virtualenv
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
 
# Activate virtualenv
source /arc/project/st-mthorogo-1/CfC/bin/activate

# Add your commands below, e.g.:
python CfC.py

# This is the allocation code (usually defined by PI)
ALLOC=st-mthorogo-1
export ALLOC

# This is the path to project
PROJECT_PATH=/arc/project/st-mthorogo-1/CfC/yjoshi03
export PROJECT_PATH

# This is the path to scratcg
SCRATCH_PATH=/scratch/st-mthorogo-1/yjoshi03
export SCRATCH_PATH
 