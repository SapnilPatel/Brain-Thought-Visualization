#!/bin/bash
#SBATCH -J threeminds_atmpt_1               # Job name
#SBATCH --nodes=1             # Total # of nodes
#SBATCH --ntasks-per-node=1  # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gres=gpu:2          # Number of GPUs per node
#SBATCH --time=1-00:00:00     # Total run time limit (hh:mm:ss)
#SBATCH -o job.o%j            # Name of stdout output file
#SBATCH -e job.e%j            # Name of stderr error file
#SBATCH -p gpusinglenode      # Queue (partition) name
#SBATCH --reservation=three-minds        # Reservation Name

# Manage processing environment, load compilers, and applications.

. /opt/ohpc/admin/lmod/lmod/init/sh # Do not change or remove this line

# Load Packages through Module load
module load python/conda-python/3.7
module load cuda/11.2

cd $SLURM_SUBMIT_DIR

# Give Job execution command
source /scratch/three-minds/workstation/pythonenvs/env2-python37/env2-lbtv/bin/activate

# Set WANDB_MODE to offline for not visualizing results
export WANDB_MODE=offline

nsys profile --trace=osrt,cuda,nvtx python3 /scratch/three-minds/workstation/lab/LatestBrain-TV/neel_tmp/demo.py > output.txt
