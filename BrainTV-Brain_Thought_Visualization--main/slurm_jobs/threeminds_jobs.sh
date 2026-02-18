#!/bin/bash
#SBATCH -J threeminds_atmpt_3               # Job name
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
source /scratch/three-minds/workstation/pythonenvs/env1-python37/env1-lbtv/bin/activate

# Set WANDB_MODE to offline for not visualizing results
export WANDB_MODE=offline

# normal execution: python /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output.txt
python /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output.txt
# single node multi-gpu execution: python -m torch.distributed.launch --nproc_per_node=2 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output.txt
# python -m torch.distributed.launch --nproc_per_node=1 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output.txt
# python -m torch.distributed.launch --nproc-per-node=2 --nnodes=2 --node-rank=0 --master-addr="172.10.3.75" --master-port=29500 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output_nr0.txt
# python -m torch.distributed.launch --nproc-per-node=2 --nnodes=2 --node-rank=1 --master-addr="172.10.3.75" --master-port=29500 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output_nr1.txt
# python -m torch.distributed.launch --nproc-per-node=2 --nnodes=2 --node-rank=2 --master-addr="172.20.3.75" --master-port=29500 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output_nr2.txt
# python -m torch.distributed.launch --nproc-per-node=2 --nnodes=2 --node-rank=3 --master-addr="172.20.3.75" --master-port=29500 /scratch/three-minds/workstation/lab/LatestBrain-TV/code/stageA1_mbm_pretrain.py > output_nr3.txt