#!/bin/bash
#SBATCH --job-name=flip_sculpt  # Job name
#SBATCH --output=/projects/b1094/balmer/flip_results/hd95086_f410m_bar/output.txt
#SBATCH --error=/projects/b1094/balmer/flip_results/hd95086_f410m_bar/error.txt
#SBATCH --account=b53080              # allocation/account
#SBATCH --partition=buyin             # Partition or queue name, debug, shared, wholenode
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:1                  # Specify the gpu and card
#SBATCH --mem-per-cpu=80G	      # set the amount of memory for the node
#SBATCH --time=30:00                  # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=balmer@northwestern.edu  # Email address for notifications

# get bashrc
source /home/efn6832/.bashrc

# Activate the virtual environment
mamba activate pyflip

# Run your Python script
chmod u+x run_JWST_test_refac_bar.sh
./run_JWST_test_refac_bar.sh

# Deactivate the virtual environment
conda deactivate
