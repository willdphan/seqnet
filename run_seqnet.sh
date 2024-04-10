#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu
#SBATCH --output=/home/wp2240/seqNet/results/output-%j.txt # Standard output & error log, %j = id
#SBATCH --mail-type=END
#SBATCH --mail-user=willdphan@gmail.com

module purge

# Activate the Python environment
source /home/wp2240/seqNet/seqnet-env/bin/activate

# Change directory to where your Python script is located
cd /home/wp2240/seqNet

# Execute the Python script with desired arguments
python3 main.py --mode train --pooling seqnet_mix --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5"
