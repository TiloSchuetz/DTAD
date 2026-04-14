#!/bin/bash

#SBATCH --job-name=prepr_DTAD
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=24:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate DTAD

python ddim_inversion.py --data_root '/ceph/tischuet/replication_data' --dataset ForenSynths --denoising_root '/ceph/tischuet/DTAD_ouput_root'