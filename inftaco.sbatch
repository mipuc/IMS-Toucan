#!/bin/bash

# Begin SLURM directives
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

singularity exec --nv IMS-Toucan.sif python run_text_to_file_reader.py
