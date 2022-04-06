#!/bin/bash

# Begin SLURM directives
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=1:00:00
#SBATCH --qos=medium

singularity exec --nv IMS-Toucan.sif python run_training_pipeline.py taco_aridialect  --gpu_id 0 
