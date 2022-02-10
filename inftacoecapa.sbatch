#!/bin/bash

# Begin SLURM directives
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

singularity exec --nv IMS-Toucan.sif python run_text_to_file_reader.py --spklist speaker.list --alpha "0.0 0.2 0.4 0.5 0.6 0.8 1.0" --speaker_embedding_type "ecapa"
