#!/bin/bash


singularity exec --nv --nvccli --bind /run/shm IMS-Toucan.sif python run_training_pipeline.py taco_aridialect --config traininf/paramsspo.ini
