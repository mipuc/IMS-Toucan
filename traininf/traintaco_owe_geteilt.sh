#!/bin/bash


singularity exec --nv --nvccli --bind /run/shm IMS-Toucan.sif python run_training_pipeline.py --config traininf/taco_owe_geteilt.ini
