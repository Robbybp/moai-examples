#!/bin/bash
#
#SBATCH --job-name=runtime-sweep
#SBATCH --time=05:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output="output/slurm-%A.out"
#
#This recomend to minimize clashes with other users
#SBATCH --partition=shared-gpu
#SBATCH --qos=normal
#
module load miniconda3
source activate torch
julia --project=. neurips2025-scaleopt/analyze-runtime-slurm.jl
