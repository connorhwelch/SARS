#!/bin/bash
#SBATCH --job-name=env_build
#SBATCH --output=env_build_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Path to Python in existing environment
mamba_bin=/nas/rhome/cwelch/.local/bin/micromamba
# Create or update environment
$mamba_bin create -f juice_r.yml -y
