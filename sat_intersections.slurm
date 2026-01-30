#!/bin/bash
#SBATCH --job-name=sat_intersect
#SBATCH --array=0-17              # 36 jobs for 36 months
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00           # Adjust based on your needs
#SBATCH --mem=8G                  # Adjust based on data size
#SBATCH -p standard
#SBATCH --output=logs/sat_%A_%a.out
#SBATCH --error=logs/sat_%A_%a.err
#SBATCH --mail-user=cw0181@uah.edu
#SBATCH --mail-type=END,FAIL

# Load any required modules
mamba_bin=/rhome/cwelch/micromamba/envs/juice_r/bin/python  # Adjust to your HPC's module system

# Run the Python script with the month index
$mamba_bin run_sat_intersections.py --data-dir ./ --month-index $SLURM_ARRAY_TASK_ID