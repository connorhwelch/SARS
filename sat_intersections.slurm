#!/bin/bash
#SBATCH --job-name=sat_intersect
#SBATCH --array=0-35              # 36 jobs for 36 months
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00           # Adjust based on your needs
#SBATCH --mem=8G                  # Adjust based on data size
#SBATCH --output=logs/sat_%A_%a.out
#SBATCH --error=logs/sat_%A_%a.err

# Load any required modules
### PATH TO MAMBA   # Adjust to your HPC's module system

# Run the Python script with the month index
$(pypath) process_satellite_intersections.py --month-index $SLURM_ARRAY_TASK_ID