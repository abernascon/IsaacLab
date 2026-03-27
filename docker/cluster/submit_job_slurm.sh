#!/usr/bin/env bash

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --tmp=100G
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=abernascon@ethz.ch
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"


# Load ETH proxy for internet access on compute nodes
module load eth_proxy

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh

