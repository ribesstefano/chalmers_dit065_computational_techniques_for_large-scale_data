#!/bin/bash
#SBATCH --partition=cpu-markov
#SBATCH --cpus-per-task=1
#SBATCH --output=/bayes_datainbackup/home/2022/ribes/chalmers_dit065_computational_techniques_for_large-scale_data/assignment_1/src/serverinfo_markov.out
#SBATCH --error=/bayes_datainbackup/home/2022/ribes/chalmers_dit065_computational_techniques_for_large-scale_data/assignment_1/src/serverinfo_markov.error
#SBATCH --chdir=/bayes_datainbackup/home/2022/ribes/chalmers_dit065_computational_techniques_for_large-scale_data/assignment_1/src # Working directory
#SBATCH --export=ALL,TEMP=/scratch,TMP=/scratch,TMPDIR=/scratch
echo "================================================================================"
echo "Number of CPU cores"
echo "================================================================================"
lscpu | egrep 'CPU\(s\)|Core|Socket|Thread'
echo "================================================================================"
echo "CPU Type"
echo "================================================================================"
lscpu | egrep "Model name| MHz"
echo "================================================================================"
echo "Disk Amount"
echo "================================================================================"
df -h --total
echo "================================================================================"
echo "Login shell memory"
echo "================================================================================"
ps auxU ribes | awk '{memory +=$4}; END {print memory }'