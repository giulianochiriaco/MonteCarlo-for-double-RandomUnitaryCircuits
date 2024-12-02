#!/usr/bin/env bash
#
#SBATCH --job-name=Potts_Single
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gchiriac@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#
#SBATCH --mem-per-cpu=4000mb
#
#SBATCH --array=0,2,4,6,8,10,12,14,16,18,20,22,24
#SBATCH --partition=regular1,regular2
#SBATCH --time=11:59:00
#SBATCH --output=./Output/Potts_Single.o%A-%a
#SBATCH --error=./Errors/Potts_Single.e%A-%a
#

## YOUR CODE GOES HERE (load the modules and do the calculations)
## Sample code:

# Make sure it's the same module used at compile time
module load python3

# Calculate the parameter of the calculation based on the array index,
# e.g. in this case as 5 times the array index
lA=$((${SLURM_ARRAY_TASK_ID}))
p=0.05
Lx=30
Ly=36
Nstep=200000
# Run calculation
python Potts_single.py $Lx $Ly $p $lA $Nstep
