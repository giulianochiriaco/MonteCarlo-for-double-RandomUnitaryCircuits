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
#SBATCH --array=0-30:2
#SBATCH --partition=regular1,regular2
#SBATCH --time=11:59:00
#SBATCH --output=./Output/Potts_Single.o%A-%a
#SBATCH --error=./Errors/Potts_Single.e%A-%a
#

# Make sure it's the same module used at compile time
module load python3

# Calculate the parameter of the calculation based on the array index, which indicates lA, i.e. the size of the partition. It is an even number ranging from 0 to Lx (or to Lx/2)
# p is the measurement probability, Lx the size in the x direction (i.e. size of the system), Ly the size in time, Nstep is the number of Wolff steps
lA=$((${SLURM_ARRAY_TASK_ID}))
p=0.05
Lx=30
Ly=36
Nstep=200000
# Run calculation
python Potts_single.py $Lx $Ly $p $lA $Nstep
