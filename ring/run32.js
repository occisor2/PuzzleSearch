#!/bin/bash
#SBATCH --account=class-cse4163
#SBATCH --job-name=Ring32P
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=ptolemy
#SBATCH --no-reque
#SBATCH --qos=class-cse4163

module load openmpi
mpirun -np 32 ./ring >& ring.out
