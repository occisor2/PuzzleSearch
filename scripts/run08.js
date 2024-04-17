#!/bin/bash
#SBATCH --account=class-cse4163
#SBATCH --job-name=Work08P
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=ptolemy
#SBATCH --no-reque
#SBATCH --qos=class-cse4163
module load openmpi
mpirun -np 8 ../project1 ../hard_sample.dat sol_hard.08>& results.08p
