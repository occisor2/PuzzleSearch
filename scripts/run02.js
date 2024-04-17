#!/bin/bash
#SBATCH --account=class-cse4163
#SBATCH --job-name=Work02P
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=ptolemy
#SBATCH --no-reque
#SBATCH --qos=class-cse4163
module load openmpi
mpirun -np 2 ./project1 hard_sample.dat sol_hard.02 >& results.02p
