#!/bin/bash
#SBATCH --account=class-cse4163
#SBATCH --job-name=Debug08P
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:02:00
#SBATCH --partition=ptolemy
#SBATCH --no-reque
#SBATCH --qos=class-cse4163
module load openmpi
mpirun -np 8 ../project1 ../easy_sample.dat sol_easy.08 >& out.08p
