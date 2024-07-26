#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1 
#SBATCH -c 160
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH --exclusive
export OMP_NUN_THREADS=160
export OMP_PROC_BIND=true
export OMP_PLACES=threads

# module load libs/kml

./winograd small.conf 0
