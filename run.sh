#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1 
#SBATCH -c 160
#SBATCH -o run.out
#SBATCH -e run.err
export OMP_NUN_THREADS=160

./winograd small.conf 0
