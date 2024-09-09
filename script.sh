#!/bin/bash
#SBATCH --job-name GAPfitting
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 719:30:00
#SBATCH --mem=3005500M
#SBATCH --output output
#SBATCH --error log



cd $PWD
ulimit -s unlimited

module purge
module load gcc/11.2.0  blas/3.12.0   lapack/3.12.0   tbb/2021.8.0   compiler-rt/2023.0.0   mkl/2023.0.0
module load python/3.9.7
export OMP_STACKSIZE=3T


python3 Gap_fit.py


for id in $(seq 0 1000)
        do
                FILE=result_try_$id
                if [ -f "$FILE" ]; then
                        echo "$FILE file exists."
                else
                        mv output result_try_$id
                        break
                fi
done

