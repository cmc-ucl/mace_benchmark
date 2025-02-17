#!/bin/bash --login

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=python_test
#SBATCH --time=24:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=e05-power-smw
#SBATCH --partition=standard
#SBATCH --qos=standard

source /work/e05/e05/bcamino/miniconda3/bin/activate #replace this with your path
conda activate mace

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Run your Python program
#   Note that srun MUST be used to wrap the call to python,
#   otherwise your code will run serially
export OMP_NUM_THREADS=128
export XDG_CACHE_HOME='/work/e05/e05/bcamino/'
#srun --distribution=block:block --hint=nomultithread mace_run_train --config  ft_medium.yml
mace_run_train --config  ft_medium.yml