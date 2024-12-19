#!/bin/bash
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-algor-smw
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0
export SLURM_CPU_FREQ_REQ=2250000

# Run calculations
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_11_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_10_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_7_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_6_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_25_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_24_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_38_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_39_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi config_13_0.s &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_45_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_44_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_32_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_33_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_48_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_49_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_42_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_43_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_35_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_34_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_16_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_17_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_28_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_29_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_22_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_23_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_46_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_47_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_31_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_30_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_18_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_19_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_51_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_50_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_4_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_5_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_26_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_27_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_12_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_13_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_3_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_21_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_20_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_9_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_8_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_15_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_14_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_41_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_40_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_36_13_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_37_13_0 &
wait