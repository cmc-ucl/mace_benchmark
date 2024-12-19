#!/bin/bash
#SBATCH --nodes=52
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
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_25_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_24_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_7_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_6_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_52_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_11_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_10_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_32_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_33_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_45_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_44_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_38_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_39_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_35_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_34_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi config_11_0.s &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_42_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_43_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_48_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_49_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_22_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_23_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_28_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_29_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_16_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_17_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_31_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_30_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_46_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_47_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_12_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_13_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_26_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_27_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_4_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_5_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_51_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_50_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_18_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_19_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_15_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_14_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_9_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_8_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_21_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_20_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_3_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_2_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_36_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_37_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_41_11_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_40_11_0 &
wait