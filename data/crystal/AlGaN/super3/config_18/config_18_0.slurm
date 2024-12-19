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
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_30_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_31_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_47_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_46_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_27_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_26_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_50_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_51_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_5_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_4_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_19_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_18_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_13_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_12_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_20_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_21_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_3_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_14_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_15_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi config_18_0.s &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_8_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_9_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_37_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_36_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_40_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_41_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_10_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_11_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_24_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_25_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_6_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_7_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_39_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_38_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_33_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_32_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_44_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_45_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_49_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_48_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_34_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_35_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_43_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_42_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_29_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_28_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_17_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_16_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_23_18_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_22_18_0 &
wait