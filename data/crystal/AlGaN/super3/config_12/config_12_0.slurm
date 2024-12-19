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
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_42_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_43_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_35_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_34_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_48_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_49_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi config_12_0.s &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_22_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_23_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_16_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_17_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_28_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_29_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_7_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_6_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_25_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_24_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_11_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_10_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_45_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_44_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_32_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_33_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_38_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_39_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_15_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_14_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_9_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_8_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_3_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_21_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_20_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_41_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_40_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_36_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_37_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_46_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_47_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_31_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_30_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_12_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_13_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_51_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_50_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_4_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_5_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_18_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_19_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_26_12_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_27_12_0 &
wait