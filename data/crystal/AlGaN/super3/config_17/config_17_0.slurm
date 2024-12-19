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
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_32_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_33_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_45_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_44_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_38_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_39_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_25_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_24_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_7_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_6_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_11_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_10_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_22_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_23_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_28_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_29_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_16_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_17_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_35_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_34_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_42_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_43_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_48_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_49_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_12_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_13_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_26_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_27_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_18_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_19_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_51_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_50_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_4_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_5_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_31_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_30_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_46_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_47_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_36_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_37_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_41_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_40_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi config_17_0.s &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_9_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_8_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_15_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_14_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_21_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_20_17_0 &
timeout 2870m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi AlGaN_super3_3_17_0 &
wait