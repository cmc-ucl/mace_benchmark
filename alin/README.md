generate descriptors

use janus descriptors --config descriptors.yml

visualise distributions with

distributions_dft.py

split data for fine tunning with sample_splut.py

0. Installation

python3 -m pip uninstall mace-torch

python3 -m pip install -U torch torchvision torchaudio
python3 -m pip install git+https://github.com/ACEsuit/mace.git

_____________________
python3 -m pip install uv  
uv pip install git+https://github.com/stfc/janus-core.git  
python3 -m pip uninstall mace-torch
python3 -m pip install -U torch torchvision torchaudio
python3 -m pip install git+https://github.com/ACEsuit/mace.git
python3 -m pip install cuequivariance-torch
python3 -m pip install cuequivariance
python3 -m pip install cuequivariance-ops-torch-cu12
python3 -m pip install git+https://gitlab.com/drFaustroll/ase.git@npt_triangular
python3 -m pip install -U git+https://github.com/CheukHinHoJerry/torch-dftd.git
_____________________

1. Build descriptors 
janus descriptors --config ../descriptors.yml --struct ../../data/crystal/AlGaN/super3/concatenated_files/AlGaN_super3_all_test.xyz

2. Visualise data

3. Split data

sample_split.py --n_samples 2000 --config_types random --pre AlGaN AlGaN_super3_all-descriptors.extxyz    

4. Train 

NOTES:
I need the isolated atom descricptors in the train set

ERRORS:

When trying to use mpi on Archer2
Matplotlib created a temporary cache directory at /tmp/matplotlib-gzbmg41t because there was an issue with the default path (/home/e05/e05/bcamino/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=8556533.0. Some of your processes may have been killed by the cgroup out-of-memory handler.
srun: error: nid002715: task 85: Out Of Memory
srun: launch/slurm: _step_signal: Terminating StepId=8556533.0
slurmstepd: error: *** STEP 8556533.0 ON nid002715 CANCELLED AT 2025-01-23T11:58:37 ***

2025-02-12 15:20:35.104 INFO: Using foundation model for multiheads finetuning with Materials Project data
Traceback (most recent call last):
  File "/work/e05/e05/bcamino/miniconda3/envs/mace/lib/python3.11/site-packages/mace/tools/multihead_tools.py", line 119, in assemble_mp_data
    os.makedirs(cache_dir, exist_ok=True)
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  [Previous line repeated 1 more time]
  File "<frozen os>", line 225, in makedirs
PermissionError: [Errno 13] Permission denied: '/home/e05'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/work/e05/e05/bcamino/miniconda3/envs/mace/bin/mace_run_train", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/work/e05/e05/bcamino/miniconda3/envs/mace/lib/python3.11/site-packages/mace/cli/run_train.py", line 66, in main
    run(args)
  File "/work/e05/e05/bcamino/miniconda3/envs/mace/lib/python3.11/site-packages/mace/cli/run_train.py", line 291, in run
    collections = assemble_mp_data(args, tag, head_configs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/e05/e05/bcamino/miniconda3/envs/mace/lib/python3.11/site-packages/mace/tools/multihead_tools.py", line 186, in assemble_mp_data
    raise RuntimeError(
RuntimeError: Model or descriptors download failed and no local model found
srun: error: nid002390: task 0: Exited with exit code 1
srun: launch/slurm: _step_signal: Terminating StepId=8713593.0