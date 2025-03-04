# mace_benchmark

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

python script/sample_split.py --n_samples 2000 --config_types random --pre AlGaN AlGaN_super3_all-descriptors.extxyz

4. Train 

NOTES:
I need the isolated atom descricptors in the train set.

The system_name needs to be the same for the problems I am interested in. They would be different if it was, for example, a molecule on a surface.

The num_samples_pt is the number of structures used from the original model. If this is high, max_num_epochs is low because the model is already converged.

5. Running on Archer2
The files (mp files) are saved in the .cache in home, but they need to be on the compute nodes. Download them and then move to the work home. Set the cache path in the slurm script.

</b>Can I get GPU access?</b>

6. WIP

Tests:
num_samples_pt: 100000, max_num_epochs: 10
num_samples_pt: 10000, max_num_epochs: 100
num_samples_pt: 1000, max_num_epochs: 1000

I have restarted them to have:

num_samples_pt: 100000, max_num_epochs: 4
num_samples_pt: 10000, max_num_epochs: 50
num_samples_pt: 1000, max_num_epochs: 100

because it was taking too long to get to the max_num_epochs above.

The errors on forces are lower than on the energies.


7. Questions
Which models to test? https://github.com/ACEsuit/mace-mp

MPA first - baseline
Same testing wiht the fine tuned one
(MACE-OMAT-0 - no ft)

Do the mp files in cache depend on the model used? If so, how can I separate them?

Don't I need the descriptor in the IsoaltedAtom? No

What is valid_indices_2025.txt? - saved so we know which elements from the mp datasets are used.

I ran the convergence test on num_samples_pt using 2000 structures in the training set. Is that ok? - Yes and it doesn't really change the total time. I can test - after num_samples_pt



Should we test ema too? - no need

Patience is larger than 10 - it never reaches it anyway

Once I get below a certan error are any model the same? - yes


8. ERRORS:
Using ceq on Archer2

Traceback (most recent call last):
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/bin/mace_run_train", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/lib/python3.12/site-packages/mace/cli/run_train.py", line 66, in main
    run(args)
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/lib/python3.12/site-packages/mace/cli/run_train.py", line 593, in run
    model = run_e3nn_to_cueq(deepcopy(model), device=device)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/lib/python3.12/site-packages/mace/cli/convert_e3nn_cueq.py", line 151, in run
    transfer_weights(source_model, target_model, max_L, correlation)
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/lib/python3.12/site-packages/mace/cli/convert_e3nn_cueq.py", line 111, in transfer_weights
    target_model.load_state_dict(target_dict)
  File "/work/e05/e05/bcamino/miniconda3/envs/janus-amd/lib/python3.12/site-packages/torch/nn/modules/module.py", line 2581, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for ScaleShiftMACE:
	Unexpected key(s) in state_dict: "products.0.symmetric_contractions.weight", "products.1.symmetric_contractions.weight".


9. Future projects
</b>AlGaN</b>
First do some MC on the structures (MACE optgeom) and then calculate the DFT energy of the low energy structures
Build clusters using KLMC (genetic) - Talk to Dong-gi

</b>LiMnO2</b>
Use the current model to calculate the energy of the structures we have with GULP. What starting point?
See if I find a Jahn-Teller distorsion. Section 12 in the MACE paper (SI)
Compare to experiment
Then calculate the PBE0 energies (start fm and then afm)

</b>LTS</b>
Calculate the energy and optgeom with the model as is of the structures we have and compare to the B3LYP energies.

Expand the work done with Alex (see if we can ft the model and then use it for larger cells.)