import time
import numpy as np
import os
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint

# === Config ===
device = "cpu"
model_path = "/Users/brunocamino/Desktop/UCL/mace_benchmark/models/mace-mpa-0-medium.model"
calc_kwargs = {"default_dtype": "float64"}
write_step = 2         # How often to write np_opt.xyz
init_write_step = 3    # How often to write np_opt_initial.xyz

# === Step 1: Read input structures ===
structures = read('np.xyz', index=':')

# === Step 2: Evaluate and write initial structures ===
print("Evaluating unrelaxed energies and forces...")
initial_structs_batch = []
first_init_write = True

for i, struct in enumerate(structures):
    print(f"→ Structure {i+1}/{len(structures)}: unrelaxed")

    sp = SinglePoint(
        struct=struct.copy(),
        arch="mace_mp",
        device=device,
        model_path=model_path,
        calc_kwargs=calc_kwargs,
    )
    sp.run()
    evaluated = sp.struct
    energy = evaluated.get_potential_energy()
    forces = evaluated.get_forces()
    max_force = np.linalg.norm(forces, axis=1).max()

    evaluated.info["mace_energy"] = energy
    evaluated.info["max_force"] = max_force

    initial_structs_batch.append(evaluated)

    if (i + 1) % init_write_step == 0 or (i + 1) == len(structures):
        write("np_opt_initial.xyz", initial_structs_batch, format="extxyz", append=not first_init_write)
        first_init_write = False
        print(f"→ Wrote {len(initial_structs_batch)} initial structures to np_opt_initial.xyz")
        initial_structs_batch = []  #  Clear memory!

# === Step 3: Re-read just-written initial structures (chunk by chunk) ===
optimized_structs_batch = []
first_opt_write = True
opt_index = 0
structures = read('np_opt_initial.xyz', index=':')  # Read all, still memory-safe

print("\nStarting geometry optimization with MACE...")
for i, struct in enumerate(structures):
    print(f"\n→ Structure {i+1}/{len(structures)}: optimizing")

    opt = GeomOpt(
        struct=struct.copy(),
        arch="mace_mp",
        device=device,
        model_path=model_path,
        calc_kwargs=calc_kwargs,
        fmax=0.001,
    )

    time0 = time.time()
    opt.run()
    opt_time = time.time() - time0

    relaxed = opt.struct
    energy = relaxed.get_potential_energy()
    forces = relaxed.get_forces()
    max_force = np.linalg.norm(forces, axis=1).max()

    # Try to retrieve optimizer info
    # try:
    #     n_steps = opt.optimizer.nsteps
    # except AttributeError:
    #     n_steps = -1
    # try:
    #     converged = opt.optimizer.converged()
    # except AttributeError:
    converged = max_force < 0.001

    # relaxed.info["mace_energy"] = energy
    relaxed.info["max_force"] = max_force
    relaxed.info["converged"] = converged
    # relaxed.info["n_steps"] = n_steps
    relaxed.info["time"] = opt_time

    optimized_structs_batch.append(relaxed)

    if (i + 1) % write_step == 0 or (i + 1) == len(structures):
        write("np_opt.xyz", optimized_structs_batch, format="extxyz", append=not first_opt_write)
        first_opt_write = False
        print(f"→ Wrote {len(optimized_structs_batch)} optimized structures to np_opt.xyz")
        optimized_structs_batch = []  #  Clear memory!