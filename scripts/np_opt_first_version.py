import numpy as np
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt

# 1. Read all structures from input file
structures = read('np.xyz', index=':')  # List of ASE Atoms objects

# 2. Energy calculator using MACE
def compute_energy(struct):
    optimized_structure = GeomOpt(
        struct=struct.copy(),
        arch="mace_mp",
        device="cpu",  # adjust if needed
        # model_path="/work/e05/e05/bcamino/mace/fine_tuning/test_ft_model/1000/AlGaN-medium-mpa-0.model",
        model_path="/Users/brunocamino/Desktop/UCL/mace_benchmark/models/mace-mpa-0-medium.model",
        calc_kwargs={"default_dtype": "float64"},
        fmax=0.001,
        # filter_kwargs={"hydrostatic_strain": True},
    )
    optimized_structure.run()
    
    return optimized_structure.struct, optimized_structure.struct.get_potential_energy()

# 3. Optimize and write to np_opt.xyz
optimized_structs = []

for i, struct in enumerate(structures):
    print(f"Optimizing structure {i + 1}/{len(structures)}...")
    opt_struct, energy = compute_energy(struct)
    opt_struct.info["comment"] = f"mace_energy = {energy:.6f} eV"
    optimized_structs.append(opt_struct)

# 4. Write all optimized structures with energy comments
write("np_opt.xyz", optimized_structs)
print("All optimized structures written to np_opt.xyz")