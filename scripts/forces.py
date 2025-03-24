import csv
import json
from janus_core.calculations.single_point import SinglePoint
from ase.io import read

# Load structures
# ase_structure = read('/work/e05/e05/bcamino/mace/fine_tuning/AlGaN_super3_all-descriptors.extxyz', index=":")
ase_structure = read('data/crystal/AlGaN/super3/concatenated_files/AlGaN_super3_all_test.xyz', index=":")

# Extract DFT energies and forces
energies_dft = [atoms.info["dft_energy"] for atoms in ase_structure]
dft_forces = [atoms.arrays["dft_forces"].tolist() for atoms in ase_structure]
print(dft_forces)
# # Run MACE single point for energy and forces
# sp_mace = SinglePoint(
#     struct=ase_structure.copy(),
#     arch="mace_mp",
#     device="cuda",
#     model_path="AlGaN-medium-mpa-0.model",
#     calc_kwargs={"default_dtype": "float64"},
#     properties=["energy", "forces"],
# )

# results = sp_mace.run()
# energies_mace = results["energy"]
# mace_forces = [f.tolist() for f in results["forces"]]

# # # ----------------------
# # # Save Energies to CSV
# # # ----------------------
# # with open("all_energies.csv", "w", newline="") as f:
# #     writer = csv.writer(f)
# #     writer.writerow(["DFT", "MACE"])  # Header
# #     writer.writerows(zip(energies_dft, energies_mace))

# # ----------------------
# # Save Forces to JSON
# # ----------------------
# forces_data = {
#     f"struct_{i}": {
#         "dft_forces": dft,
#         "mace_forces": mace
#     }
#     for i, (dft, mace) in enumerate(zip(dft_forces, mace_forces))
# }

# with open("all_forces.json", "w") as f:
#     json.dump(forces_data, f, indent=2)

# print("✅ Saved energies to all_energies.csv and forces to all_forces.json")