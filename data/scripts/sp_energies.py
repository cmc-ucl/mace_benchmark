import csv
import numpy as np
from janus_core.calculations.single_point import SinglePoint
from ase.io import read


ase_structure = read('AlGaN-test.xyz', index=":")

energies_dft = [atoms.info["dft_energy"] for atoms in ase_structure]
num_al = [np.sum(atoms.get_atomic_numbers() == 13) for atoms in ase_structure]

sp_mace = SinglePoint(
    struct=ase_structure.copy(),
    arch="mace_mp",
    device="cpu",
    model_path="AlGaN-medium-mpa-0.model",
    calc_kwargs={"default_dtype": "float64"},
    properties="energy",
)

energies_mace = sp_mace.run()["energy"]



# Save to CSV
with open("energies.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DFT", "MACE","num_al"])  # Header
    writer.writerows(zip(energies_dft, energies_mace,num_al))

print("Saved as energies.csv")