import numpy as np
from ase import Atoms
from ase.io import write, read
from copy import deepcopy
import random
from mace.calculators import MACECalculator
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint

# Load MACE calculator
mace_calc = MACECalculator(model_path="/Users/brunocamino/Desktop/UCL/mace_benchmark/models/fine_tuned/1000/AlGaN-medium-mpa-0.model")

# Initialize AlN supercell with 27 Al and 27 Ga atoms
def generate_initial_structure():
    # Load initial structure
    ase_structure = read('/Users/brunocamino/Desktop/UCL/mace_benchmark/mace_benchmark/scripts/AlGaN_super3_27_initial.xyz')  # Load your AlN supercell (replace with your code)
    
    # Select 27 Al atoms and replace them with Ga
    al_indices = [i for i, atom in enumerate(ase_structure) if atom.symbol == "Al"]
    ga_indices = random.sample(al_indices, 27)  # Randomly choose 27 Al atoms to replace

    # Replace chosen Al atoms with Ga
    for idx in ga_indices:
        ase_structure[idx].symbol = "Ga"

    return ase_structure

# Compute energy of a structure
def compute_energy(struct):
        optimized_structure = GeomOpt(
        struct=struct.copy(),
        arch="mace_mp",
        device="cpu", #check this
        model_path="/Users/brunocamino/Desktop/UCL/mace_benchmark/models/fine_tuned/1000/AlGaN-medium-mpa-0.model",
        calc_kwargs={"default_dtype": "float64"},
        fmax=0.01,
        filter_kwargs={"hydrostatic_strain": True},
                )
        
        optimized_structure.run()

        # sp_mace = SinglePoint(
        # struct=optimized_structure,
        # arch="mace_mp",
        # device="cpu",
        # model_path="small",
        # calc_kwargs={"default_dtype": "float64"},
        # properties="energy",
        # )

        # energy = sp_mace.run()["energy"]

        return optimized_structure.struct, optimized_structure.struct.get_potential_energy()

# Simulated Annealing for Al/Ga swapping with energy tracking
def simulated_annealing(struct, T_init=1000, cooling_rate=0.95, steps=1000, output_file="sa_trajectory.xyz"):
    current_structure = deepcopy(struct)
    best_structure = deepcopy(struct)
    optimised_structure, current_energy = compute_energy(current_structure)
    best_energy = current_energy
    T = T_init  # Initial temperature
    
    all_structures = [optimised_structure]  # Include the initial relaxed structure
    all_energies = [current_energy]

    for step in range(steps):
        # Generate new structure by swapping one Al and one Ga
        new_structure = deepcopy(current_structure)
        al_atoms = [i for i, atom in enumerate(new_structure) if atom.symbol == "Al"]
        ga_atoms = [i for i, atom in enumerate(new_structure) if atom.symbol == "Ga"]

        # Randomly pick one Al and one Ga and swap them
        al_idx = random.choice(al_atoms)
        ga_idx = random.choice(ga_atoms)
        new_structure[al_idx].symbol, new_structure[ga_idx].symbol = (
            new_structure[ga_idx].symbol,
            new_structure[al_idx].symbol,
        )

        # Compute energy of the new structure
        new_optimised_struct, new_energy = compute_energy(new_structure)
        print(new_optimised_struct)
        # Store structure and energy
        # new_structure.info["energy"] = new_energy
        all_structures.append(deepcopy(new_optimised_struct))
        all_energies.append(new_energy)

        # Metropolis criterion: Accept move if lower energy or with probability exp(-ΔE/T)
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_structure = deepcopy(new_structure)
            current_energy = new_energy

            # Update best structure
            if current_energy < best_energy:
                best_structure = deepcopy(current_structure)
                best_energy = current_energy

        # Cool down temperature
        T *= cooling_rate

        print(f"Step {step}: Current Energy = {current_energy:.5f}, Best Energy = {best_energy:.5f}, T = {T:.2f}")
        # Print progress every 50 steps
        if step % 50 == 0:
            write(output_file, new_optimised_struct, append=True)  # Append to XYZ file
    # Save all explored structures to an extended XYZ file
    write(output_file, all_structures)
    print(f"Saved all {len(all_structures)} structures to {output_file}")

    return best_structure, best_energy

# Run the optimization
initial_structure = generate_initial_structure()
optimized_structure, optimized_energy = simulated_annealing(initial_structure)

# Save the lowest-energy structure separately
optimized_structure.write("optimized_AlGaN.xyz")
print("Optimization complete! Lowest Energy:", optimized_energy)