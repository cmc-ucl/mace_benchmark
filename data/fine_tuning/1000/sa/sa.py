import numpy as np
import csv
from ase import Atoms
from ase.io import write, read
from copy import deepcopy
import random
from mace.calculators import MACECalculator
from janus_core.calculations.geom_opt import GeomOpt

# Load MACE calculator
mace_calc = MACECalculator(model_path="/work/e05/e05/bcamino/mace/fine_tuning/test_ft_model/1000/AlGaN-medium-mpa-0.model")

# Initialize AlN supercell with 27 Al and 27 Ga atoms
def generate_initial_structure():
    ase_structure = read('AlGaN_super3_27_initial.xyz')  # Load your AlN supercell
    
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
        device="cuda",
        model_path="/work/e05/e05/bcamino/mace/fine_tuning/test_ft_model/1000/AlGaN-medium-mpa-0.model",
        calc_kwargs={"default_dtype": "float64"},
        fmax=0.001,
        filter_kwargs={"hydrostatic_strain": True},
    )

    optimized_structure.run()
    return optimized_structure.struct, optimized_structure.struct.get_potential_energy()

# Compute RMSD between two structures
def compute_rmsd(struct1, struct2):
    pos1 = struct1.get_positions()
    pos2 = struct2.get_positions()
    return np.sqrt(np.mean((pos1 - pos2) ** 2))

# Simulated Annealing for Al/Ga swapping with energy tracking
def simulated_annealing(struct, T_init=1000, cooling_rate=0.95, steps=1000, output_file="sa_trajectory_1000.xyz", energy_output_file="sa_energies.csv", rmsd_threshold=0.01, patience=100):
    current_structure = deepcopy(struct)
    best_structure = deepcopy(struct)
    optimised_structure, current_energy = compute_energy(current_structure)
    best_energy = current_energy
    T = T_init  # Initial temperature

    all_structures = [optimised_structure]  
    all_energies = [current_energy]
    previous_structure = optimised_structure.copy()
    
    # Initialize CSV file
    with open(energy_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Energy", "RMSD"])  

    no_change_count = 0  # Counter for unchanged structures

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

        # Compute RMSD with the previous structure
        rmsd = compute_rmsd(previous_structure, new_optimised_struct)
        previous_structure = new_optimised_struct.copy()

        # Check if structure is still changing
        if rmsd < rmsd_threshold:
            no_change_count += 1
        else:
            no_change_count = 0  # Reset counter if structure changes

        # Stop if structure has not changed for `patience` steps
        if no_change_count >= patience:
            print(f"Stopping early at step {step} as structures have not changed in {patience} steps.")
            break

        # Store structure and energy
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

        # Print progress
        print(f"Step {step}: Energy = {current_energy:.5f}, Best = {best_energy:.5f}, RMSD = {rmsd:.5f}, T = {T:.2f}")

        # Append results every 50 steps
        if step % 50 == 0:
            write(output_file, all_structures, append=True)  
            with open(energy_output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for i, energy in enumerate(all_energies):
                    writer.writerow([step - len(all_energies) + i + 1, energy, rmsd])  
            
            all_structures = []
            all_energies = []

    # Save remaining structures and energies at the end
    write(output_file, all_structures, append=True)
    with open(energy_output_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i, energy in enumerate(all_energies):
            writer.writerow([steps - len(all_energies) + i + 1, energy, rmsd])

    print(f"Saved all structures to {output_file}")
    print(f"Saved all energies to {energy_output_file}")

    return best_structure, best_energy

# Run the optimization
initial_structure = generate_initial_structure()
optimized_structure, optimized_energy = simulated_annealing(initial_structure)

# Save the lowest-energy structure separately
optimized_structure.write("optimized_AlGaN.xyz")
print("Optimization complete! Lowest Energy:", optimized_energy)
