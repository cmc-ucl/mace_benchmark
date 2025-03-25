from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import copy
import numpy as np

import numpy as np
from pymatgen.core import Structure


def build_symmetry_equivalent_configurations(atom_indices,N_index):
    
    if len(N_index) == 0:
        return np.array([np.zeros(len(atom_indices[0]),dtype='int')]) # TEST
    configurations = atom_indices == -1

    for index in N_index:
        configurations += atom_indices == index
    configurations = configurations.astype(int)

    unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
    return unique_configurations

def generate_random_structures(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False):
    
    #N_atoms: number of sites to replace
    #N_config: number of attempts
    #DFT_config: number of final structures generated
    #new_species: new atomic number
    #active_sites: sites in the structure to replace (useful for Al/GaN)
    #atom_indices: indices obtained from get_all_configurations
    #Returns: symmetry independent structures

    all_structures = []

    
    if active_sites is False:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)
        
    

    # Generate a random configurations
    descriptor_all = []
    structures_all = []
    config_all = []
    config_unique = []
    config_unique_count = []
    n_sic = 0
    N_attempts= 0
    
    while n_sic < DFT_config and N_attempts <N_config:
        N_attempts += 1
        sites_index = np.random.choice(num_sites,N_atoms,replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(initial_structure)
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)

        sic = sec[0]
        
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)

        if not is_in_config_unique:  
            config_unique.append(sic)

            config_unique_count.append(len(sec))
            n_sic += 1


    final_structures = []

    for config in config_unique:

        N_index = np.where(config==1)[0]
        structure_tmp = copy.deepcopy(initial_structure)
        for N in N_index:
            structure_tmp.replace(N,new_species)
        final_structures.append(structure_tmp)
    if return_multiplicity == True:
        return final_structures,config_unique_count
    else:
        return final_structures

def get_all_configurations_pmg(structure_pmg,prec=6):

    symmops = SpacegroupAnalyzer(structure_pmg).get_symmetry_operations()

    coordinates = np.round(np.array(structure_pmg.frac_coords),prec)
    n_symmops = len(symmops)
    atom_numbers = np.array(structure_pmg.atomic_numbers)
    lattice = structure_pmg.lattice.matrix
    
    original_structure_pmg = copy.deepcopy(structure_pmg)
            
    rotations = []
    translation = []

    atom_indices = np.ones((len(symmops),structure_pmg.num_sites),dtype='int')
    atom_indices *= -1

    structures = []
    for i,symmop in enumerate(symmops):

        atom_indices_tmp = []
        coordinates_new = []
        for site in coordinates:
            coordinates_new.append(np.round(symmop.operate(site),prec))

        structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=False,
                                  to_unit_cell=False)

        for k,coord in enumerate(original_structure_pmg.frac_coords):
            structure_tmp.append(original_structure_pmg.atomic_numbers[k],coord,coords_are_cartesian=False,
                                 validate_proximity=False)
        

        for m in range(len(atom_numbers)):
            index = len(atom_numbers)+m
            for n in range(len(atom_numbers)):

                if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index],tolerance=0.001):

                    atom_indices[i,m] = n
                    break


    return atom_indices

def write_CRYSTAL_gui_from_data(lattice_matrix,atomic_numbers,
                                cart_coords, file_name, dimensionality = 3):

    input_data = [f'{dimensionality} 1 1\n']
    
    identity = np.eye(3).astype('float')

    for row in lattice_matrix:
        input_data.append(' '.join(f'{val:.6f}' for val in row)+'\n')
    input_data.append('1\n')    
    for row in identity:
        input_data.append(' '.join(f'{val:.6f}' for val in row)+'\n')
    input_data.append('0.000000 0.000000 0.000000\n')
    input_data.append(f'{len(cart_coords)}\n')
    for row, row2 in zip(atomic_numbers, cart_coords):
        input_data.append(f'{row} '+' '.join(f'{val:.6f}' for val in row2)+'\n')
    input_data.append('0 0')

    with open(file_name, 'w') as file:
        for line in input_data:
            file.write(f"{line}")


def write_extended_xyz(structure, filename="structure.exyz", forces=None, energy=None):
    """
    Write a Pymatgen Structure to an extended XYZ (EXYZ) file.
    
    Parameters:
    - structure: Pymatgen Structure object
    - filename: Name of the file to save the extended XYZ
    - forces: Optional. List of forces for each atom. Shape: (n_atoms, 3)
    - energy: Optional. Total energy of the system
    
    """
    with open(filename, 'w') as f:
        # 1️⃣ Write the number of atoms
        num_atoms = structure.num_sites
        f.write(f"{num_atoms}\n")
        
        # 2️⃣ Write the metadata line
        lattice = structure.lattice.matrix
        # Format the lattice vectors into a single row
        lattice_flat = " ".join(f"{val:.6f}" for row in lattice for val in row)
        
        metadata_parts = [f'Lattice="{lattice_flat}"']
        if energy is not None:
            metadata_parts.append(f'Energy={energy:.6f}')
        
        # If forces are provided, flatten them for the metadata line
        if forces is not None:
            forces_flat = " ".join(f"{force:.6f}" for force in np.array(forces).flatten())
            metadata_parts.append(f'Forces="{forces_flat}"')
        
        metadata_line = " ".join(metadata_parts)
        f.write(f"{metadata_line}\n")
        
        # 3️⃣ Write atomic positions and optional forces
        atomic_symbols = [site.species_string for site in structure.sites]
        cart_coords = structure.cart_coords
        
        for i, (element, pos) in enumerate(zip(atomic_symbols, cart_coords)):
            x, y, z = pos
            line = f"{element} {x:.6f} {y:.6f} {z:.6f}"
            if forces is not None:
                fx, fy, fz = forces[i]
                line += f" {fx:.6f} {fy:.6f} {fz:.6f}"
            f.write(line + "\n")
    
    print(f"Extended XYZ file saved as '{filename}'")

