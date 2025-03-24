import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from scipy.stats import spearmanr, kendalltau

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Convert lattice parameters to a 3x3 lattice matrix.

    Parameters:
        a, b, c (float): Lattice constants.
        alpha, beta, gamma (float): Angles (in degrees) between the lattice vectors.

    Returns:
        numpy.ndarray: 3x3 lattice matrix.
    """
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Compute the lattice vectors
    v_x = a
    v_y = b * np.cos(gamma_rad)
    v_z = c * np.cos(beta_rad)

    w_y = b * np.sin(gamma_rad)
    w_z = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)

    u_z = np.sqrt(c**2 - v_z**2 - w_z**2)

    # Assemble the lattice matrix
    lattice_matrix = np.array([
        [v_x, 0, 0],
        [v_y, w_y, 0],
        [v_z, w_z, u_z],
    ])

    return lattice_matrix

import numpy as np
from sklearn.metrics import root_mean_squared_error

def get_errors(dft_energies_form, mace_energies_form, 
               dft_forces=None, mace_forces=None,
               dft_stress=None, mace_stress=None,
               sig_figs=3, return_errors=False):
    """
    Compute and print RMSE and %RMSE for energy, forces, and stress.

    All RMSE values are reported in meV. Percentage RMSEs are unitless.

    Parameters:
        dft_energies_form (np.ndarray): DFT formation energies
        mace_energies_form (np.ndarray): MACE formation energies
        dft_forces (np.ndarray): Optional list or array of DFT forces (N, atoms, 3)
        mace_forces (np.ndarray): Optional list or array of MACE forces (N, atoms, 3)
        dft_stress (np.ndarray): Optional array of DFT stresses (N, 3, 3)
        mace_stress (np.ndarray): Optional array of MACE stresses (N, 3, 3)
        sig_figs (int): Significant figures for printing
        return_errors (bool): If True, return errors as dictionary

    Returns:
        dict (optional): Errors if return_errors=True
    """
    def sci_notation(value, sig_figs):
        return f"{value:.{sig_figs}e}"

    errors = {}

    # --- Energy Errors ---
    energy_rmse = root_mean_squared_error(dft_energies_form, mace_energies_form)
    mean_ref_energy = np.mean(np.abs(dft_energies_form))
    energy_pct_rmse = 100 * energy_rmse / mean_ref_energy if mean_ref_energy > 0 else np.nan

    energy_rmse_meV = energy_rmse * 1000

    print("Energy Errors:")
    print("RMSE:", sci_notation(energy_rmse_meV, sig_figs), "meV")
    print("%RMSE:", sci_notation(energy_pct_rmse, sig_figs), "%")

    errors.update({
        "energy_rmse_meV": energy_rmse_meV,
        "energy_pct_rmse": energy_pct_rmse
    })

    # --- Force Errors ---
    if dft_forces is not None and mace_forces is not None:
        ref_flat = np.concatenate(dft_forces, axis=0).ravel()
        pred_flat = np.concatenate(mace_forces, axis=0).ravel()

        force_rmse = root_mean_squared_error(ref_flat, pred_flat)
        mean_ref_force = np.mean(np.abs(ref_flat))
        force_pct_rmse = 100 * force_rmse / mean_ref_force if mean_ref_force > 0 else np.nan

        force_rmse_meV = force_rmse * 1000

        print("\nForce Errors:")
        print("RMSE:", sci_notation(force_rmse_meV, sig_figs), "meV/Å")
        print("%RMSE:", sci_notation(force_pct_rmse, sig_figs), "%")

        errors.update({
            "force_rmse_meV": force_rmse_meV,
            "force_pct_rmse": force_pct_rmse
        })

    # --- Stress Errors ---
    if dft_stress is not None and mace_stress is not None:
        dft_flat = np.reshape(dft_stress, (-1, 9))
        mace_flat = np.reshape(mace_stress, (-1, 9))

        stress_rmse = root_mean_squared_error(dft_flat, mace_flat)
        mean_ref_stress = np.mean(np.abs(dft_flat))
        stress_pct_rmse = 100 * stress_rmse / mean_ref_stress if mean_ref_stress > 0 else np.nan

        stress_rmse_meV = stress_rmse * 1000

        print("\nStress Errors:")
        print("RMSE:", sci_notation(stress_rmse_meV, sig_figs), "meV")
        print("%RMSE:", sci_notation(stress_pct_rmse, sig_figs), "%")

        errors.update({
            "stress_rmse_meV": stress_rmse_meV,
            "stress_pct_rmse": stress_pct_rmse
        })

    if return_errors:
        return errors

def get_rankings(dft_energies_form,mace_energies_form, print_coeff=True):
    spearman_corr, _ = spearmanr(dft_energies_form, mace_energies_form)
    kendall_corr, _ = kendalltau(dft_energies_form, mace_energies_form)

    if print_coeff == True:

        print("Spearman's Rank Correlation:", round(spearman_corr,5))
        print("Kendall's Tau:", round(kendall_corr,5))
    else:
        return spearman_corr, kendall_corr

def parse_dft_stress(stress_list):
    """Convert DFT stress (9 elements, full 3x3 matrix) into a NumPy 3x3 array."""
    return np.array([
        [stress_list[0], stress_list[1], stress_list[2]],
        [stress_list[3], stress_list[4], stress_list[5]],
        [stress_list[6], stress_list[7], stress_list[8]]
    ])

def parse_mace_stress(stress_list):
    """Convert MACE stress (6-element Voigt notation) into a symmetric 3x3 NumPy array."""
    s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = stress_list
    return np.array([
        [s_xx,  s_xy,  s_xz],
        [s_xy,  s_yy,  s_yz],
        [s_xz,  s_yz,  s_zz]
    ])    

