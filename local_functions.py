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

def get_errors(dft_energies_form,mace_energies_form):
    mse = mean_squared_error(dft_energies_form,mace_energies_form)
    rmse = root_mean_squared_error(dft_energies_form,mace_energies_form)
    # Compute Percentage MSE
    percentage_mse = (mse / np.mean(dft_energies_form**2)) * 100

    # Compute absolute errors
    absolute_errors = np.abs(dft_energies_form - mace_energies_form)

    # Compute max absolute error
    max_absolute_error = np.max(absolute_errors)

    # Compute max percentage error (avoid division by zero)
    max_percentage_error = np.max((absolute_errors / np.abs(dft_energies_form)) * 100)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Percentage MSE:", percentage_mse, "%")
    print("Max Absolute Error:", max_absolute_error)
    print("Max Percentage Error:", max_percentage_error, "%")

def get_rankings(dft_energies_form,mace_energies_form):
    spearman_corr, _ = spearmanr(dft_energies_form, mace_energies_form)
    kendall_corr, _ = kendalltau(dft_energies_form, mace_energies_form)

    print("Spearman's Rank Correlation:", spearman_corr)
    print("Kendall's Tau:", kendall_corr)

