import numpy as np
import scipy.ndimage as ndimage
import autograd.numpy as npa
from autograd import grad

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from splitter import LightSplitterModel
from ceviche_challenges.beam_splitter import prefabs

# Initialize splitter parameters and model instance from splitter.py
params = prefabs.pico_splitter_sim_params()
spec = prefabs.pico_splitter_spec()
model = LightSplitterModel(params, spec)

nx, ny = model.shape     
dl = model.dl           

# Base physical constants for silicon thermo-optic effect
N0 = 3.48               # Base refractive index of Silicon at 1550 nm
DN_DT = 1.8e-4          # Thermo-optic coefficient dn/dT for Silicon (1/K)
EPS_BG = 1.0 ** 2       # Background relative permittivity (Air/Cladding)


def apply_moulding_filter(design_grid, radius=2):
    return ndimage.gaussian_filter(design_grid, sigma=radius)


def remove_floaters(moulded_grid, threshold=0.5, min_size=5):
    binary_grid = moulded_grid > threshold
    labeled_array, num_features = ndimage.label(binary_grid)
    sizes = ndimage.sum(binary_grid, labeled_array, range(1, num_features + 1))
    
    mask_size = sizes >= min_size
    remove_pixel = mask_size[labeled_array - 1]
    binary_grid[~remove_pixel] = 0
    
    return ndimage.gaussian_filter(binary_grid.astype(float), sigma=0.8)


def projection_filter(design_grid, beta=8.0, eta=0.5):
    """
    Differentiable Heaviside projection using autograd.numpy (npa)
    to push fluid optimization values towards binary states (0 or 1).
    """
    numerator = npa.tanh(beta * eta) + npa.tanh(beta * (design_grid - eta))
    denominator = npa.tanh(beta * eta) + npa.tanh(beta * (1.0 - eta))
    return numerator / denominator


def _gaussian_kernel_2d(radius=2, truncate=4.0):
    r = int(truncate * radius + 0.5)
    coords = npa.arange(-r, r + 1)
    kernel_1d = npa.exp(-(coords ** 2) / (2 * radius ** 2))
    kernel_1d /= kernel_1d.sum()
    return npa.outer(kernel_1d, kernel_1d)

_MOULD_KERNEL_2D = _gaussian_kernel_2d(radius=2)


def apply_moulding_filter_differentiable(design_variable):
    """
    Autograd-traceable 2D spatial filter using pure matrix operations.
    """
    k_shape = _MOULD_KERNEL_2D.shape
    r_x, r_y = k_shape[0] // 2, k_shape[1] // 2
    
    padded = npa.pad(design_variable, ((r_x, r_x), (r_y, r_y)), mode='constant')
    out = npa.zeros_like(design_variable)
    
    for i in range(k_shape[0]):
        for j in range(k_shape[1]):
            weight = _MOULD_KERNEL_2D[i, j]
            shifted = padded[i:i + design_variable.shape[0], j:j + design_variable.shape[1]]
            out = out + weight * shifted
            
    return out


def solve_temperature_map_ghost_nodes(total_power_watts, heater_bounds, grid_shape=(nx, ny), 
                                      heater_thickness=220e-9, box_thickness=2.0e-6, 
                                      dl=dl, max_iter=20, tol=1e-3):
    """
    2D Thermal solver incorporating 3D vertical conduction loss to the substrate.
    """
    nx_grid, ny_grid = grid_shape
    N = nx_grid * ny_grid
    
    h_top = 1e3            # Top convective coefficient (W/m²·K)
    k_box = 1.4            # SiO2 conductivity (W/m·K)
    T_ambient = 293.15     # Baseline room temperature (K)
    
    g_substrate = k_box / box_thickness 
    
    (x0, x1), (y0, y1) = heater_bounds
    num_cells_x = x1 - x0
    num_cells_y = y1 - y0
    heater_area = (num_cells_x * dl) * (num_cells_y * dl)
    Q_surface = total_power_watts / heater_area
    
    Q = np.zeros(grid_shape)
    Q[x0:x1, y0:y1] = Q_surface

    dl2 = dl**2
    main_x = 2.0 * np.ones(nx_grid)
    off_x = -1.0 * np.ones(nx_grid - 1)
    D2_x = sp.diags([off_x, main_x, off_x], [-1, 0, 1]) / dl2

    main_y = 2.0 * np.ones(ny_grid)
    off_y = -1.0 * np.ones(ny_grid - 1)
    D2_y = sp.diags([off_y, main_y, off_y], [-1, 0, 1]) / dl2

    K = sp.kron(D2_x, sp.eye(ny_grid)) + sp.kron(sp.eye(nx_grid), D2_y)

    boundary_counts = np.zeros(grid_shape, dtype=float)
    boundary_counts[0, :] += 1.0
    boundary_counts[-1, :] += 1.0
    boundary_counts[:, 0] += 1.0
    boundary_counts[:, -1] += 1.0
    
    boundary_counts_flat = boundary_counts.flatten()
    boundary_indices = np.where(boundary_counts_flat > 0)[0]

    G = sp.diags(boundary_counts_flat / dl2, format='csr')

    T_current = np.full(grid_shape, T_ambient)
    
    for _ in range(max_iter):
        k_eff = np.mean(149.0 * ((298.15 / np.maximum(T_current, 1.0)) ** 1.33))
        gamma = (h_top + g_substrate) / (k_eff * heater_thickness)
        
        A = (K + G + gamma * sp.eye(N)).tocsr()
        b = (Q.flatten() / (k_eff * heater_thickness)) + (gamma * T_ambient)
        b[boundary_indices] += (2.0 * T_ambient * boundary_counts_flat[boundary_indices]) / dl2

        temp_vector = spla.spsolve(A, b)
        T_new = temp_vector.reshape(grid_shape)
        
        if np.max(np.abs(T_new - T_current)) < tol:
            break
            
        T_current = T_new

    return T_current


def make_objective_function(delta_T_map, beta_proj=8.0):
    def objective_function(design_variable):
        # 1. Filter layout to maintain fabricability
        smoothed_design = apply_moulding_filter_differentiable(design_variable)
        
        # 2. Project layout to clear binary values (0 or 1)
        binarized_design = projection_filter(smoothed_design, beta=beta_proj)

        # 3. Calculate local thermo-optic permittivity: eps_r = (n_0 + (dn/dT)*delta_T)^2
        n_silicon_thermal = N0 + (DN_DT * delta_T_map)
        eps_silicon_thermal = n_silicon_thermal ** 2
        
        # Interpolate relative permittivity map based on binarized design density
        eps_r_map = EPS_BG + binarized_design * (eps_silicon_thermal - EPS_BG)

   
        output = model.simulate(eps_r_map)

   
        if isinstance(output, tuple):
            Ez = output[0]
        else:
            Ez = output

        output_power = npa.sum(npa.abs(Ez[:, -10:]) ** 2)

        return -output_power
        
    return objective_function


def run_inverse_design(iterations=25):
    design = np.full((nx, ny), 0.5)
    
    heater_bounds = ((40, 60), (40, 60))
    P_joule = 0.015  # 15 mW Joule heating
    
    
    temp_map = solve_temperature_map_ghost_nodes(
        total_power_watts=P_joule, 
        heater_bounds=heater_bounds, 
        grid_shape=(nx, ny), 
        heater_thickness=220e-9,
        box_thickness=2.0e-6,
        dl=dl
    )
    
    delta_T_map = temp_map - 293.15
    
    learning_rate = 0.05
    m_state = np.zeros_like(design)
    v_state = np.zeros_like(design)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for step in range(iterations):
        
        beta_proj = 1.0 + (step / max(1, iterations - 1)) * 15.0
        
        obj_fn = make_objective_function(delta_T_map, beta_proj=beta_proj)
        grad_fn = grad(obj_fn)

        grads = grad_fn(design)

        
        m_state = beta1 * m_state + (1 - beta1) * grads
        v_state = beta2 * v_state + (1 - beta2) * (grads ** 2)
        m_hat = m_state / (1 - beta1 ** (step + 1))
        v_hat = v_state / (1 - beta2 ** (step + 1))

        design -= learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)
        design = np.clip(design, 0, 1)
        print(f"Iteration {step+1}/{iterations} completed (Beta = {beta_proj:.1f}).")


    smoothed_final = apply_moulding_filter(design)
    projected_final = projection_filter(smoothed_final, beta=16.0)
    final_feasible_design = remove_floaters(projected_final)
    
    return final_feasible_design, temp_map


def plot_output_maps(chip_geometry, thermal_map):
    """
    Plots and saves side-by-side maps for the optimized chip layout 
    and the thermal temperature distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Optimized Chip Geometry Layout
    im1 = axes[0].imshow(chip_geometry.T, origin='lower', cmap='binary')
    axes[0].set_title("Optimized Chip Geometry Map")
    axes[0].set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")
    fig.colorbar(im1, ax=axes[0], label="Material Density")

    # Temperature Thermal Profile
    im2 = axes[1].imshow(thermal_map.T, origin='lower', cmap='inferno')
    axes[1].set_title("Thermal Profile Map (K)")
    axes[1].set_xlabel("X (cells)")
    axes[1].set_ylabel("Y (cells)")
    fig.colorbar(im2, ax=axes[1], label="Temperature (K)")

    plt.tight_layout()
    

    plt.savefig("inverse_design_output_maps.png", dpi=300)
    print("Output maps saved as 'inverse_design_output_maps.png'.")
    
    plt.show()


if __name__ == "__main__":
    final_chip_geometry, thermal_profile = run_inverse_design(iterations=5)
    print("\nPeak Temperature (K):", np.max(thermal_profile))
    print("Peak Delta T (K):", np.max(thermal_profile) - 293.15)
    
    plot_output_maps(final_chip_geometry, thermal_profile)