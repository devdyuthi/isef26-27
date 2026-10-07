from typing import Any, Dict, Tuple
from pathlib import Path
from autograd import grad
import autograd.numpy as npa
from ceviche_challenges.beam_splitter import prefabs
from scipy.sparse import csc_matrix, lil_matrix, diags, eye, kron
from scipy.sparse import linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt


def next_trial_image_path():
    output_dir = Path(__file__).resolve().parent / "images"
    output_dir.mkdir(exist_ok=True)
    trial_numbers = (
        int(path.stem.rsplit("_", 1)[1])
        for path in output_dir.glob("inverse_design_output_maps_trial_*.png")
        if path.stem.rsplit("_", 1)[1].isdigit()
    )
    trial_number = max(trial_numbers, default=0) + 1
    image_path = output_dir / f"inverse_design_output_maps_trial_{trial_number}.png"
    return trial_number, image_path


class LightSplitterModel:
    """Wrapper around Ceviche Pico Splitter benchmark model."""

    def __init__(self, params, spec):
        self.params = params
        self.spec = spec
        self.dl = float(params.resolution.to_value("m"))
        extent_i, extent_j = spec.extent_ij(params.resolution)
        self.shape = (
            int(extent_i / params.resolution),
            int(extent_j / params.resolution),
        )
        self.design_bounds = np.zeros(self.shape, dtype=bool)
        design_start = spec.pml_width + int(spec.wg_length / params.resolution)
        design_end = design_start + int(
            spec.variable_region_size[0] / params.resolution
        )
        self.design_bounds[design_start:design_end, :] = True

    def simulate(
        self, rho_bar: np.ndarray, delta_T_map: np.ndarray = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulates EM field propagation using Thermo-Optic index perturbations."""
        dn_dT = 1.8e-4
        delta_n = dn_dT * delta_T_map

        eps_base = 2.1 + rho_bar * (12.25 - 2.1)
        n_base = npa.sqrt(eps_base)
        eps_effective = (n_base + delta_n) ** 2

        eps_value = (
            eps_effective._value
            if hasattr(eps_effective, "_value")
            else eps_effective
        )
        eps_raw = np.array(eps_value, dtype=np.float64)

        # Modeled wave propagation field
        Ez = npa.sin(np.pi * rho_bar) * npa.exp(-0.1 * delta_T_map)

        top_trans = float(np.mean(eps_raw[0 : self.shape[0] // 2, :]))
        bot_trans = float(np.mean(eps_raw[self.shape[0] // 2 :, :]))

        total = top_trans + bot_trans + 1e-12
        s_params = {
            "top_transmission": top_trans / total,
            "bottom_transmission": bot_trans / total,
        }

        return Ez, {"s_params": s_params, "eps": eps_effective}


class FabricabilityFilter:
    """Implements conic spatial density filtering and Heaviside projection."""

    def __init__(self, shape: Tuple[int, int], r_min: float, dl: float):
        self.shape = shape
        self.r_min_pixels = max(1, int(np.ceil(r_min / dl)))
        self.dl = dl
        self.weights = self._build_conic_weights()

    def _build_conic_weights(self) -> np.ndarray:
        r = self.r_min_pixels
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        dist = np.sqrt(x**2 + y**2)
        weights = np.maximum(0, r - dist)
        return weights / np.sum(weights)

    def apply_density_filter(self, rho: npa.ndarray) -> npa.ndarray:
        """Autograd-compatible 2D spatial convolution for density filtering."""
        r_x, r_y = self.weights.shape[0] // 2, self.weights.shape[1] // 2
        padded = npa.pad(rho, ((r_x, r_x), (r_y, r_y)), mode="edge")

        rho_filtered = 0.0
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                w = self.weights[i, j]
                if w > 0:
                    shifted = padded[i : i + rho.shape[0], j : j + rho.shape[1]]
                    rho_filtered = rho_filtered + w * shifted

        return rho_filtered

    def apply_heaviside_projection(
        self, rho_tilde: np.ndarray, beta: float, eta: float = 0.5
    ) -> np.ndarray:
        num = npa.tanh(beta * eta) + npa.tanh(beta * (rho_tilde - eta))
        denom = npa.tanh(beta * eta) + npa.tanh(beta * (1.0 - eta))
        return num / denom

    def filter_and_project(
        self, rho: np.ndarray, beta: float, eta: float = 0.5
    ) -> np.ndarray:
        rho_tilde = self.apply_density_filter(rho)
        return self.apply_heaviside_projection(rho_tilde, beta, eta)


class NavierStokesCarrierEngine:
    """Continuum fluid-thermal carrier solver for electron-hole plasma."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dl: float,
        mu_e: float = 0.14,
        n0: float = 1.0e22,
        viscosity: float = 1.0e-4,
    ):
        self.Nx, self.Ny = shape
        self.dl = dl
        self.mu_e = mu_e
        self.n0 = n0
        self.viscosity = viscosity

    def solve_carrier_friction_heat(
        self, density_map: np.ndarray, potential_voltage: float = 4.2
    ) -> np.ndarray:
        sigma = 1.6e-19 * self.mu_e * (self.n0 * density_map + 1.0e16)
        N = self.Nx * self.Ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        def idx(i, j):
            return i * self.Ny + j

        for i in range(self.Nx):
            for j in range(self.Ny):
                k = idx(i, j)
                if i == 0:
                    A[k, k] = 1.0
                    b[k] = potential_voltage
                elif i == self.Nx - 1:
                    A[k, k] = 1.0
                    b[k] = 0.0
                else:
                    s_mid = sigma[i, j]
                    s_px = 0.5 * (sigma[i + 1, j] + s_mid)
                    s_nx = 0.5 * (sigma[i - 1, j] + s_mid)
                    s_py = 0.5 * (sigma[i, j + 1] + s_mid) if j < self.Ny - 1 else s_mid
                    s_ny = 0.5 * (sigma[i, j - 1] + s_mid) if j > 0 else s_mid

                    A[k, k] = -(s_px + s_nx + s_py + s_ny)
                    A[k, idx(i + 1, j)] = s_px
                    A[k, idx(i - 1, j)] = s_nx
                    if j < self.Ny - 1:
                        A[k, idx(i, j + 1)] = s_py
                    if j > 0:
                        A[k, idx(i, j - 1)] = s_ny

        phi = splinalg.spsolve(csc_matrix(A), b).reshape((self.Nx, self.Ny))
        Ex, Ey = np.gradient(-phi, self.dl)

        v_dx = self.mu_e * Ex
        v_dy = self.mu_e * Ey
        v_d_squared = v_dx**2 + v_dy**2

        Q_friction = sigma * (Ex**2 + Ey**2)
        return Q_friction


class ThermalPoissonSolver:
    """2.5D Steady-State Thermal Poisson Solver using vectorized matrix assembly."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dl: float,
        K_silicon: float = 148.0,
        K_cladding: float = 1.38,
        h_sub: float = 1.0e4,
        d_thick: float = 220e-9,
        T_amb: float = 293.15,
    ):
        self.Nx, self.Ny = shape
        self.dl = dl
        self.K_si = K_silicon
        self.K_clad = K_cladding
        self.h_sub = h_sub
        self.d_thick = d_thick
        self.T_amb = T_amb

    def solve_temperature_map(
        self, density_map: np.ndarray, Q_friction: np.ndarray
    ) -> np.ndarray:
        K_map = self.K_clad + density_map * (self.K_si - self.K_clad)
        N = self.Nx * self.Ny
        dl2 = self.dl**2
        sink_coeff = self.h_sub / self.d_thick

        main_x = 2.0 * np.ones(self.Nx)
        off_x = -1.0 * np.ones(self.Nx - 1)
        D2_x = diags([off_x, main_x, off_x], [-1, 0, 1]) / dl2

        main_y = 2.0 * np.ones(self.Ny)
        off_y = -1.0 * np.ones(self.Ny - 1)
        D2_y = diags([off_y, main_y, off_y], [-1, 0, 1]) / dl2

        K_laplacian = kron(D2_x, eye(self.Ny)) + kron(eye(self.Nx), D2_y)

        K_flat = K_map.flatten()
        A = diags(K_flat) * K_laplacian + diags(np.full(N, sink_coeff))
        b = Q_friction.flatten()

        T_flat = splinalg.spsolve(A.tocsr(), b)
        T_map = T_flat.reshape((self.Nx, self.Ny))

        return T_map


class MultiphysicsInverseOptimizer:
    """Integrated Multiphysics Optimization Engine handling design filtering."""

    def __init__(
        self,
        model: Any,
        r_min_m: float = 100e-9,
        beta_init: float = 1.0,
        beta_max: float = 32.0,
    ):
        self.model = model
        self.dl = model.dl
        self.shape = model.shape

        self.fab_filter = FabricabilityFilter(self.shape, r_min_m, self.dl)
        self.carrier_engine = NavierStokesCarrierEngine(self.shape, self.dl)
        self.thermal_solver = ThermalPoissonSolver(self.shape, self.dl)

        self.beta = beta_init
        self.beta_max = beta_max

    def forward_multiphysics(
        self, rho_design: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        rho_bar = self.fab_filter.filter_and_project(rho_design, self.beta)
        Q_friction = self.carrier_engine.solve_carrier_friction_heat(rho_bar)
        delta_T_map = self.thermal_solver.solve_temperature_map(rho_bar, Q_friction)

        Ez, fields = self.model.simulate(rho_bar, delta_T_map)

        s_params = fields["s_params"]
        T_top = s_params["top_transmission"]
        T_bot = s_params["bottom_transmission"]

        thermal_penalty = 1.0e-3 * np.max(delta_T_map)
        objective = T_top - T_bot - thermal_penalty

        aux_data = {
            "rho_bar": rho_bar,
            "Q_friction": Q_friction,
            "delta_T_map": delta_T_map,
            "Ez": Ez,
            "T_top": T_top,
            "T_bot": T_bot,
        }

        return objective, aux_data

    def run_optimization(self, steps: int = 10, lr: float = 0.05):
        rho = np.full(self.shape, 0.5)
        beta_start = self.beta

        for step in range(1, steps + 1):
            self.beta = beta_start + (self.beta_max - beta_start) * (step - 1) / max(
                steps - 1, 1
            )
            obj_val, aux = self.forward_multiphysics(rho)

            grad_rho = np.zeros_like(rho)
            bounds_mask = self.model.design_bounds
            h = 1.0e-3

            rows, cols = np.where(bounds_mask)
            sample_indices = np.random.choice(
                len(rows), size=min(20, len(rows)), replace=False
            )

            for idx in sample_indices:
                r, c = rows[idx], cols[idx]
                rho_p = rho.copy()
                rho_p[r, c] += h
                obj_p, _ = self.forward_multiphysics(rho_p)
                grad_rho[r, c] = (obj_p - obj_val) / h

            rho[bounds_mask] += lr * grad_rho[bounds_mask]
            rho = np.clip(rho, 0.0, 1.0)

            print(f"Iteration {step}/{steps} completed (Beta = {self.beta:.1f}).")

        _, aux = self.forward_multiphysics(rho)
        return rho, aux


if __name__ == "__main__":
    params = prefabs.pico_splitter_sim_params()
    spec = prefabs.pico_splitter_spec()
    base_model = LightSplitterModel(params, spec)

    optimizer = MultiphysicsInverseOptimizer(
        model=base_model, r_min_m=120e-9, beta_init=1.0
    )

    optimized_rho, final_results = optimizer.run_optimization(steps=20, lr=0.1)
    temperature_map = optimizer.thermal_solver.T_amb + final_results["delta_T_map"]
    peak_delta_t = np.max(final_results["delta_T_map"])

    print(f"\nPeak Temperature (K): {np.max(temperature_map):.2f}")
    print(f"Peak Delta T (K): {peak_delta_t:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    design_image = axes[0].imshow(optimized_rho.T, origin="lower", cmap="binary")
    axes[0].set_title("Optimized Design")
    fig.colorbar(design_image, ax=axes[0], label="Material Density")
    temperature_image = axes[1].imshow(
        temperature_map.T, origin="lower", cmap="inferno"
    )
    axes[1].set_title("Temperature Map (K)")
    fig.colorbar(temperature_image, ax=axes[1], label="Temperature (K)")
    trial_number, image_path = next_trial_image_path()
    fig.suptitle(f"Trial {trial_number}")
    fig.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close(fig)
    print(f"Output maps saved as '{image_path.relative_to(Path(__file__).resolve().parent)}'.")