from typing import Tuple, List
from ceviche_challenges import units as u
from ceviche_challenges import defs
from ceviche_challenges import model_base
from ceviche_challenges import modes
from ceviche_challenges import params as _params
from ceviche_challenges.beam_splitter import spec as _spec
import autograd.numpy as npa
import numpy as np


class LightSplitterModel(model_base.Model):
    """A wavelength-selective directional coupler."""

    def __init__(
        self,
        params: _params.CevicheSimParams,
        spec: _spec.BeamSplitterSpec,
    ):
        self.params = params
        self.spec = spec

        extent_i, extent_j = spec.extent_ij(params.resolution)

        self._shape = (
            u.resolve(extent_i, params.resolution),
            u.resolve(extent_j, params.resolution),
        )

        self.bg_density_ports()

    def bg_density_ports(self, init_design_region: bool = False):
        p = self.params
        s = self.spec

        density = np.zeros(self.shape)

        monitor_offset = u.resolve(s.input_monitor_offset, p.resolution)
        extent_i, extent_j = s.extent_ij(p.resolution)
        center_j = extent_j / 2

        wg_offset = s.wg_separation / 2 + s.wg_width / 2
        wgs_j = [center_j - wg_offset, center_j + wg_offset]

        # Create two parallel waveguides
        for wg_j in wgs_j:
            y1 = int(u.resolve(wg_j - s.wg_width / 2, p.resolution))
            y2 = int(u.resolve(wg_j + s.wg_width / 2, p.resolution))
            density[:, y1:y2] = 1.0

        # Define 4 ports
        port_i_input = int(s.pml_width + u.resolve(s.port_pml_offset, p.resolution))
        port_i_output = int(u.resolve(extent_i - s.port_pml_offset, p.resolution) - s.pml_width)

        port_js = [
            int(u.resolve(wgs_j[1], p.resolution)),
            int(u.resolve(wgs_j[1], p.resolution)),
            int(u.resolve(wgs_j[0], p.resolution)),
            int(u.resolve(wgs_j[0], p.resolution)),
        ]
        port_is = [port_i_input, port_i_output, port_i_output, port_i_input]
        port_dirs = [
            defs.Direction.X_POS,
            defs.Direction.X_NEG,
            defs.Direction.X_NEG,
            defs.Direction.X_POS,
        ]

        ports = []
        for port_i, port_j, port_dir in zip(port_is, port_js, port_dirs):
            ports.append(
                modes.WaveguidePort(
                    x=port_i,
                    y=port_j,
                    width=int(u.resolve(s.wg_width + 2 * s.wg_mode_padding, p.resolution)),
                    order=1,
                    dir=port_dir,
                    offset=monitor_offset,
                )
            )

        # Save state
        self._bg_density = density
        self._ports = ports

        # Create design mask
        design_mask = np.zeros(self.shape, dtype=bool)
        design_i_start = int(
            u.resolve(s.pml_width * p.resolution + s.wg_length, p.resolution)
        )
        design_i_end = design_i_start + u.resolve(
            s.variable_region_size[0], p.resolution
        )
        design_mask[design_i_start:design_i_end, :] = True
        self._design_bounds = design_mask


    # --- CLASS PROPERTIES & METHODS (Outside of bg_density_ports) ---

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def design_bounds(self) -> np.ndarray:
        return self._design_bounds

    @property
    def design_region_coords(self) -> Tuple[int, int, int, int]:
        rows, cols = np.where(self._design_bounds)
        return (rows.min(), cols.min(), rows.max() + 1, cols.max() + 1)

    @property
    def density_bg(self) -> np.ndarray:
        return self._bg_density

    @property
    def ports(self) -> List[modes.WaveguidePort]:
        return self._ports

    @property
    def dl(self) -> float:
        return float(self.params.resolution.to("m").value)

    @property
    def pml_width(self) -> int:
        return self.spec.pml_width

    @property
    def cladding_permittivity(self) -> float:
        return self.spec.cladding_permittivity

    @property
    def slab_permittivity(self) -> float:
        return self.spec.slab_permittivity

    @property
    def output_wavelengths(self) -> np.ndarray:
        return np.asarray(self.params.wavelengths.to_value(u.nm))

    def density(self, design_variable: np.ndarray) -> np.ndarray:
        return npa.where(self._design_bounds, design_variable, self._bg_density)

    def relative_permittivity(self, design_variable: np.ndarray) -> np.ndarray:
        dens = self.density(design_variable)
        return self.spec.cladding_permittivity + dens * (
            self.spec.slab_permittivity - self.spec.cladding_permittivity
        )

    def output_ports(self) -> List[modes.WaveguidePort]:
        return self._ports

    def relative_permittivity_with_heat(
        self,
        design_variable: np.ndarray,
        delta_T_map: np.ndarray,
        dn_dT: float = 1.8e-4,
    ) -> np.ndarray:
        # Base permittivity to refractive index
        eps_base = self.relative_permittivity(design_variable)
        n_0 = npa.sqrt(eps_base)

        # Refractive index change due to temperature
        n_thermal = n_0 + dn_dT * delta_T_map

        # Return perturbed permittivity: eps_r = n^2
        return n_thermal**2


# --- SIMULATION INSTANTIATION ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ceviche_challenges.beam_splitter import prefabs

    # Instantiate using the package's compatible pico splitter defaults.
    params = prefabs.pico_splitter_sim_params()
    spec = prefabs.pico_splitter_spec()
    model = LightSplitterModel(params, spec)

    # 2. Initial design array (0.5 represents mid-density material)
    design_vars = np.full(model.shape, 0.5)

    # 3. Calculate permittivity map
    eps_r = model.relative_permittivity(design_vars)
    
    print("--- SIMULATION GRID READY ---")
    print(f"Grid shape (Nx, Ny): {eps_r.shape}")
    print(f"Cladding permittivity: {np.min(eps_r):.2f}")
    print(f"Silicon core permittivity: {np.max(eps_r):.2f}")

    # 4. Simulate a local hotspot (+50 K in the grid center)
    delta_T = np.zeros(model.shape)
    cx, cy = model.shape[0] // 2, model.shape[1] // 2
    delta_T[cx - 5 : cx + 5, cy - 5 : cy + 5] = 50.0  # 50 Kelvin shift

    eps_r_heated = model.relative_permittivity_with_heat(design_vars, delta_T)
    print(f"Peak permittivity under heat: {np.max(eps_r_heated):.4f}")

    # Run one excitation to verify the full ceviche simulation path.
    s_params, _ = model.simulate(design_vars, max_parallelizm=1)
    print(f"S-parameter result shape: {s_params.shape}")

    # 5. Plot the structure to verify visually
    plt.figure(figsize=(10, 4))
    plt.imshow(eps_r.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Permittivity (ε_r)")
    plt.title("Waveguide Splitter Layout")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.show()