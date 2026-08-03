---
name: add_dataset_parser
description: Guide for creating new full-field displacement dataset parsers capable of handling diverse experimental data formats (DIC grids, CSV, VTK, HDF5, MAT) and integrating them into the JAX hyperelastic discovery pipeline. Use this when adding new synthetic or experimental datasets to the project.
---

# Skill: Adding a New Full-Field Dataset Parser (`add_dataset_parser`)

This skill provides step-by-step guidance for integrating new full-field displacement datasets—especially experimental Digital Image Correlation (DIC) grids, finite element benchmark exports, or non-standard file formats (VTK, HDF5, CSV, MAT)—into the repository's data ingest framework.

---

## 1. Handling Diverse Experimental Formats
Experimental DIC systems and simulation engines output full-field kinematic measurements in diverse formats. Before modifying code, inspect the data files to identify:
1. **Node Coordinates & Displacements:** Locate reference position meshes ($x, y$) and observed displacement fields ($u_x, u_y$).
2. **Element Connectivity (Mesh Cells):** If triangular or quadrangular triangulation is present (e.g., node indices per element), extract it as an array of shape `(n_elements, num_nodes_per_elem)`. If data is an unmeshed regular DIC grid, construct Delaunay triangulations or simple Cartesian elemental connectivity.
3. **Boundary Conditions & Reaction Forces:** Extract reaction force time-series or boundary loadstep forces (`forces: [right, left, top, bottom]`) required for loss function weighting during unsupervised GP optimization.

### Quick-Reference Loaders for Common Formats
- **Tabular CSV (DIC / Simulation):** Use `pandas.read_csv` to load node tables and element tables.
- **HDF5 / NetCDF:** Use `h5py.File(path, 'r')` or `xarray` to extract structured multi-dimensional coordinate arrays.
- **VTK / VTU (FEA Exports):** Use `meshio.read(path)` to extract `mesh.points`, `mesh.point_data["displacement"]`, and `mesh.cells`.
- **NumPy (.npy / .npz):** Direct JAX load via `dict(jnp.load(path))` or `np.load()`.

---

## 2. Implement the Dataset Subclass (`core/datasetclass.py`)
Create a new parser subclassing `HyperelasticDataset` inside `core/datasetclass.py`. 

```python
from .datasetclass import HyperelasticDataset

class ExperimentalDICDataset(HyperelasticDataset):
    def __init__(self, data_dir: os.PathLike, **kwargs):
        self.data_dir = data_dir
        self.loadsteps = sorted(os.listdir(data_dir))
        # Store metadata or calibration factors here

    def __len__(self) -> int:
        return len(self.loadsteps)

    def __getitem__(self, idx: int) -> dict:
        step_path = os.path.join(self.data_dir, self.loadsteps[idx])
        
        # 1. Parse raw coordinates, displacements, and connectivity
        mesh_pos, u, cells, reaction_forces, bc = self._parse_raw_step(step_path)
        
        # 2. Extract element-level kinematics
        coords_elems = mesh_pos[cells]  # Shape: (n_elems, nodes_per_elem, 2)
        disp_elems   = u[cells]         # Shape: (n_elems, nodes_per_elem, 2)

        # 3. Compute Deformation Gradients using JAX element-wise vectorization
        f_2x2 = jax.vmap(lambda ce, de: deformation_gradient_element(ce, de))(coords_elems, disp_elems)
        f_2x2 = f_2x2.reshape(-1, 2, 2)

        # 4. Promote 2D Deformation Gradient to 3x3 (Assuming Plane Stress/Strain with F_33 = 1.0)
        n_elems = f_2x2.shape[0]
        F = jnp.tile(jnp.eye(3), (n_elems, 1, 1))
        F = F.at[:, :2, :2].set(f_2x2)

        # 5. Compute Isotropic Invariants and Eigenvalues
        B_train = B_func(F)
        I1_train, I2_train, I3_train = I1_func(B_train), I2_func(B_train), I3_func(B_train)
        invariants = jnp.stack([I1_train, I2_train, I3_train], axis=-1)

        B_eig_val = jnp.real(jnp.linalg.eigvalsh(B_train))
        
        # Note: If ground-truth stresses (P or sigma) are unknown for experimental data, 
        # set to None or zeros, as unsupervised GP relies on equilibrium & reaction forces.
        sigma_placeholder = jnp.zeros_like(F)
        sigma_eig_val = jnp.real(jnp.linalg.eigvalsh(sigma_placeholder))
        coeffs, _ = solve_for_coefficients_batched(B_eig_val, sigma_eig_val)

        return dict(
            F=F, 
            sigma=sigma_placeholder, 
            coeffs=coeffs, 
            invariants=invariants, 
            cells=cells, 
            coords_elems=coords_elems, 
            disp_elems=disp_elems, 
            bc=bc, 
            reaction_forces=reaction_forces
        )
        
    def get_data(self) -> list:
        return [self[i] for i in range(len(self))]
```

---

## 3. Register in `DatasetFactory`
Locate `DatasetFactory.create()` in `core/datasetclass.py` and register your new dataset class under a recognizable identifier string:

```python
class DatasetFactory:
    @staticmethod
    def create(dataset_type: str, **kwargs) -> HyperelasticDataset:
        if dataset_type == "dataset/precomputed_vfm":
            return PrecomputedVFMDataset(kwargs["data_path"])
        elif dataset_type == "benchmark":
            return BenchmarkDataset(kwargs["data_dir"], kwargs["noise"], kwargs["mat_model"])
        elif dataset_type == "traction":
            return TractionDataset(kwargs.get("data_dir", "/home/mmdiscovery/shared/dataset/isihara_fix"))
        elif dataset_type == "experimental_dic":  # <-- NEW ENTRY
            return ExperimentalDICDataset(kwargs["data_dir"])
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
```

---

## 4. Verification & Testing
Always verify that your newly added dataset parser behaves correctly in 64-bit precision without generating `NaN` values during deformation gradient inversion:
1. Enable x64 precision: `jax.config.update("jax_enable_x64", True)`.
2. Instantiate the parser via `DatasetFactory.create("experimental_dic", data_dir="path/to/test_dir")`.
3. Assert that `data["F"].shape` equals `(num_elements, 3, 3)` and that `jnp.all(jnp.isfinite(data["invariants"]))` returns `True`.
