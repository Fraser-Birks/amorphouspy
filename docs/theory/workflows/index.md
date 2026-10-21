# Simulation Workflows

All simulation workflows use LAMMPS as the MD engine via `lammpsparser`. Each workflow handles the complete lifecycle: setting up the simulation, running LAMMPS in a temporary directory, parsing outputs, and returning structured results.

---

## Overview

| Workflow | Function | Purpose |
|---|---|---|
| [**Melt-Quench**](melt_quench.md) | `melt_quench_simulation()` | Generate realistic glass structures from random initial configs |
| [**Molecular Dynamics**](md.md) | `md_simulation()` | Run NVE/NVT/NPT equilibration or production simulations |
| [**Elastic Moduli**](elastic.md) | `elastic_simulation()` | Calculate $C_{11}$, $C_{12}$, $C_{44}$ via stress-strain |
| [**Viscosity**](viscosity.md) | `viscosity_simulation()` | Compute viscosity via Green-Kubo (SACF) |
| [**CTE**](cte.md) | `cte_simulation()` | Coefficient of thermal expansion from NPT |

---

## Common Architecture

All workflows share a common execution pattern:

```mermaid
graph LR
    A[Structure + Potential] --> B[Build LAMMPS input]
    B --> C[Execute LAMMPS]
    C --> D[Parse output]
    D --> E[Return results]
```

1. **Input**: ASE `Atoms` object + potential `DataFrame`
2. **Build**: `lammpsparser` generates LAMMPS input scripts and data files
3. **Execute**: LAMMPS runs in a temporary directory via subprocess
4. **Parse**: Output files (dump, log, custom) are read back
5. **Return**: Structured dictionary with results + final structure

### LAMMPS Command

The LAMMPS launch command is determined by `get_lammps_command()`:

```python
from amorphouspy.lammps.runner import get_lammps_command

cmd = get_lammps_command({"cores": 4})
# Returns: "mpiexec -n 4 lmp_mpi -in lmp.in"
```

By default, it runs `lmp_mpi` from the system PATH via `mpiexec`, with the rank count from `server_kwargs["cores"]` (default 1). To use a different launch command, e.g. a Kokkos/GPU build under SLURM, set the `LAMMPS_COMMAND` environment variable to the full command:

```bash
export LAMMPS_COMMAND="srun lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in lmp.in"
```

A `server_kwargs["lmp_command"]` entry overrides it for a single call. An override replaces the whole command (`cores` is not applied) and must read the input file `lmp.in`.

---

## Detailed Guides

Each workflow has a dedicated page with full parameter documentation, implementation details, and examples:

- [**Melt-Quench Simulation**](melt_quench.md) — Melt equilibration and cooling protocols
- [**Molecular Dynamics**](md.md) — NVT/NPT single-point simulations
- [**Elastic Moduli**](elastic.md) — Stress-strain finite differences method
- [**Viscosity**](viscosity.md) — Green-Kubo stress autocorrelation
- [**CTE Simulation**](cte.md) — NPT fluctuation-based thermal expansion
