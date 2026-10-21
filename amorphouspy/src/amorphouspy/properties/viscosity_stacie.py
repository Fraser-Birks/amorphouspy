"""Shear viscosity from independent NVE runs, prepared for analysis with STACIE.

The MD protocol follows the recommendations of STACIE
(https://molmod.github.io/stacie, "Recommendations for MD Simulations"):

1. Every replica is fully independent: its own random structure and its own melt-quench.
2. Each replica is equilibrated at the target temperature and pressure with a Langevin
   thermostat (NPT, ``fix nph`` + ``fix langevin``).
3. Production runs in NVE, continuing from the end of the replica's own equilibration
   without rescaling velocities or volume. The set of replicas as a whole then samples
   the NPT ensemble, while each production run follows unperturbed Newtonian dynamics.

The full pressure tensor is stored at every step, so the viscosity analysis can be
redone later with different choices (block size, run length, number of replicas).

The caller is responsible for giving each replica its own random structure and seed.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.atoms import Atoms

from amorphouspy.fabrication.meltquench import melt_quench_simulation
from amorphouspy.lammps.runner import Thermostat, _run_lammps_md

# Voigt order used by STACIE: xx, yy, zz, yz, zx, xy
_VOIGT_INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))


def _nve_segment(
    structure: Atoms,
    potential: pd.DataFrame,
    n_steps: int,
    timestep: float,
    server_kwargs: dict[str, Any] | None,
    tmp_working_directory: str | Path | None,
) -> tuple[Atoms, dict[str, Any]]:
    """Run one NVE segment from the velocities carried by ``structure``.

    The thermo frame at step 0 (the starting state) is dropped, so ``n_steps`` steps
    give exactly ``n_steps`` samples and consecutive segments join without duplicates.
    """
    final_structure, parsed = _run_lammps_md(
        structure=structure,
        potential=potential,
        n_ionic_steps=n_steps,
        timestep=timestep,
        ensemble="nve",
        n_print_thermo=1,
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
    )
    generic = parsed["generic"]
    pressures = np.asarray(generic["pressures"])
    if len(pressures) != n_steps + 1:
        msg = f"expected {n_steps + 1} thermo frames (steps 0..{n_steps}), got {len(pressures)}"
        raise RuntimeError(msg)
    segment = {
        "pressure_voigt_gpa": np.stack([pressures[1:, i, j] for i, j in _VOIGT_INDICES], axis=1),
        "temperature_k": np.asarray(generic["temperature"])[1:],
        "energy_tot_ev": np.asarray(generic["energy_tot"])[1:],
        "volume_a3": np.asarray(generic["volume"])[1:],
    }
    return final_structure, segment


def stacie_replica_md(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature: float,
    pressure: float,
    *,
    equilibration_steps: int,
    production_steps: int,
    seed: int,
    timestep: float = 1.0,
    melt_temperature: float | None = None,
    cooling_rate: float = 1e13,
    melt_equilibration_steps: int | None = None,
    melt_thermostat: Thermostat = "nose_hoover",
    server_kwargs: dict[str, Any] | None = None,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run one independent replica: melt-quench, Langevin NPT equilibration, NVE production.

    Args:
        structure: Starting structure for this replica (e.g. randomly placed atoms).
            Use a different random structure for each replica.
        potential: LAMMPS potential DataFrame.
        temperature: Target temperature (K).
        pressure: Target pressure (GPa) of the equilibration.
        equilibration_steps: Length of the Langevin NPT equilibration at ``temperature``.
        production_steps: Length of the NVE production run.
        seed: Seed for this replica. Use a different seed for each replica.
        timestep: MD timestep (fs) for all stages after the melt-quench.
        melt_temperature: Melt temperature (K). ``None`` uses the potential's default.
        cooling_rate: Cooling rate (K/s) from ``melt_temperature`` to ``temperature``.
        melt_equilibration_steps: Overrides the equilibration stages of the melt-quench
            protocol. ``None`` uses the protocol defaults.
        melt_thermostat: Thermostat of the melt-quench stages.
        server_kwargs: Forwarded to the LAMMPS runner, e.g. ``{"cores": 4}``.
        tmp_working_directory: Directory in which the LAMMPS run directories are kept.

    Returns:
        A dictionary with:

        - ``production``: per-step NVE data, ``pressure_voigt_gpa`` (shape ``(n, 6)``, order
          xx, yy, zz, yz, zx, xy), ``temperature_k``, ``energy_tot_ev`` and ``volume_a3``.
        - ``structure``: final structure, carrying velocities, for ``extend_nve_production``.
        - ``equilibration``: per-step data of the Langevin NPT stage (``temperature_k``, ``volume_a3``).
        - ``settings``: the arguments that define the replica (temperature, pressure, timestep, seeds).
    """
    rng = np.random.default_rng(seed)
    # LAMMPS' Marsaglia generator (fix langevin, velocity create) accepts seeds in [1, 900_000_000].
    melt_seed, langevin_seed = (int(s) for s in rng.integers(1, 900_000_000, size=2, endpoint=True))

    mq_result = melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=melt_temperature,
        temperature_low=temperature,
        timestep=timestep,
        cooling_rate=cooling_rate,
        equilibration_steps=melt_equilibration_steps,
        thermostat=melt_thermostat,
        seed=melt_seed,
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
    )

    equilibrated, equil_parsed = _run_lammps_md(
        structure=mq_result["structure"],
        potential=potential,
        n_ionic_steps=equilibration_steps,
        timestep=timestep,
        temperature=temperature,
        initial_temperature=0,  # keep the melt-quench velocities
        npt_pressure=pressure,
        ensemble="npt_langevin",
        seed=langevin_seed,
        n_print_thermo=max(1, equilibration_steps // 1000),
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
    )
    equil_generic = equil_parsed["generic"]

    final_structure, production = _nve_segment(
        equilibrated, potential, production_steps, timestep, server_kwargs, tmp_working_directory
    )

    return {
        "production": production,
        "structure": final_structure,
        "equilibration": {
            "temperature_k": np.asarray(equil_generic["temperature"]),
            "volume_a3": np.asarray(equil_generic["volume"]),
        },
        "settings": {
            "temperature": temperature,
            "pressure": pressure,
            "timestep": timestep,
            "seed": seed,
            "melt_seed": melt_seed,
            "langevin_seed": langevin_seed,
        },
    }


def extend_nve_production(
    replica: dict[str, Any],
    potential: pd.DataFrame,
    n_steps: int,
    server_kwargs: dict[str, Any] | None = None,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Continue a replica's NVE production run and append the new steps.

    The run continues from the final positions and velocities of the previous
    segment, so the extended trajectory is one continuous NVE trajectory.

    Args:
        replica: Return value of ``stacie_replica_md`` (or of a previous extension).
        potential: The LAMMPS potential DataFrame used for the replica.
        n_steps: Number of additional NVE steps.
        server_kwargs: Forwarded to the LAMMPS runner.
        tmp_working_directory: Directory in which the LAMMPS run directories are kept.

    Returns:
        A new replica dictionary with the extended ``production`` data and updated ``structure``.
    """
    final_structure, segment = _nve_segment(
        replica["structure"],
        potential,
        n_steps,
        replica["settings"]["timestep"],
        server_kwargs,
        tmp_working_directory,
    )
    production = {key: np.concatenate([replica["production"][key], segment[key]]) for key in segment}
    return {**replica, "production": production, "structure": final_structure}
