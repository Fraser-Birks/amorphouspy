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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from numpy.typing import ArrayLike, NDArray
from stacie import LorentzModel, UnitConfig, compute_spectrum, estimate_acint
from stacie.model import SpectrumModel
from stacie.utils import block_average

from amorphouspy.fabrication.meltquench import melt_quench_simulation
from amorphouspy.lammps.io import frames_from_melt_quench_result
from amorphouspy.lammps.runner import Thermostat, _run_lammps_md

# Voigt order used by STACIE: xx, yy, zz, yz, zx, xy
_VOIGT_INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))

KB_J_PER_K = 1.380649e-23
KB_EV_PER_K = 8.617333262e-5
GPA_TO_PA = 1e9
A3_TO_M3 = 1e-30
FS_TO_S = 1e-15
_MIN_BLOCKS = 2

VISCOSITY_UNITS = UnitConfig(
    acint_symbol="η",
    acint_unit_str="Pa s",
    acint_fmt=".4e",
    time_unit=1e-12,
    time_unit_str="ps",
    time_fmt=".3f",
    freq_unit=1e12,
    freq_unit_str="THz",
)


def _nve_segment(
    structure: Atoms,
    potential: pd.DataFrame,
    n_steps: int,
    timestep: float,
    server_kwargs: dict[str, Any] | None,
    tmp_working_directory: str | Path | None,
    n_dump: int | None = None,
) -> tuple[Atoms, dict[str, Any], dict[str, Any]]:
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
        n_dump=n_dump,
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
    return final_structure, segment, generic


def _stage_frames(template: Atoms, generic: dict[str, Any] | None, stage: str) -> list[Atoms]:
    """Snapshots of one MD stage, wrapped into the cell and labelled with ``stage``."""
    if generic is None or "positions" not in generic:
        return []
    frames = frames_from_melt_quench_result({"result": [generic]}, template)
    for i, frame in enumerate(frames):
        frame.wrap()
        # Thermo data is written on a different cadence from the snapshots, so the per-frame
        # thermo values attached by frames_from_melt_quench_result would be misaligned; drop them.
        for key in ("temperature", "energy_pot", "energy_tot", "volume", "pressure", "step"):
            frame.info.pop(key, None)
        frame.info["stage"] = stage
        frame.info["stage_frame"] = i
    return frames


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
    n_dump: int | None = None,
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
        n_dump: If set, every stage writes a snapshot every ``n_dump`` steps, returned as ``trajectory``.
        server_kwargs: Forwarded to the LAMMPS runner, e.g. ``{"cores": 4}``.
        tmp_working_directory: Directory in which the LAMMPS run directories are kept.

    Returns:
        A dictionary with:

        - ``production``: per-step NVE data, ``pressure_voigt_gpa`` (shape ``(n, 6)``, order
          xx, yy, zz, yz, zx, xy), ``temperature_k``, ``energy_tot_ev`` and ``volume_a3``.
        - ``structure``: final structure, carrying velocities, for ``extend_nve_production``.
        - ``equilibration``: per-step data of the Langevin NPT stage (``temperature_k``, ``volume_a3``).
        - ``settings``: the arguments that define the replica (temperature, pressure, timestep, seeds).
        - ``trajectory``: snapshots of all stages in order (``atoms.info["stage"]`` names the stage),
          wrapped into the cell. Empty unless ``n_dump`` is set.
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
        n_dump=n_dump,
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
        n_dump=n_dump,
        n_print_thermo=max(1, equilibration_steps // 1000),
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
    )
    equil_generic = equil_parsed["generic"]

    final_structure, production, prod_generic = _nve_segment(
        equilibrated, potential, production_steps, timestep, server_kwargs, tmp_working_directory, n_dump
    )

    trajectory: list[Atoms] = []
    if n_dump is not None:
        for i, stage_generic in enumerate(mq_result["result"]):
            trajectory += _stage_frames(structure, stage_generic, f"melt_quench_{i}")
        trajectory += _stage_frames(structure, equil_generic, "npt_langevin")
        trajectory += _stage_frames(structure, prod_generic, "nve")

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
        "trajectory": trajectory,
    }


def extend_nve_production(
    replica: dict[str, Any],
    potential: pd.DataFrame,
    n_steps: int,
    server_kwargs: dict[str, Any] | None = None,
    tmp_working_directory: str | Path | None = None,
    n_dump: int | None = None,
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
        n_dump: If set, snapshots every ``n_dump`` steps are appended to ``trajectory``.

    Returns:
        A new replica dictionary with the extended ``production`` data and updated ``structure``.
    """
    final_structure, segment, generic = _nve_segment(
        replica["structure"],
        potential,
        n_steps,
        replica["settings"]["timestep"],
        server_kwargs,
        tmp_working_directory,
        n_dump,
    )
    trajectory = list(replica.get("trajectory", []))
    if n_dump is not None:
        trajectory += _stage_frames(replica["structure"], generic, "nve")
    production = {key: np.concatenate([replica["production"][key], segment[key]]) for key in segment}
    return {**replica, "production": production, "structure": final_structure, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# Analysis with STACIE
# ---------------------------------------------------------------------------


def anisotropic_pressure_components(pressure_voigt: ArrayLike) -> NDArray[np.float64]:
    """Transform a pressure tensor time series into five independent anisotropic components.

    For an isotropic liquid, each of the five components has the same autocorrelation
    function as an off-diagonal element, and they are statistically independent
    (see STACIE's documentation, "Shear Viscosity").

    Args:
        pressure_voigt: Pressure tensor, shape ``(n_steps, 6)``, order xx, yy, zz, yz, zx, xy.

    Returns:
        Array of shape ``(5, n_steps)``.
    """
    p = np.asarray(pressure_voigt, dtype=float)
    xx, yy, zz, yz, zx, xy = p.T
    return np.array([(xx - 0.5 * yy - 0.5 * zz) / np.sqrt(3), 0.5 * yy - 0.5 * zz, yz, zx, xy])


def _select(replicas: Sequence[dict[str, Any]], n_steps: int | None) -> tuple[list[dict[str, Any]], int]:
    lengths = [len(r["production"]["temperature_k"]) for r in replicas]
    if n_steps is None:
        if len(set(lengths)) != 1:
            msg = f"replicas have different production lengths {sorted(set(lengths))}; pass n_steps"
            raise ValueError(msg)
        n_steps = lengths[0]
    if min(lengths) < n_steps:
        msg = f"n_steps={n_steps} exceeds the shortest production run ({min(lengths)} steps)"
        raise ValueError(msg)
    timesteps = {r["settings"]["timestep"] for r in replicas}
    if len(timesteps) != 1:
        msg = f"replicas use different timesteps {sorted(timesteps)}"
        raise ValueError(msg)
    return list(replicas), n_steps


def stacie_viscosity(
    replicas: Sequence[dict[str, Any]],
    block_size: int,
    n_steps: int | None = None,
    model: SpectrumModel | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Estimate the shear viscosity and its uncertainty from NVE replicas with STACIE.

    Each replica contributes five independent sequences (see
    ``anisotropic_pressure_components``), block-averaged over ``block_size`` steps.
    The Green-Kubo prefactor ``V / (k_B T)`` uses each replica's own volume and the
    mean NVE temperature over all replicas.

    Args:
        replicas: Results of ``stacie_replica_md`` / ``extend_nve_production``. Only the
            ``production`` and ``settings`` entries are used.
        block_size: Number of MD steps per block average.
        n_steps: Use only the first ``n_steps`` production steps of every replica.
            ``None`` uses all steps (all replicas must then have the same length).
        model: STACIE spectrum model. Defaults to ``LorentzModel()``.
        verbose: Print STACIE's report.

    Returns:
        A dictionary with the viscosity (``viscosity``, ``viscosity_std``, Pa s), STACIE's
        diagnostics, the amount of data used, and the STACIE ``result`` object (for plotting).
    """
    replicas, n_steps = _select(replicas, n_steps)
    timestep_fs = replicas[0]["settings"]["timestep"]
    if n_steps // block_size < _MIN_BLOCKS:
        msg = f"n_steps={n_steps} gives fewer than {_MIN_BLOCKS} blocks of {block_size} steps"
        raise ValueError(msg)

    mean_temperature = float(np.mean([r["production"]["temperature_k"][:n_steps].mean() for r in replicas]))
    sequences = []
    prefactors = []
    for r in replicas:
        prod = r["production"]
        components = anisotropic_pressure_components(prod["pressure_voigt_gpa"][:n_steps]) * GPA_TO_PA
        sequences.append(block_average(components, block_size))
        volume_m3 = float(prod["volume_a3"][0]) * A3_TO_M3
        prefactors.append(np.full(len(components), volume_m3 / (KB_J_PER_K * mean_temperature)))

    spectrum = compute_spectrum(sequences, prefactors=prefactors, timestep=timestep_fs * block_size * FS_TO_S)
    result = estimate_acint(spectrum, model or LorentzModel(), verbose=verbose, uc=VISCOSITY_UNITS)

    props = result.props
    production_time_ps = n_steps * timestep_fs * 1e-3
    out = {
        "viscosity": float(result.acint),
        "viscosity_std": float(result.acint_std),
        "n_replicas": len(replicas),
        "n_sequences": 5 * len(replicas),
        "n_steps": n_steps,
        "block_size": block_size,
        "production_time_per_replica_ps": production_time_ps,
        "total_production_time_ns": production_time_ps * len(replicas) * 1e-3,
        "mean_temperature_k": mean_temperature,
        "neff": float(result.neff),
        "cost_zscore": float(props["cost_zscore"]),
        "criterion_zscore": float(props["criterion_zscore"]),
        "result": result,
    }
    if "corrtime_exp" in props:
        out["corrtime_exp_ps"] = float(props["corrtime_exp"]) * 1e12
        out["corrtime_exp_std_ps"] = float(props["corrtime_exp_std"]) * 1e12
        out["recommended_simulation_time_ps"] = float(props["exp_simulation_time"]) * 1e12
        out["recommended_block_time_ps"] = float(props["exp_block_time"]) * 1e12
    return out


def stacie_viscosity_convergence(
    replicas: Sequence[dict[str, Any]],
    block_size: int,
    n_steps_list: Sequence[int],
    n_replicas_list: Sequence[int] | None = None,
    model: SpectrumModel | None = None,
) -> list[dict[str, Any]]:
    """Viscosity estimates for subsets of the data, to show convergence with simulation time.

    For every combination of run length (first ``n_steps`` of each replica) and number of
    replicas (the first ``n_replicas``), the viscosity is re-estimated with ``stacie_viscosity``.
    Combinations that STACIE cannot fit are reported with an ``error`` entry instead of failing.

    Args:
        replicas: Results of ``stacie_replica_md`` / ``extend_nve_production``.
        block_size: Number of MD steps per block average.
        n_steps_list: Run lengths (MD steps per replica) to analyse.
        n_replicas_list: Numbers of replicas to analyse. Defaults to all replicas only.
        model: STACIE spectrum model. Defaults to ``LorentzModel()``.

    Returns:
        One dictionary per combination, as returned by ``stacie_viscosity`` but without
        the ``result`` object.
    """
    if n_replicas_list is None:
        n_replicas_list = [len(replicas)]
    rows = []
    for n_rep in n_replicas_list:
        for n_steps in n_steps_list:
            try:
                row = stacie_viscosity(replicas[:n_rep], block_size, n_steps=n_steps, model=model)
                row.pop("result")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                timestep_fs = replicas[0]["settings"]["timestep"]
                row = {
                    "n_replicas": n_rep,
                    "n_steps": n_steps,
                    "total_production_time_ns": n_steps * timestep_fs * 1e-6 * n_rep,
                    "error": str(exc),
                }
            rows.append(row)
    return rows


def nve_ensemble_checks(replicas: Sequence[dict[str, Any]], n_atoms: int) -> dict[str, Any]:
    """Validity checks for a set of NVE replicas, as recommended by STACIE.

    - Energy conservation: per replica, the drift of the total energy over the run
      (from a linear fit) and the RMS fluctuation around it, per atom, in units of k_B T.
      Both should be a small fraction of 1.
    - Ensemble: pooled over all replicas, the instantaneous temperature should have the
      canonical mean ``T`` and spread ``T * sqrt(2 / (3 N - 3))``. With few replicas this
      check is noisy.

    Args:
        replicas: Results of ``stacie_replica_md`` / ``extend_nve_production``.
        n_atoms: Number of atoms per replica.

    Returns:
        Dictionary of per-replica and pooled diagnostics.
    """
    target = replicas[0]["settings"]["temperature"]
    kt = KB_EV_PER_K * target
    drift, noise = [], []
    for r in replicas:
        energy = r["production"]["energy_tot_ev"]
        t = np.arange(len(energy))
        slope, intercept = np.polyfit(t, energy, 1)
        drift.append(float(slope * len(energy) / n_atoms / kt))
        noise.append(float(np.std(energy - (slope * t + intercept)) / n_atoms / kt))
    pooled_t = np.concatenate([r["production"]["temperature_k"] for r in replicas])
    expected_std = target * np.sqrt(2.0 / (3 * n_atoms - 3))
    return {
        "target_temperature_k": target,
        "energy_drift_per_atom_kt": drift,
        "energy_noise_per_atom_kt": noise,
        "max_abs_energy_drift_per_atom_kt": float(np.max(np.abs(drift))),
        "pooled_temperature_mean_k": float(pooled_t.mean()),
        "replica_temperature_means_k": [float(r["production"]["temperature_k"].mean()) for r in replicas],
        "pooled_temperature_std_k": float(pooled_t.std()),
        "canonical_temperature_std_k": float(expected_std),
        "pooled_over_canonical_std": float(pooled_t.std() / expected_std),
    }
