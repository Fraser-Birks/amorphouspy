"""Tests for the STACIE-oriented viscosity MD workflow (runs LAMMPS on a tiny SHIK system)."""

from pathlib import Path

import numpy as np
import pytest
from amorphouspy.lammps.potentials.shik_potential import generate_shik_potential
from amorphouspy.properties.viscosity_stacie import (
    anisotropic_pressure_components,
    extend_nve_production,
    nve_ensemble_checks,
    stacie_replica_md,
    stacie_viscosity,
    stacie_viscosity_convergence,
)

from amorphouspy import get_ase_structure, get_structure_dict


@pytest.fixture(scope="module")
def replica(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, object, Path]:
    """Run one tiny replica shared by all tests in this module."""
    workdir = tmp_path_factory.mktemp("stacie_md")
    atoms_dict = get_structure_dict({"CaO": 25, "MgO": 25, "SiO2": 50}, target_atoms=100, random_seed=1)
    structure = get_ase_structure(atoms_dict)
    potential = generate_shik_potential(atoms_dict, output_dir=str(workdir / "tables"))
    result = stacie_replica_md(
        structure,
        potential,
        temperature=3000.0,
        pressure=0.1,
        equilibration_steps=500,
        production_steps=200,
        seed=7,
        melt_equilibration_steps=200,
        cooling_rate=1e15,
        tmp_working_directory=workdir,
    )
    return result, potential, workdir


def test_production_has_one_sample_per_step(replica: tuple[dict, object, Path]) -> None:
    """N production steps give exactly N samples (the step-0 frame is dropped)."""
    result, _, _ = replica
    production = result["production"]
    assert production["pressure_voigt_gpa"].shape == (200, 6)
    for key in ["temperature_k", "energy_tot_ev", "volume_a3"]:
        assert production[key].shape == (200,)


def test_production_is_nve_without_velocity_reset(replica: tuple[dict, object, Path]) -> None:
    """Production runs at fixed volume, conserves energy and never re-creates velocities."""
    result, _, workdir = replica
    production = result["production"]
    assert np.ptp(production["volume_a3"]) == 0.0
    # Energy is conserved to a small fraction of kT per atom.
    kt_ev = 8.617333e-5 * 3000.0
    assert np.ptp(production["energy_tot_ev"]) / 100 < 0.01 * kt_ev
    # The only velocity initialisation is in the melt-quench; equilibration and NVE keep velocities.
    lines = [line for path in workdir.glob("*/lmp.in") for line in path.read_text().splitlines()]
    velocity_lines = [line for line in lines if line.startswith("velocity")]
    assert len(velocity_lines) == 1


def test_seeds_are_valid_for_lammps(replica: tuple[dict, object, Path]) -> None:
    """Derived seeds are in the range accepted by LAMMPS."""
    settings = replica[0]["settings"]
    for key in ["melt_seed", "langevin_seed"]:
        assert 1 <= settings[key] <= 900_000_000


def test_extension_is_continuous(replica: tuple[dict, object, Path]) -> None:
    """An extension appends to the same continuous NVE trajectory."""
    result, potential, workdir = replica
    extended = extend_nve_production(result, potential, 100, tmp_working_directory=workdir)
    production = extended["production"]
    assert production["pressure_voigt_gpa"].shape == (300, 6)
    np.testing.assert_array_equal(production["pressure_voigt_gpa"][:200], result["production"]["pressure_voigt_gpa"])
    # No jump at the join: the energy step across it is like any other step.
    steps = np.abs(np.diff(production["energy_tot_ev"]))
    assert steps[199] < 5 * np.median(steps) + 1e-6
    assert production["volume_a3"][200] == production["volume_a3"][199]


# ---------------------------------------------------------------------------
# Analysis on synthetic data with a known answer (no LAMMPS)
# ---------------------------------------------------------------------------

_T = 3000.0
_V_A3 = 1500.0
_SIGMA_GPA = 1.0
_TAU_FS = 50.0
_TIMESTEP_FS = 1.0


def _voigt_from_anisotropic(aniso: np.ndarray) -> np.ndarray:
    """Inverse of anisotropic_pressure_components for a traceless tensor; (5, n) -> (n, 6)."""
    a1, a2, yz, zx, xy = aniso
    xx = 2 * a1 / np.sqrt(3)
    yy = (-xx + 2 * a2) / 2
    zz = (-xx - 2 * a2) / 2
    return np.stack([xx, yy, zz, yz, zx, xy], axis=1)


def _ou_replicas(n_replicas: int, n_steps: int, seed: int = 0) -> list[dict]:
    """Replicas whose five anisotropic components are independent Ornstein-Uhlenbeck processes."""
    rng = np.random.default_rng(seed)
    a = np.exp(-_TIMESTEP_FS / _TAU_FS)
    replicas = []
    for _ in range(n_replicas):
        x = np.empty((5, n_steps))
        x[:, 0] = rng.normal(0, _SIGMA_GPA, 5)
        noise = rng.normal(0, _SIGMA_GPA * np.sqrt(1 - a**2), (5, n_steps))
        for n in range(1, n_steps):
            x[:, n] = a * x[:, n - 1] + noise[:, n]
        replicas.append(
            {
                "production": {
                    "pressure_voigt_gpa": _voigt_from_anisotropic(x),
                    "temperature_k": np.full(n_steps, _T),
                    "energy_tot_ev": np.zeros(n_steps),
                    "volume_a3": np.full(n_steps, _V_A3),
                },
                "settings": {"temperature": _T, "timestep": _TIMESTEP_FS},
            }
        )
    return replicas


def _exact_ou_viscosity() -> float:
    """Discrete Green-Kubo integral of sigma^2 a^|n| with prefactor V/(kB T), in Pa s."""
    a = np.exp(-_TIMESTEP_FS / _TAU_FS)
    prefactor = _V_A3 * 1e-30 / (1.380649e-23 * _T)
    h = _TIMESTEP_FS * 1e-15
    return prefactor * h * (_SIGMA_GPA * 1e9) ** 2 / 2 * (1 + a) / (1 - a)


def test_anisotropic_components_round_trip() -> None:
    """The five components of a traceless tensor are recovered exactly."""
    aniso = np.random.default_rng(1).normal(size=(5, 10))
    np.testing.assert_allclose(anisotropic_pressure_components(_voigt_from_anisotropic(aniso)), aniso)


def test_anisotropic_components_ignore_isotropic_pressure() -> None:
    """Adding a hydrostatic pressure does not change the anisotropic components."""
    voigt = np.random.default_rng(2).normal(size=(10, 6))
    shifted = voigt.copy()
    shifted[:, :3] += 3.7
    np.testing.assert_allclose(anisotropic_pressure_components(shifted), anisotropic_pressure_components(voigt))


@pytest.mark.parametrize("block_size", [1, 5])
def test_stacie_recovers_known_viscosity(block_size: int) -> None:
    """STACIE recovers the exact integral of synthetic OU stresses within its error bar."""
    out = stacie_viscosity(_ou_replicas(n_replicas=10, n_steps=4000), block_size=block_size)
    exact = _exact_ou_viscosity()
    assert abs(out["viscosity"] - exact) < 4 * out["viscosity_std"]
    assert out["viscosity_std"] / exact < 0.15
    assert out["n_sequences"] == 50
    np.testing.assert_allclose(out["corrtime_exp_ps"], _TAU_FS * 1e-3, rtol=0.3)


def test_convergence_rows_cover_requested_grid() -> None:
    """One row per (n_replicas, n_steps); errors shrink with more data."""
    replicas = _ou_replicas(n_replicas=6, n_steps=4000, seed=3)
    rows = stacie_viscosity_convergence(replicas, block_size=5, n_steps_list=[1000, 4000], n_replicas_list=[2, 6])
    assert [(r["n_replicas"], r["n_steps"]) for r in rows] == [(2, 1000), (2, 4000), (6, 1000), (6, 4000)]
    assert all("error" not in r for r in rows)
    assert rows[-1]["viscosity_std"] < rows[0]["viscosity_std"]
    assert "result" not in rows[0]


def test_select_rejects_too_long_n_steps() -> None:
    """Asking for more steps than were run is an error, not a silent truncation."""
    with pytest.raises(ValueError, match="exceeds the shortest"):
        stacie_viscosity(_ou_replicas(n_replicas=2, n_steps=500), block_size=1, n_steps=600)


def test_nve_checks_flag_drift() -> None:
    """A linear energy drift of 0.1 kT per atom over the run is reported as such."""
    replicas = _ou_replicas(n_replicas=2, n_steps=1000)
    n_atoms = 100
    kt = 8.617333262e-5 * _T
    replicas[1]["production"]["energy_tot_ev"] = np.linspace(0, 0.1 * kt * n_atoms, 1000)
    checks = nve_ensemble_checks(replicas, n_atoms=n_atoms)
    assert checks["energy_drift_per_atom_kt"][0] == pytest.approx(0.0, abs=1e-12)
    assert checks["energy_drift_per_atom_kt"][1] == pytest.approx(0.1, rel=0.01)
    assert checks["max_abs_energy_drift_per_atom_kt"] == pytest.approx(0.1, rel=0.01)
