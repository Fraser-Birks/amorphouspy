"""Tests for the STACIE-oriented viscosity MD workflow (runs LAMMPS on a tiny SHIK system)."""

from pathlib import Path

import numpy as np
import pytest
from amorphouspy.lammps.potentials.shik_potential import generate_shik_potential
from amorphouspy.properties.viscosity_stacie import extend_nve_production, stacie_replica_md

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
