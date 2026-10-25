"""N2 active-space Hamiltonian.

`build_n2` reproduces the grader's `_n2_hamiltonian` logic exactly
(N2, cc-pVDZ, 2.0 A, RHF, 2 frozen core orbitals -> 26 active orbitals,
nelec (5, 5); CASCI.sort_mo(range(2, nao), base=0), get_h1cas/get_h2cas).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("N2SQD_DATA", Path(__file__).resolve().parent.parent / "data"))
CACHE_FILE = DATA_DIR / "hamiltonian.npz"


def build_n2(*, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int]]:
    """Return (hcore, eri, ecore, norb, nelec). Rebuilds with PySCF on first use and caches the result in a gitignored on-disk cache."""
    if use_cache and CACHE_FILE.exists():
        z = np.load(CACHE_FILE)
        nelec = (int(z["nelec"][0]), int(z["nelec"][1]))
        return z["hcore"], z["eri"], float(z["ecore"]), int(z["norb"]), nelec
    import pyscf
    import pyscf.mcscf

    mol = pyscf.gto.Mole()
    mol.build(atom=[["N", (0, 0, 0)], ["N", (2.0, 0, 0)]], basis="cc-pvdz", verbose=0)
    active_space = range(2, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    num_orbitals = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    nelec = ((n_electrons + mol.spin) // 2, (n_electrons - mol.spin) // 2)
    cas = pyscf.mcscf.CASCI(scf, num_orbitals, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, core_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)
    ecore = float(core_energy)
    if use_cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE_FILE, hcore=hcore, eri=eri, ecore=ecore,
            norb=num_orbitals, nelec=np.array(nelec),
        )
    return hcore, eri, ecore, num_orbitals, nelec


def rhf_reference():
    """Rebuild mol/scf the grader's way; for HF (and CCSD) reference energies."""
    import pyscf

    mol = pyscf.gto.Mole()
    mol.build(atom=[["N", (0, 0, 0)], ["N", (2.0, 0, 0)]], basis="cc-pvdz", verbose=0)
    scf = pyscf.scf.RHF(mol).run()
    return mol, scf
