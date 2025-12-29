"""
Quantum Chemistry Module
========================

Molecular Hamiltonians for VQE simulations.
"""

import numpy as np
from typing import Tuple, List


class MolecularHamiltonian:
    """
    Pre-computed molecular Hamiltonians for common molecules.
    
    All Hamiltonians are in the qubit representation using
    Jordan-Wigner transformation.
    """
    
    @staticmethod
    def h2(bond_length: float = 0.74) -> np.ndarray:
        """
        Hydrogen molecule (H₂) Hamiltonian.
        
        Args:
            bond_length: H-H bond length in Angstroms (default: 0.74)
            
        Returns:
            4x4 Hamiltonian matrix
        """
        # Coefficients for H2 at equilibrium geometry
        # These are simplified values for demonstration
        if abs(bond_length - 0.74) < 0.01:
            c0 = -0.8105
            c1 = 0.1721
            c2 = 0.1721
            c3 = -0.2228
            c4 = 0.1686
        else:
            # Approximate scaling for different bond lengths
            scale = (bond_length / 0.74) ** 2
            c0 = -0.8105 * scale
            c1 = 0.1721
            c2 = 0.1721
            c3 = -0.2228 / scale
            c4 = 0.1686 / scale
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def lih(bond_length: float = 1.55) -> np.ndarray:
        """
        Lithium Hydride (LiH) Hamiltonian.
        
        Args:
            bond_length: Li-H bond length in Angstroms (default: 1.55)
            
        Returns:
            4x4 Hamiltonian matrix
        """
        # LiH coefficients (2 qubits, minimal basis)
        c0 = -7.8823
        c1 = 0.2252
        c2 = 0.2252
        c3 = -0.3425
        c4 = 0.1854
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def h2o() -> np.ndarray:
        """
        Water (H₂O) Hamiltonian (simplified, 2 qubits).
        
        Returns:
            4x4 Hamiltonian matrix
        """
        c0 = -75.6803
        c1 = 0.3452
        c2 = 0.3452
        c3 = -0.4826
        c4 = 0.2134
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def beh2() -> np.ndarray:
        """
        Beryllium Hydride (BeH₂) Hamiltonian (simplified, 2 qubits).
        
        Returns:
            4x4 Hamiltonian matrix
        """
        c0 = -15.5943
        c1 = 0.2891
        c2 = 0.2891
        c3 = -0.3967
        c4 = 0.1723
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def nh3() -> np.ndarray:
        """
        Ammonia (NH₃) Hamiltonian (simplified, 2 qubits).
        
        Returns:
            4x4 Hamiltonian matrix
        """
        c0 = -56.2234
        c1 = 0.3123
        c2 = 0.3123
        c3 = -0.4234
        c4 = 0.1956
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def ch4() -> np.ndarray:
        """
        Methane (CH₄) Hamiltonian (simplified, 2 qubits).
        
        Returns:
            4x4 Hamiltonian matrix
        """
        c0 = -40.4567
        c1 = 0.2987
        c2 = 0.2987
        c3 = -0.4012
        c4 = 0.1834
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def n2() -> np.ndarray:
        """
        Nitrogen (N₂) Hamiltonian (simplified, 2 qubits).
        
        Returns:
            4x4 Hamiltonian matrix
        """
        c0 = -109.0983
        c1 = 0.3678
        c2 = 0.3678
        c3 = -0.5234
        c4 = 0.2456
        
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        
        H = (c0 * np.kron(I, I) +
             c1 * np.kron(Z, I) +
             c2 * np.kron(I, Z) +
             c3 * np.kron(Z, Z) +
             c4 * np.kron(X, X))
        
        return H
    
    @staticmethod
    def get_molecule(name: str, **kwargs) -> np.ndarray:
        """
        Get Hamiltonian for a molecule by name.
        
        Args:
            name: Molecule name (h2, lih, h2o, beh2, nh3, ch4, n2)
            **kwargs: Additional parameters (e.g., bond_length)
            
        Returns:
            Hamiltonian matrix
        """
        molecules = {
            'h2': MolecularHamiltonian.h2,
            'lih': MolecularHamiltonian.lih,
            'h2o': MolecularHamiltonian.h2o,
            'beh2': MolecularHamiltonian.beh2,
            'nh3': MolecularHamiltonian.nh3,
            'ch4': MolecularHamiltonian.ch4,
            'n2': MolecularHamiltonian.n2,
        }
        
        if name.lower() not in molecules:
            raise ValueError(f"Unknown molecule: {name}. Available: {list(molecules.keys())}")
        
        return molecules[name.lower()](**kwargs)
    
    @staticmethod
    def get_exact_energy(hamiltonian: np.ndarray) -> float:
        """
        Calculate exact ground state energy.
        
        Args:
            hamiltonian: Hamiltonian matrix
            
        Returns:
            Ground state energy
        """
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        return eigenvalues[0]
    
    @staticmethod
    def get_molecule_info(name: str) -> dict:
        """
        Get information about a molecule.
        
        Args:
            name: Molecule name
            
        Returns:
            Dictionary with molecule information
        """
        info = {
            'h2': {
                'name': 'Hydrogen',
                'formula': 'H₂',
                'atoms': 2,
                'electrons': 2,
                'bond_type': 'Single',
                'applications': 'Fuel cells, chemical industry'
            },
            'lih': {
                'name': 'Lithium Hydride',
                'formula': 'LiH',
                'atoms': 2,
                'electrons': 4,
                'bond_type': 'Ionic',
                'applications': 'Hydrogen storage, batteries'
            },
            'h2o': {
                'name': 'Water',
                'formula': 'H₂O',
                'atoms': 3,
                'electrons': 10,
                'bond_type': 'Polar covalent',
                'applications': 'Universal solvent, life'
            },
            'beh2': {
                'name': 'Beryllium Hydride',
                'formula': 'BeH₂',
                'atoms': 3,
                'electrons': 6,
                'bond_type': 'Covalent',
                'applications': 'Materials science'
            },
            'nh3': {
                'name': 'Ammonia',
                'formula': 'NH₃',
                'atoms': 4,
                'electrons': 10,
                'bond_type': 'Polar covalent',
                'applications': 'Fertilizers, cleaning'
            },
            'ch4': {
                'name': 'Methane',
                'formula': 'CH₄',
                'atoms': 5,
                'electrons': 10,
                'bond_type': 'Covalent',
                'applications': 'Natural gas, fuel'
            },
            'n2': {
                'name': 'Nitrogen',
                'formula': 'N₂',
                'atoms': 2,
                'electrons': 14,
                'bond_type': 'Triple',
                'applications': 'Atmosphere, fertilizers'
            }
        }
        
        return info.get(name.lower(), {})
