"""
Example: VQE for H2 Molecule
=============================

Demonstrates using VQE to find the ground state energy of H2 molecule.
"""

from quantum_forge import VQESolver
import numpy as np


def create_h2_hamiltonian():
    """
    Create simplified H2 Hamiltonian (2 qubits).
    
    This is a simplified version. In practice, you would use
    quantum chemistry libraries like PySCF or OpenFermion.
    """
    # Simplified H2 Hamiltonian in qubit representation
    # Real values would come from molecular orbital calculations
    
    # Identity and Pauli matrices
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    # Build 2-qubit Hamiltonian
    # H = c0*I⊗I + c1*Z⊗I + c2*I⊗Z + c3*Z⊗Z + c4*X⊗X
    
    c0 = -0.8105  # Constant term
    c1 = 0.1721   # Z on qubit 0
    c2 = 0.1721   # Z on qubit 1
    c3 = -0.2228  # ZZ interaction
    c4 = 0.1686   # XX interaction
    
    H = (c0 * np.kron(I, I) +
         c1 * np.kron(Z, I) +
         c2 * np.kron(I, Z) +
         c3 * np.kron(Z, Z) +
         c4 * np.kron(X, X))
    
    return H


def main():
    print("=" * 60)
    print("VQE for H2 Molecule Ground State")
    print("=" * 60)
    
    # Create Hamiltonian
    print("\n🔬 Creating H2 Hamiltonian...")
    hamiltonian = create_h2_hamiltonian()
    
    # Exact diagonalization for comparison
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    exact_ground_state = eigenvalues[0]
    print(f"  Exact ground state energy: {exact_ground_state:.6f} Ha")
    
    # Run VQE with different ansatze
    for ansatz_type in ['hardware_efficient', 'uccsd']:
        print(f"\n🚀 Running VQE with {ansatz_type} ansatz...")
        
        vqe = VQESolver(
            hamiltonian=hamiltonian,
            num_qubits=2,
            ansatz=ansatz_type,
            optimizer='COBYLA',
            max_iterations=500
        )
        
        result = vqe.solve()
        
        print(f"  VQE ground state energy: {result['energy']:.6f} Ha")
        print(f"  Error: {abs(result['energy'] - exact_ground_state):.6f} Ha")
        print(f"  Iterations: {result['num_iterations']}")
        print(f"  Success: {result['success']}")
        
        # Calculate accuracy
        accuracy = (1 - abs(result['energy'] - exact_ground_state) / abs(exact_ground_state)) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
    
    print("\n✅ VQE simulation complete!")


if __name__ == "__main__":
    main()
