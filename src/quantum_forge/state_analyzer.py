"""
Quantum State Analyzer Module
==============================

Tools for analyzing and visualizing quantum states.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QuantumStateAnalyzer:
    """
    Comprehensive quantum state analysis toolkit.
    
    Features:
    - Entanglement measures
    - Fidelity calculations
    - Purity and coherence metrics
    - State visualization
    """
    
    def __init__(self, state_vector: np.ndarray):
        """
        Initialize analyzer with a quantum state.
        
        Args:
            state_vector: Quantum state vector (normalized)
        """
        self.state_vector = np.array(state_vector, dtype=complex)
        self.num_qubits = int(np.log2(len(state_vector)))
        
        # Normalize if needed
        norm = np.linalg.norm(self.state_vector)
        if not np.isclose(norm, 1.0):
            self.state_vector /= norm
        
        # Compute density matrix
        self.density_matrix = np.outer(
            self.state_vector,
            self.state_vector.conj()
        )
    
    def purity(self) -> float:
        """
        Calculate state purity: Tr(ρ²)
        
        Returns:
            Purity value (1 for pure states, <1 for mixed)
        """
        return np.real(np.trace(self.density_matrix @ self.density_matrix))
    
    def von_neumann_entropy(self) -> float:
        """
        Calculate von Neumann entropy: -Tr(ρ log ρ)
        
        Returns:
            Entropy value (0 for pure states)
        """
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        # Filter out zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def entanglement_entropy(
        self,
        subsystem_qubits: Optional[list] = None
    ) -> float:
        """
        Calculate entanglement entropy of a subsystem.
        
        Args:
            subsystem_qubits: Qubits in subsystem A (default: first half)
            
        Returns:
            Entanglement entropy
        """
        if subsystem_qubits is None:
            subsystem_qubits = list(range(self.num_qubits // 2))
        
        # Compute reduced density matrix
        rho_a = self._partial_trace(subsystem_qubits)
        
        # Calculate entropy
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def concurrence(self) -> float:
        """
        Calculate concurrence for two-qubit states.
        
        Returns:
            Concurrence value (0 to 1)
        """
        if self.num_qubits != 2:
            raise ValueError("Concurrence only defined for two-qubit states")
        
        # Pauli Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # Spin-flipped state
        spin_flip = np.kron(sigma_y, sigma_y)
        rho_tilde = spin_flip @ self.density_matrix.conj() @ spin_flip
        
        # Calculate R matrix
        R = self.density_matrix @ rho_tilde
        
        # Get eigenvalues and sort
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Concurrence
        C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
        
        return C
    
    def fidelity(self, other_state: np.ndarray) -> float:
        """
        Calculate fidelity with another state.
        
        Args:
            other_state: Another quantum state vector
            
        Returns:
            Fidelity value (0 to 1)
        """
        other_state = np.array(other_state, dtype=complex)
        other_state /= np.linalg.norm(other_state)
        
        # For pure states: F = |<ψ|φ>|²
        overlap = np.abs(np.vdot(self.state_vector, other_state))
        return overlap ** 2
    
    def _partial_trace(self, keep_qubits: list) -> np.ndarray:
        """
        Compute partial trace over complement of keep_qubits.
        
        Args:
            keep_qubits: Qubits to keep in reduced density matrix
            
        Returns:
            Reduced density matrix
        """
        all_qubits = list(range(self.num_qubits))
        trace_qubits = [q for q in all_qubits if q not in keep_qubits]
        
        # Reshape density matrix
        shape = [2] * (2 * self.num_qubits)
        rho_reshaped = self.density_matrix.reshape(shape)
        
        # Trace out unwanted qubits
        for qubit in sorted(trace_qubits, reverse=True):
            # Sum over diagonal elements of traced qubit
            rho_reshaped = np.trace(
                rho_reshaped,
                axis1=qubit,
                axis2=qubit + self.num_qubits
            )
        
        # Reshape back to matrix
        dim = 2 ** len(keep_qubits)
        return rho_reshaped.reshape(dim, dim)
    
    def plot_state_city(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot state as a 3D bar chart (state city plot).
        
        Args:
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        dim = len(self.state_vector)
        x = np.arange(dim)
        y = np.zeros(dim)
        
        # Real and imaginary parts
        real_parts = np.real(self.state_vector)
        imag_parts = np.imag(self.state_vector)
        
        width = 0.3
        
        ax.bar(x - width/2, real_parts, width, label='Real', alpha=0.8)
        ax.bar(x + width/2, imag_parts, width, label='Imaginary', alpha=0.8)
        
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Amplitude')
        ax.set_title('Quantum State City Plot')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_bloch_sphere(self):
        """
        Plot single-qubit state on Bloch sphere.
        """
        if self.num_qubits != 1:
            raise ValueError("Bloch sphere only for single-qubit states")
        
        # Calculate Bloch vector
        rho = self.density_matrix
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        # Plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(
            x_sphere, y_sphere, z_sphere,
            alpha=0.1, color='cyan'
        )
        
        # Draw axes
        ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1)
        ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1)
        ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1)
        
        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Bloch Sphere Representation')
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive state statistics.
        
        Returns:
            Dictionary with various metrics
        """
        stats = {
            'num_qubits': self.num_qubits,
            'purity': self.purity(),
            'von_neumann_entropy': self.von_neumann_entropy(),
        }
        
        if self.num_qubits >= 2:
            stats['entanglement_entropy'] = self.entanglement_entropy()
        
        if self.num_qubits == 2:
            stats['concurrence'] = self.concurrence()
        
        return stats
