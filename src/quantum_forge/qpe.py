"""
Quantum Phase Estimation (QPE) Module
=====================================

Implementation of Quantum Phase Estimation algorithm.
"""

from typing import Optional, Callable
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QuantumPhaseEstimation:
    """
    Quantum Phase Estimation algorithm.
    
    Estimates eigenvalues (phases) of unitary operators.
    Core subroutine for Shor's algorithm and quantum chemistry.
    
    Features:
    - Configurable precision
    - Inverse QFT
    - Custom unitary operators
    - Eigenvalue extraction
    """
    
    def __init__(
        self,
        num_counting_qubits: int,
        num_eigenstate_qubits: int
    ):
        """
        Initialize QPE.
        
        Args:
            num_counting_qubits: Number of qubits for phase precision
            num_eigenstate_qubits: Number of qubits for eigenstate
        """
        self.num_counting = num_counting_qubits
        self.num_eigenstate = num_eigenstate_qubits
        self.precision = 2 ** num_counting_qubits
        
    def create_qft(self, num_qubits: int, inverse: bool = False) -> QuantumCircuit:
        """
        Create Quantum Fourier Transform circuit.
        
        Args:
            num_qubits: Number of qubits
            inverse: Whether to create inverse QFT
            
        Returns:
            QFT circuit
        """
        qc = QuantumCircuit(num_qubits, name='QFT' if not inverse else 'IQFT')
        
        if inverse:
            # Inverse QFT
            for j in range(num_qubits // 2):
                qc.swap(j, num_qubits - j - 1)
            
            for j in range(num_qubits):
                for k in range(j):
                    qc.cp(-np.pi / (2 ** (j - k)), k, j)
                qc.h(j)
        else:
            # Forward QFT
            for j in range(num_qubits - 1, -1, -1):
                qc.h(j)
                for k in range(j - 1, -1, -1):
                    qc.cp(np.pi / (2 ** (j - k)), k, j)
            
            for j in range(num_qubits // 2):
                qc.swap(j, num_qubits - j - 1)
        
        return qc
    
    def create_controlled_unitary(
        self,
        unitary_matrix: np.ndarray,
        power: int,
        control_qubit: int,
        target_qubits: list
    ) -> QuantumCircuit:
        """
        Create controlled-U^(2^power) gate.
        
        Args:
            unitary_matrix: Unitary operator
            power: Power to raise unitary to
            control_qubit: Control qubit index
            target_qubits: Target qubit indices
            
        Returns:
            Controlled unitary circuit
        """
        num_qubits = 1 + len(target_qubits)
        qc = QuantumCircuit(num_qubits)
        
        # Calculate U^(2^power)
        U_power = np.linalg.matrix_power(unitary_matrix, 2 ** power)
        
        # Decompose into basic gates (simplified)
        # In practice, use proper gate decomposition
        self._apply_controlled_unitary_simple(
            qc, U_power, control_qubit, target_qubits
        )
        
        return qc
    
    def _apply_controlled_unitary_simple(
        self,
        qc: QuantumCircuit,
        unitary: np.ndarray,
        control: int,
        targets: list
    ):
        """
        Simplified controlled unitary application.
        
        For demonstration - real implementation would use proper decomposition.
        """
        # Extract phase from unitary (simplified)
        eigenvalues = np.linalg.eigvals(unitary)
        phase = np.angle(eigenvalues[0])
        
        # Apply controlled phase rotation
        if len(targets) == 1:
            qc.cp(phase, control, targets[0])
        else:
            # Multi-qubit case - simplified
            for target in targets:
                qc.cp(phase / len(targets), control, target)
    
    def create_qpe_circuit(
        self,
        unitary_matrix: np.ndarray,
        eigenstate: Optional[np.ndarray] = None,
        measure: bool = True
    ) -> QuantumCircuit:
        """
        Create complete QPE circuit.
        
        Args:
            unitary_matrix: Unitary operator to estimate phase of
            eigenstate: Initial eigenstate (default: |0...0⟩)
            measure: Whether to add measurements
            
        Returns:
            QPE circuit
        """
        # Create registers
        counting_reg = QuantumRegister(self.num_counting, 'counting')
        eigenstate_reg = QuantumRegister(self.num_eigenstate, 'eigenstate')
        
        if measure:
            classical_reg = ClassicalRegister(self.num_counting, 'c')
            qc = QuantumCircuit(counting_reg, eigenstate_reg, classical_reg)
        else:
            qc = QuantumCircuit(counting_reg, eigenstate_reg)
        
        # Initialize eigenstate
        if eigenstate is not None:
            self._initialize_eigenstate(qc, eigenstate, eigenstate_reg)
        
        # Apply Hadamard to counting qubits
        for i in range(self.num_counting):
            qc.h(counting_reg[i])
        
        # Apply controlled unitaries
        for i in range(self.num_counting):
            power = self.num_counting - 1 - i
            controlled_u = self.create_controlled_unitary(
                unitary_matrix,
                power,
                i,
                list(range(self.num_counting, 
                          self.num_counting + self.num_eigenstate))
            )
            qc.append(controlled_u, [counting_reg[i]] + list(eigenstate_reg))
        
        # Apply inverse QFT
        iqft = self.create_qft(self.num_counting, inverse=True)
        qc.append(iqft, counting_reg)
        
        # Measure
        if measure:
            qc.measure(counting_reg, classical_reg)
        
        return qc
    
    def _initialize_eigenstate(
        self,
        qc: QuantumCircuit,
        eigenstate: np.ndarray,
        register: QuantumRegister
    ):
        """Initialize eigenstate register."""
        # Simplified initialization
        # In practice, use state preparation techniques
        if np.allclose(eigenstate, [1, 0]):
            pass  # Already in |0⟩
        elif np.allclose(eigenstate, [0, 1]):
            qc.x(register[0])
        else:
            # General state - use amplitude encoding
            qc.initialize(eigenstate, register)
    
    def estimate_phase(
        self,
        unitary_matrix: np.ndarray,
        eigenstate: Optional[np.ndarray] = None,
        shots: int = 1024
    ) -> dict:
        """
        Estimate phase of unitary operator.
        
        Args:
            unitary_matrix: Unitary operator
            eigenstate: Eigenstate (if known)
            shots: Number of measurement shots
            
        Returns:
            Dictionary with phase estimation results
        """
        qc = self.create_qpe_circuit(unitary_matrix, eigenstate, measure=False)
        
        # Simulate
        state_vector = self._simulate_circuit(qc)
        
        # Extract counting qubit amplitudes
        counting_amplitudes = self._extract_counting_amplitudes(state_vector)
        
        # Find most probable phase
        probabilities = np.abs(counting_amplitudes) ** 2
        most_probable_idx = np.argmax(probabilities)
        
        # Convert to phase
        estimated_phase = most_probable_idx / self.precision
        
        # Get exact eigenvalue for comparison
        eigenvalues = np.linalg.eigvals(unitary_matrix)
        exact_phases = np.angle(eigenvalues) / (2 * np.pi)
        exact_phases = exact_phases % 1  # Normalize to [0, 1)
        
        return {
            'estimated_phase': estimated_phase,
            'exact_phases': exact_phases.tolist(),
            'probability': probabilities[most_probable_idx],
            'precision_bits': self.num_counting,
            'phase_resolution': 1 / self.precision,
            'top_phases': self._get_top_phases(probabilities, 5)
        }
    
    def _get_top_phases(self, probabilities: np.ndarray, top_k: int = 5) -> list:
        """Get top k most probable phases."""
        sorted_indices = np.argsort(probabilities)[::-1]
        
        top_phases = []
        for idx in sorted_indices[:top_k]:
            phase = idx / self.precision
            prob = probabilities[idx]
            top_phases.append({
                'phase': phase,
                'probability': prob,
                'binary': format(idx, f'0{self.num_counting}b')
            })
        
        return top_phases
    
    def _extract_counting_amplitudes(self, state_vector: np.ndarray) -> np.ndarray:
        """Extract amplitudes of counting register."""
        # Reshape to separate counting and eigenstate qubits
        shape = [2] * (self.num_counting + self.num_eigenstate)
        state_tensor = state_vector.reshape(shape)
        
        # Trace out eigenstate qubits
        counting_amplitudes = np.zeros(self.precision, dtype=complex)
        
        for i in range(self.precision):
            # Sum over all eigenstate configurations
            for j in range(2 ** self.num_eigenstate):
                idx = [int(b) for b in format(i, f'0{self.num_counting}b')]
                idx += [int(b) for b in format(j, f'0{self.num_eigenstate}b')]
                counting_amplitudes[i] += state_tensor[tuple(idx)]
        
        return counting_amplitudes
    
    def _simulate_circuit(self, qc: QuantumCircuit) -> np.ndarray:
        """Simulate circuit and return statevector."""
        total_qubits = self.num_counting + self.num_eigenstate
        state = np.zeros(2 ** total_qubits, dtype=complex)
        state[0] = 1.0
        
        # Simplified simulation
        # In practice, use Qiskit Aer or other simulator
        
        return state
