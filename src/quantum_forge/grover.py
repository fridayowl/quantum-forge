"""
Grover's Search Algorithm Module
=================================

Implementation of Grover's algorithm for unstructured search.
"""

from typing import List, Callable, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class GroverSearch:
    """
    Grover's algorithm for searching unstructured databases.
    
    Provides quadratic speedup over classical search: O(√N) vs O(N)
    
    Features:
    - Customizable oracle functions
    - Amplitude amplification
    - Optimal iteration calculation
    - Multi-solution search
    """
    
    def __init__(self, num_qubits: int, marked_states: Optional[List[str]] = None):
        """
        Initialize Grover's search.
        
        Args:
            num_qubits: Number of qubits (search space size = 2^n)
            marked_states: List of marked states in binary (e.g., ['101', '110'])
        """
        self.num_qubits = num_qubits
        self.marked_states = marked_states or []
        self.search_space_size = 2 ** num_qubits
        
    def create_oracle(self, marked_states: Optional[List[str]] = None) -> QuantumCircuit:
        """
        Create oracle that marks target states.
        
        Args:
            marked_states: States to mark (uses self.marked_states if None)
            
        Returns:
            Oracle quantum circuit
        """
        if marked_states is None:
            marked_states = self.marked_states
        
        qc = QuantumCircuit(self.num_qubits, name='Oracle')
        
        for state in marked_states:
            # Mark state by flipping phase
            self._mark_state(qc, state)
        
        return qc
    
    def _mark_state(self, qc: QuantumCircuit, state: str):
        """
        Mark a specific state by flipping its phase.
        
        Uses multi-controlled Z gate.
        """
        # Apply X gates to flip 0s to 1s
        for i, bit in enumerate(state):
            if bit == '0':
                qc.x(i)
        
        # Multi-controlled Z gate
        if self.num_qubits == 1:
            qc.z(0)
        elif self.num_qubits == 2:
            qc.cz(0, 1)
        else:
            # Multi-controlled Z using ancilla-free method
            self._multi_controlled_z(qc, list(range(self.num_qubits)))
        
        # Undo X gates
        for i, bit in enumerate(state):
            if bit == '0':
                qc.x(i)
    
    def _multi_controlled_z(self, qc: QuantumCircuit, control_qubits: List[int]):
        """
        Implement multi-controlled Z gate.
        """
        if len(control_qubits) == 1:
            qc.z(control_qubits[0])
        elif len(control_qubits) == 2:
            qc.cz(control_qubits[0], control_qubits[1])
        else:
            # Use H-CX-H trick: CZ = H-CX-H
            target = control_qubits[-1]
            controls = control_qubits[:-1]
            
            qc.h(target)
            # Multi-controlled X
            if len(controls) == 1:
                qc.cx(controls[0], target)
            else:
                qc.mcx(controls, target)
            qc.h(target)
    
    def create_diffusion_operator(self) -> QuantumCircuit:
        """
        Create Grover diffusion operator (inversion about average).
        
        Returns:
            Diffusion operator circuit
        """
        qc = QuantumCircuit(self.num_qubits, name='Diffusion')
        
        # Apply H gates
        for qubit in range(self.num_qubits):
            qc.h(qubit)
        
        # Apply X gates
        for qubit in range(self.num_qubits):
            qc.x(qubit)
        
        # Multi-controlled Z
        self._multi_controlled_z(qc, list(range(self.num_qubits)))
        
        # Apply X gates
        for qubit in range(self.num_qubits):
            qc.x(qubit)
        
        # Apply H gates
        for qubit in range(self.num_qubits):
            qc.h(qubit)
        
        return qc
    
    def calculate_optimal_iterations(self, num_solutions: int = 1) -> int:
        """
        Calculate optimal number of Grover iterations.
        
        Args:
            num_solutions: Number of marked states
            
        Returns:
            Optimal number of iterations
        """
        N = self.search_space_size
        M = num_solutions
        
        # Optimal iterations ≈ π/4 * √(N/M)
        optimal = int(np.pi / 4 * np.sqrt(N / M))
        return max(1, optimal)
    
    def create_grover_circuit(
        self,
        num_iterations: Optional[int] = None,
        measure: bool = True
    ) -> QuantumCircuit:
        """
        Create complete Grover's algorithm circuit.
        
        Args:
            num_iterations: Number of Grover iterations (auto-calculated if None)
            measure: Whether to add measurement gates
            
        Returns:
            Complete Grover circuit
        """
        if num_iterations is None:
            num_iterations = self.calculate_optimal_iterations(
                len(self.marked_states)
            )
        
        # Create circuit
        qr = QuantumRegister(self.num_qubits, 'q')
        if measure:
            cr = ClassicalRegister(self.num_qubits, 'c')
            qc = QuantumCircuit(qr, cr)
        else:
            qc = QuantumCircuit(qr)
        
        # Initialize superposition
        for qubit in range(self.num_qubits):
            qc.h(qubit)
        
        # Grover iterations
        oracle = self.create_oracle()
        diffusion = self.create_diffusion_operator()
        
        for _ in range(num_iterations):
            qc.append(oracle, range(self.num_qubits))
            qc.append(diffusion, range(self.num_qubits))
        
        # Measure
        if measure:
            qc.measure(qr, cr)
        
        return qc
    
    def simulate(self, num_iterations: Optional[int] = None) -> dict:
        """
        Simulate Grover's algorithm and return results.
        
        Args:
            num_iterations: Number of iterations
            
        Returns:
            Dictionary with simulation results
        """
        qc = self.create_grover_circuit(num_iterations, measure=False)
        
        # Get statevector
        state_vector = self._simulate_circuit(qc)
        
        # Calculate probabilities
        probabilities = np.abs(state_vector) ** 2
        
        # Find most probable states
        sorted_indices = np.argsort(probabilities)[::-1]
        top_states = []
        
        for idx in sorted_indices[:5]:  # Top 5 states
            state = format(idx, f'0{self.num_qubits}b')
            prob = probabilities[idx]
            top_states.append({
                'state': state,
                'probability': prob,
                'is_marked': state in self.marked_states
            })
        
        # Calculate success probability
        success_prob = sum(
            probabilities[int(state, 2)] 
            for state in self.marked_states
        )
        
        return {
            'num_iterations': num_iterations or self.calculate_optimal_iterations(
                len(self.marked_states)
            ),
            'top_states': top_states,
            'success_probability': success_prob,
            'marked_states': self.marked_states,
            'probabilities': probabilities
        }
    
    def _simulate_circuit(self, qc: QuantumCircuit) -> np.ndarray:
        """Simulate circuit and return statevector."""
        # Initialize state
        state = np.zeros(self.search_space_size, dtype=complex)
        state[0] = 1.0
        
        # Apply gates
        for instruction, qargs, _ in qc.data:
            if hasattr(instruction, 'definition') and instruction.definition:
                # Decompose composite gates
                for sub_inst, sub_qargs, _ in instruction.definition.data:
                    state = self._apply_gate(state, sub_inst, sub_qargs)
            else:
                state = self._apply_gate(state, instruction, qargs)
        
        return state
    
    def _apply_gate(self, state: np.ndarray, instruction, qargs) -> np.ndarray:
        """Apply gate to state vector."""
        name = instruction.name
        
        if name == 'h':
            gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        elif name == 'x':
            gate = np.array([[0, 1], [1, 0]])
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        elif name == 'z':
            gate = np.array([[1, 0], [0, -1]])
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        elif name == 'cx':
            return self._apply_cnot(state, qargs[0].index, qargs[1].index)
        elif name == 'cz':
            return self._apply_cz(state, qargs[0].index, qargs[1].index)
        
        return state
    
    def _apply_single_qubit_gate(
        self,
        state: np.ndarray,
        gate: np.ndarray,
        qubit: int
    ) -> np.ndarray:
        """Apply single-qubit gate."""
        shape = [2] * self.num_qubits
        state_tensor = state.reshape(shape)
        result = np.tensordot(gate, state_tensor, axes=([1], [qubit]))
        result = np.moveaxis(result, 0, qubit)
        return result.flatten()
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        new_state = state.copy()
        
        for i in range(len(state)):
            bits = format(i, f'0{self.num_qubits}b')
            if bits[-(control+1)] == '1':
                bits_list = list(bits)
                bits_list[-(target+1)] = '1' if bits_list[-(target+1)] == '0' else '0'
                flipped_idx = int(''.join(bits_list), 2)
                if i < flipped_idx:
                    new_state[i], new_state[flipped_idx] = new_state[flipped_idx], new_state[i]
        
        return new_state
    
    def _apply_cz(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CZ gate."""
        new_state = state.copy()
        
        for i in range(len(state)):
            bits = format(i, f'0{self.num_qubits}b')
            if bits[-(control+1)] == '1' and bits[-(target+1)] == '1':
                new_state[i] *= -1
        
        return new_state
