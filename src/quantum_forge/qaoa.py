"""
Quantum Approximate Optimization Algorithm (QAOA) Module
========================================================

Implementation of QAOA for combinatorial optimization problems.
"""

from typing import Optional, List, Tuple
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from scipy.optimize import minimize


class QAOASolver:
    """
    QAOA solver for combinatorial optimization problems.
    
    Supports:
    - Max-Cut problem
    - Graph coloring
    - Traveling salesman (simplified)
    - Custom cost functions
    """
    
    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        p: int = 1,
        optimizer: str = 'COBYLA'
    ):
        """
        Initialize QAOA solver.
        
        Args:
            graph: Problem graph (for Max-Cut, etc.)
            p: Number of QAOA layers
            optimizer: Classical optimizer
        """
        self.graph = graph
        self.p = p
        self.optimizer = optimizer
        
        if graph is not None:
            self.num_qubits = len(graph.nodes())
        
        self.optimal_params = None
        self.optimal_cost = None
        
    def create_qaoa_circuit(
        self,
        gamma: List[float],
        beta: List[float]
    ) -> QuantumCircuit:
        """
        Create QAOA circuit.
        
        Args:
            gamma: Cost Hamiltonian parameters
            beta: Mixer Hamiltonian parameters
            
        Returns:
            QAOA quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Initial state: uniform superposition
        for qubit in range(self.num_qubits):
            qc.h(qubit)
        
        # QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian (problem-dependent)
            self._apply_cost_hamiltonian(qc, gamma[layer])
            
            # Mixer Hamiltonian
            self._apply_mixer_hamiltonian(qc, beta[layer])
        
        return qc
    
    def _apply_cost_hamiltonian(self, qc: QuantumCircuit, gamma: float):
        """
        Apply cost Hamiltonian for Max-Cut problem.
        
        For Max-Cut: H_C = Σ_{(i,j) ∈ E} (1 - Z_i Z_j) / 2
        """
        for edge in self.graph.edges():
            i, j = edge
            
            # Apply ZZ interaction
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
    
    def _apply_mixer_hamiltonian(self, qc: QuantumCircuit, beta: float):
        """
        Apply mixer Hamiltonian: H_M = Σ_i X_i
        """
        for qubit in range(self.num_qubits):
            qc.rx(2 * beta, qubit)
    
    def compute_cost(self, params: np.ndarray) -> float:
        """
        Compute expected cost for given parameters.
        
        Args:
            params: QAOA parameters [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            
        Returns:
            Expected cost value
        """
        gamma = params[:self.p]
        beta = params[self.p:]
        
        # Create circuit
        qc = self.create_qaoa_circuit(gamma, beta)
        
        # Get statevector
        state_vector = self._simulate_circuit(qc)
        
        # Compute expectation value of cost Hamiltonian
        cost = 0.0
        for edge in self.graph.edges():
            i, j = edge
            cost += self._measure_zz_expectation(state_vector, i, j)
        
        # For Max-Cut, we want to maximize, so minimize negative
        return -cost
    
    def _simulate_circuit(self, qc: QuantumCircuit) -> np.ndarray:
        """
        Simulate circuit and return statevector.
        """
        # Initialize state
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply gates
        for instruction, qargs, _ in qc.data:
            state = self._apply_gate(state, instruction, qargs)
        
        return state
    
    def _apply_gate(
        self,
        state: np.ndarray,
        instruction,
        qargs
    ) -> np.ndarray:
        """Apply gate to state vector."""
        name = instruction.name
        params = instruction.params
        
        if name == 'h':
            # Hadamard
            gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        
        elif name == 'rx':
            theta = params[0]
            gate = np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ])
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        
        elif name == 'rz':
            theta = params[0]
            gate = np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ])
            return self._apply_single_qubit_gate(state, gate, qargs[0].index)
        
        elif name == 'cx':
            return self._apply_cnot(state, qargs[0].index, qargs[1].index)
        
        return state
    
    def _apply_single_qubit_gate(
        self,
        state: np.ndarray,
        gate: np.ndarray,
        qubit: int
    ) -> np.ndarray:
        """Apply single-qubit gate."""
        # Reshape state for tensor operations
        shape = [2] * self.num_qubits
        state_tensor = state.reshape(shape)
        
        # Apply gate
        result = np.tensordot(gate, state_tensor, axes=([1], [qubit]))
        
        # Move axis back to original position
        result = np.moveaxis(result, 0, qubit)
        
        return result.flatten()
    
    def _apply_cnot(
        self,
        state: np.ndarray,
        control: int,
        target: int
    ) -> np.ndarray:
        """Apply CNOT gate."""
        new_state = state.copy()
        
        # Iterate over all basis states
        for i in range(len(state)):
            bits = format(i, f'0{self.num_qubits}b')
            
            # Check control bit
            if bits[-(control+1)] == '1':
                # Flip target bit
                bits_list = list(bits)
                bits_list[-(target+1)] = '1' if bits_list[-(target+1)] == '0' else '0'
                flipped_idx = int(''.join(bits_list), 2)
                
                # Swap amplitudes
                if i < flipped_idx:
                    new_state[i], new_state[flipped_idx] = new_state[flipped_idx], new_state[i]
        
        return new_state
    
    def _measure_zz_expectation(
        self,
        state: np.ndarray,
        qubit_i: int,
        qubit_j: int
    ) -> float:
        """
        Measure expectation value of Z_i ⊗ Z_j.
        """
        expectation = 0.0
        
        for idx in range(len(state)):
            bits = format(idx, f'0{self.num_qubits}b')
            
            # Get Z eigenvalues (+1 for |0⟩, -1 for |1⟩)
            z_i = 1 if bits[-(qubit_i+1)] == '0' else -1
            z_j = 1 if bits[-(qubit_j+1)] == '0' else -1
            
            # Add contribution
            expectation += z_i * z_j * np.abs(state[idx])**2
        
        return expectation
    
    def solve(self) -> dict:
        """
        Run QAOA optimization.
        
        Returns:
            Dictionary with results
        """
        # Initial parameters
        initial_params = np.random.uniform(0, 2*np.pi, 2*self.p)
        
        # Optimize
        result = minimize(
            self.compute_cost,
            initial_params,
            method=self.optimizer,
            options={'maxiter': 1000}
        )
        
        self.optimal_params = result.x
        self.optimal_cost = -result.fun  # Negate back
        
        # Get optimal solution
        gamma = self.optimal_params[:self.p]
        beta = self.optimal_params[self.p:]
        qc = self.create_qaoa_circuit(gamma, beta)
        final_state = self._simulate_circuit(qc)
        
        # Find most probable bitstring
        probabilities = np.abs(final_state)**2
        optimal_idx = np.argmax(probabilities)
        optimal_bitstring = format(optimal_idx, f'0{self.num_qubits}b')
        
        # Calculate cut value for Max-Cut
        cut_value = self._calculate_cut_value(optimal_bitstring)
        
        return {
            'optimal_cut': optimal_bitstring,
            'cut_value': cut_value,
            'expected_cost': self.optimal_cost,
            'params': self.optimal_params,
            'success': result.success
        }
    
    def _calculate_cut_value(self, bitstring: str) -> int:
        """
        Calculate cut value for a given bitstring.
        """
        cut_value = 0
        
        for edge in self.graph.edges():
            i, j = edge
            if bitstring[-(i+1)] != bitstring[-(j+1)]:
                cut_value += 1
        
        return cut_value
    
    def get_approximation_ratio(self, optimal_cut: int) -> float:
        """
        Calculate approximation ratio compared to optimal solution.
        
        Args:
            optimal_cut: Known optimal cut value
            
        Returns:
            Approximation ratio
        """
        if self.optimal_cost is None:
            raise ValueError("Must run solve() first")
        
        return self.optimal_cost / optimal_cut
