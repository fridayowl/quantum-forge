"""
Circuit Optimizer Module
========================

Advanced quantum circuit optimization with gate reduction, depth minimization,
and hardware-aware transpilation.
"""

from typing import Optional, List, Dict, Any
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CXCancellation,
    CommutativeCancellation,
    Unroller,
    Depth,
)


class CircuitOptimizer:
    """
    Optimizes quantum circuits for reduced depth, gate count, and improved fidelity.
    
    Features:
    - Multi-level optimization (0-3)
    - Gate commutation analysis
    - Single-qubit gate merging
    - Two-qubit gate cancellation
    - Circuit depth minimization
    """
    
    def __init__(self, basis_gates: Optional[List[str]] = None):
        """
        Initialize the circuit optimizer.
        
        Args:
            basis_gates: Target gate set for transpilation. 
                        Defaults to ['u1', 'u2', 'u3', 'cx']
        """
        self.basis_gates = basis_gates or ['u1', 'u2', 'u3', 'cx']
        
    def optimize(
        self,
        circuit: QuantumCircuit,
        level: int = 2,
        preserve_measurements: bool = True
    ) -> QuantumCircuit:
        """
        Optimize a quantum circuit.
        
        Args:
            circuit: Input quantum circuit
            level: Optimization level (0-3)
                  0: No optimization
                  1: Basic gate merging
                  2: Gate cancellation + merging
                  3: Advanced commutation analysis
            preserve_measurements: Keep measurement operations intact
            
        Returns:
            Optimized quantum circuit
        """
        if level == 0:
            return circuit.copy()
        
        # Build optimization pass manager based on level
        passes = []
        
        if level >= 1:
            passes.extend([
                Optimize1qGates(basis=self.basis_gates),
            ])
        
        if level >= 2:
            passes.extend([
                CXCancellation(),
                CommutativeCancellation(),
            ])
        
        if level >= 3:
            # Advanced optimization with multiple iterations
            for _ in range(3):
                passes.extend([
                    CommutativeCancellation(),
                    Optimize1qGates(basis=self.basis_gates),
                    CXCancellation(),
                ])
        
        pm = PassManager(passes)
        optimized = pm.run(circuit)
        
        return optimized
    
    def analyze_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Analyze circuit properties.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Dictionary with circuit metrics
        """
        gate_counts = circuit.count_ops()
        
        return {
            'depth': circuit.depth(),
            'size': circuit.size(),
            'num_qubits': circuit.num_qubits,
            'num_clbits': circuit.num_clbits,
            'gate_counts': gate_counts,
            'two_qubit_gates': gate_counts.get('cx', 0) + gate_counts.get('cz', 0),
            'single_qubit_gates': sum(
                count for gate, count in gate_counts.items()
                if gate not in ['cx', 'cz', 'measure', 'barrier']
            ),
        }
    
    def compare_circuits(
        self,
        original: QuantumCircuit,
        optimized: QuantumCircuit
    ) -> Dict[str, Any]:
        """
        Compare two circuits and show improvement metrics.
        
        Args:
            original: Original circuit
            optimized: Optimized circuit
            
        Returns:
            Comparison metrics
        """
        orig_metrics = self.analyze_circuit(original)
        opt_metrics = self.analyze_circuit(optimized)
        
        depth_reduction = (
            (orig_metrics['depth'] - opt_metrics['depth']) / orig_metrics['depth'] * 100
            if orig_metrics['depth'] > 0 else 0
        )
        
        size_reduction = (
            (orig_metrics['size'] - opt_metrics['size']) / orig_metrics['size'] * 100
            if orig_metrics['size'] > 0 else 0
        )
        
        return {
            'original': orig_metrics,
            'optimized': opt_metrics,
            'depth_reduction_percent': depth_reduction,
            'size_reduction_percent': size_reduction,
            'two_qubit_gate_reduction': (
                orig_metrics['two_qubit_gates'] - opt_metrics['two_qubit_gates']
            ),
        }
    
    def estimate_fidelity_improvement(
        self,
        original: QuantumCircuit,
        optimized: QuantumCircuit,
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01
    ) -> float:
        """
        Estimate fidelity improvement from optimization.
        
        Args:
            original: Original circuit
            optimized: Optimized circuit
            single_qubit_error: Error rate for single-qubit gates
            two_qubit_error: Error rate for two-qubit gates
            
        Returns:
            Estimated fidelity improvement ratio
        """
        def estimate_fidelity(circuit: QuantumCircuit) -> float:
            metrics = self.analyze_circuit(circuit)
            single_fidelity = (1 - single_qubit_error) ** metrics['single_qubit_gates']
            two_fidelity = (1 - two_qubit_error) ** metrics['two_qubit_gates']
            return single_fidelity * two_fidelity
        
        orig_fidelity = estimate_fidelity(original)
        opt_fidelity = estimate_fidelity(optimized)
        
        return opt_fidelity / orig_fidelity if orig_fidelity > 0 else 1.0
