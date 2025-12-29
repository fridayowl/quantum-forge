"""
Unit tests for CircuitOptimizer
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from quantum_forge import CircuitOptimizer


class TestCircuitOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = CircuitOptimizer()
    
    def test_optimization_reduces_depth(self):
        """Test that optimization reduces circuit depth."""
        # Create circuit with redundant gates
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)  # Redundant
        qc.cx(0, 1)
        qc.cx(0, 1)  # Redundant
        
        optimized = self.optimizer.optimize(qc, level=2)
        
        self.assertLessEqual(optimized.depth(), qc.depth())
    
    def test_analyze_circuit(self):
        """Test circuit analysis."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        metrics = self.optimizer.analyze_circuit(qc)
        
        self.assertEqual(metrics['num_qubits'], 3)
        self.assertGreater(metrics['depth'], 0)
        self.assertGreater(metrics['size'], 0)
    
    def test_compare_circuits(self):
        """Test circuit comparison."""
        original = QuantumCircuit(2)
        for _ in range(5):
            original.h(0)
            original.h(0)
        
        optimized = self.optimizer.optimize(original, level=3)
        comparison = self.optimizer.compare_circuits(original, optimized)
        
        self.assertIn('depth_reduction_percent', comparison)
        self.assertIn('size_reduction_percent', comparison)
    
    def test_fidelity_estimation(self):
        """Test fidelity improvement estimation."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        optimized = self.optimizer.optimize(qc, level=1)
        fidelity_ratio = self.optimizer.estimate_fidelity_improvement(
            qc, optimized
        )
        
        self.assertGreaterEqual(fidelity_ratio, 1.0)


if __name__ == '__main__':
    unittest.main()
