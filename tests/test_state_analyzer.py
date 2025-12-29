"""
Unit tests for QuantumStateAnalyzer
"""

import unittest
import numpy as np
from quantum_forge import QuantumStateAnalyzer


class TestQuantumStateAnalyzer(unittest.TestCase):
    
    def test_purity_pure_state(self):
        """Test purity calculation for pure state."""
        # |0⟩ state
        state = np.array([1, 0])
        analyzer = QuantumStateAnalyzer(state)
        
        purity = analyzer.purity()
        self.assertAlmostEqual(purity, 1.0, places=10)
    
    def test_entropy_pure_state(self):
        """Test entropy is zero for pure states."""
        state = np.array([1, 0])
        analyzer = QuantumStateAnalyzer(state)
        
        entropy = analyzer.von_neumann_entropy()
        self.assertAlmostEqual(entropy, 0.0, places=10)
    
    def test_bell_state_entanglement(self):
        """Test entanglement entropy for Bell state."""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        analyzer = QuantumStateAnalyzer(bell_state)
        
        # Bell state should have maximum entanglement
        ent_entropy = analyzer.entanglement_entropy()
        self.assertAlmostEqual(ent_entropy, 1.0, places=5)
    
    def test_concurrence_bell_state(self):
        """Test concurrence for Bell state."""
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        analyzer = QuantumStateAnalyzer(bell_state)
        
        # Bell state should have maximum concurrence
        concurrence = analyzer.concurrence()
        self.assertAlmostEqual(concurrence, 1.0, places=5)
    
    def test_fidelity_identical_states(self):
        """Test fidelity between identical states."""
        state1 = np.array([1, 0, 0, 0])
        state2 = np.array([1, 0, 0, 0])
        
        analyzer = QuantumStateAnalyzer(state1)
        fidelity = analyzer.fidelity(state2)
        
        self.assertAlmostEqual(fidelity, 1.0, places=10)
    
    def test_fidelity_orthogonal_states(self):
        """Test fidelity between orthogonal states."""
        state1 = np.array([1, 0])
        state2 = np.array([0, 1])
        
        analyzer = QuantumStateAnalyzer(state1)
        fidelity = analyzer.fidelity(state2)
        
        self.assertAlmostEqual(fidelity, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
