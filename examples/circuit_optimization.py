"""
Example: Circuit Optimization
==============================

Demonstrates how to use the CircuitOptimizer to reduce circuit depth and gate count.
"""

from quantum_forge import CircuitOptimizer
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def create_example_circuit():
    """Create a sample quantum circuit."""
    qc = QuantumCircuit(4)
    
    # Create some redundant gates
    qc.h(0)
    qc.h(0)  # Redundant - cancels out
    qc.h(0)
    
    qc.cx(0, 1)
    qc.cx(0, 1)  # Redundant - cancels out
    qc.cx(0, 1)
    
    qc.t(0)
    qc.t(0)
    qc.t(0)
    qc.t(0)  # Four T gates = S gate
    
    qc.cx(1, 2)
    qc.cx(2, 3)
    
    return qc


def main():
    print("=" * 60)
    print("Circuit Optimization Example")
    print("=" * 60)
    
    # Create circuit
    original_circuit = create_example_circuit()
    
    # Initialize optimizer
    optimizer = CircuitOptimizer()
    
    # Analyze original circuit
    print("\n📊 Original Circuit Analysis:")
    orig_metrics = optimizer.analyze_circuit(original_circuit)
    for key, value in orig_metrics.items():
        print(f"  {key}: {value}")
    
    # Optimize at different levels
    for level in [1, 2, 3]:
        print(f"\n🔧 Optimization Level {level}:")
        optimized = optimizer.optimize(original_circuit, level=level)
        
        # Compare circuits
        comparison = optimizer.compare_circuits(original_circuit, optimized)
        
        print(f"  Depth reduction: {comparison['depth_reduction_percent']:.1f}%")
        print(f"  Size reduction: {comparison['size_reduction_percent']:.1f}%")
        print(f"  Two-qubit gates reduced: {comparison['two_qubit_gate_reduction']}")
        
        # Estimate fidelity improvement
        fidelity_ratio = optimizer.estimate_fidelity_improvement(
            original_circuit,
            optimized
        )
        print(f"  Estimated fidelity improvement: {fidelity_ratio:.4f}x")
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()
