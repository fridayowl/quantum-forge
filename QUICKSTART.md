# Quick Start Guide

## Installation

```bash
pip install quantum-forge
```

Or from source:

```bash
git clone https://github.com/fridayowl/quantum-forge.git
cd quantum-forge
pip install -e .
```

## 5-Minute Tutorial

### 1. Optimize a Quantum Circuit

```python
from quantum_forge import CircuitOptimizer
from qiskit import QuantumCircuit

# Create a circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Optimize it
optimizer = CircuitOptimizer()
optimized_qc = optimizer.optimize(qc, level=3)

print(f"Original depth: {qc.depth()}")
print(f"Optimized depth: {optimized_qc.depth()}")
```

### 2. Analyze Quantum States

```python
from quantum_forge import QuantumStateAnalyzer
import numpy as np

# Bell state
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)

analyzer = QuantumStateAnalyzer(bell)
print(f"Entanglement: {analyzer.entanglement_entropy():.3f}")
print(f"Concurrence: {analyzer.concurrence():.3f}")
```

### 3. Run VQE

```python
from quantum_forge import VQESolver
import numpy as np

# Simple 2-qubit Hamiltonian
H = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

vqe = VQESolver(hamiltonian=H, num_qubits=2)
result = vqe.solve()
print(f"Ground state energy: {result['energy']:.4f}")
```

### 4. Solve Max-Cut with QAOA

```python
from quantum_forge import QAOASolver
import networkx as nx

# Create a graph
G = nx.Graph()
G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])

# Solve with QAOA
qaoa = QAOASolver(graph=G, p=2)
result = qaoa.solve()
print(f"Max cut: {result['cut_value']}")
```

### 5. Build Noise Models

```python
from quantum_forge import NoiseModelBuilder

noise = NoiseModelBuilder() \
    .add_depolarizing_error(0.001, ['u1', 'u2', 'u3']) \
    .add_thermal_relaxation(t1=50e-6, t2=70e-6) \
    .add_readout_error([[0.95, 0.05], [0.1, 0.9]]) \
    .build()
```

## Next Steps

- Check out the [examples/](examples/) directory for more detailed examples
- Read the full [README.md](README.md) for comprehensive documentation
- Explore the API documentation
- Join our community discussions

## Common Use Cases

### Quantum Chemistry
Use VQE to find molecular ground states:
```python
# See examples/vqe_h2_molecule.py
```

### Optimization Problems
Solve combinatorial optimization with QAOA:
```python
# See examples/qaoa_max_cut.py
```

### Circuit Design
Optimize circuits for real quantum hardware:
```python
# See examples/circuit_optimization.py
```

## Getting Help

- 📖 [Documentation](https://quantum-forge.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/fridayowl/quantum-forge/discussions)
- 🐛 [Issue Tracker](https://github.com/fridayowl/quantum-forge/issues)

Happy quantum computing! 🚀⚛️
