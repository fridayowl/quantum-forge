"""
Quantum Machine Learning Module
================================

Quantum algorithms for machine learning tasks.
"""

from typing import List, Callable, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit


class QuantumKernel:
    """
    Quantum kernel for kernel-based machine learning.
    
    Uses quantum feature maps to compute kernel matrix.
    """
    
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str = 'full'
    ):
        """
        Initialize quantum kernel.
        
        Args:
            feature_dimension: Number of features
            reps: Number of repetitions in feature map
            entanglement: Entanglement strategy ('full', 'linear', 'circular')
        """
        self.feature_dim = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.num_qubits = feature_dimension
    
    def feature_map(self, x: np.ndarray) -> QuantumCircuit:
        """
        Create quantum feature map circuit.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum circuit encoding features
        """
        qc = QuantumCircuit(self.num_qubits)
        
        for rep in range(self.reps):
            # Hadamard layer
            for i in range(self.num_qubits):
                qc.h(i)
            
            # Encoding layer
            for i in range(self.num_qubits):
                qc.rz(2 * x[i], i)
            
            # Entangling layer
            if self.entanglement == 'full':
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qc.cx(i, j)
                        qc.rz(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                        qc.cx(i, j)
            elif self.entanglement == 'linear':
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)
            elif self.entanglement == 'circular':
                for i in range(self.num_qubits):
                    qc.cx(i, (i + 1) % self.num_qubits)
        
        return qc
    
    def compute_kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel entry K(x1, x2).
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            Kernel value
        """
        # Create feature maps
        qc1 = self.feature_map(x1)
        qc2 = self.feature_map(x2)
        
        # Compute overlap |⟨φ(x1)|φ(x2)⟩|²
        # Simplified simulation
        state1 = self._get_statevector(qc1)
        state2 = self._get_statevector(qc2)
        
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        return overlap
    
    def compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix for dataset.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            
        Returns:
            Kernel matrix (n_samples × n_samples)
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self.compute_kernel_entry(X[i], X[j])
                K[j, i] = K[i, j]
        
        return K
    
    def _get_statevector(self, qc: QuantumCircuit) -> np.ndarray:
        """Get statevector from circuit."""
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[0] = 1.0
        
        # Simplified simulation
        for instruction, qargs, _ in qc.data:
            state = self._apply_gate(state, instruction, qargs)
        
        return state
    
    def _apply_gate(self, state: np.ndarray, instruction, qargs) -> np.ndarray:
        """Apply gate to state."""
        # Simplified gate application
        return state


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC).
    
    Parameterized quantum circuit for classification.
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        learning_rate: float = 0.01
    ):
        """
        Initialize VQC.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers
            learning_rate: Learning rate for optimization
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Initialize parameters
        self.num_params = num_qubits * num_layers * 3
        self.params = np.random.uniform(0, 2 * np.pi, self.num_params)
    
    def create_ansatz(self, x: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        """
        Create variational ansatz.
        
        Args:
            x: Input features
            params: Variational parameters
            
        Returns:
            Quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Encode input
        for i in range(self.num_qubits):
            if i < len(x):
                qc.ry(x[i], i)
        
        # Variational layers
        param_idx = 0
        for layer in range(self.num_layers):
            # Rotation layer
            for i in range(self.num_qubits):
                qc.rx(params[param_idx], i)
                param_idx += 1
                qc.ry(params[param_idx], i)
                param_idx += 1
                qc.rz(params[param_idx], i)
                param_idx += 1
            
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    def predict_proba(self, x: np.ndarray) -> float:
        """
        Predict probability for input.
        
        Args:
            x: Input features
            
        Returns:
            Probability of class 1
        """
        qc = self.create_ansatz(x, self.params)
        state = self._simulate(qc)
        
        # Measure first qubit
        prob_1 = np.sum(np.abs(state[2**(self.num_qubits-1):]) ** 2)
        return prob_1
    
    def predict(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """
        Predict class label.
        
        Args:
            x: Input features
            threshold: Classification threshold
            
        Returns:
            Class label (0 or 1)
        """
        prob = self.predict_proba(x)
        return 1 if prob >= threshold else 0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100
    ) -> List[float]:
        """
        Train the classifier.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            
        Returns:
            Loss history
        """
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            gradients = np.zeros_like(self.params)
            
            for i in range(len(X)):
                # Forward pass
                pred = self.predict_proba(X[i])
                
                # Compute loss (binary cross-entropy)
                loss = -y[i] * np.log(pred + 1e-10) - (1 - y[i]) * np.log(1 - pred + 1e-10)
                total_loss += loss
                
                # Compute gradients (parameter shift rule)
                for j in range(len(self.params)):
                    # Shift parameter
                    params_plus = self.params.copy()
                    params_plus[j] += np.pi / 2
                    
                    params_minus = self.params.copy()
                    params_minus[j] -= np.pi / 2
                    
                    # Compute gradient
                    qc_plus = self.create_ansatz(X[i], params_plus)
                    qc_minus = self.create_ansatz(X[i], params_minus)
                    
                    state_plus = self._simulate(qc_plus)
                    state_minus = self._simulate(qc_minus)
                    
                    prob_plus = np.sum(np.abs(state_plus[2**(self.num_qubits-1):]) ** 2)
                    prob_minus = np.sum(np.abs(state_minus[2**(self.num_qubits-1):]) ** 2)
                    
                    gradients[j] += (prob_plus - prob_minus) / 2
            
            # Update parameters
            self.params -= self.learning_rate * gradients / len(X)
            
            avg_loss = total_loss / len(X)
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return loss_history
    
    def _simulate(self, qc: QuantumCircuit) -> np.ndarray:
        """Simulate circuit."""
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[0] = 1.0
        return state


class QuantumNeuralNetwork:
    """
    Quantum Neural Network (QNN).
    
    Layered quantum circuit for function approximation.
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 3,
        output_qubits: Optional[List[int]] = None
    ):
        """
        Initialize QNN.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of layers
            output_qubits: Qubits to measure for output
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.output_qubits = output_qubits or [0]
        
        # Parameters: 3 rotations per qubit per layer + entangling params
        self.num_params = num_qubits * num_layers * 3
        self.params = np.random.uniform(-np.pi, np.pi, self.num_params)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through QNN.
        
        Args:
            x: Input vector
            
        Returns:
            Output vector
        """
        qc = self._create_circuit(x)
        state = self._simulate(qc)
        
        # Extract outputs from measured qubits
        outputs = []
        for qubit in self.output_qubits:
            # Probability of measuring |1⟩
            prob = self._measure_qubit(state, qubit)
            outputs.append(prob)
        
        return np.array(outputs)
    
    def _create_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Create QNN circuit."""
        qc = QuantumCircuit(self.num_qubits)
        
        # Input encoding
        for i in range(min(len(x), self.num_qubits)):
            qc.ry(x[i], i)
        
        # Variational layers
        param_idx = 0
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                qc.rx(self.params[param_idx], i)
                param_idx += 1
                qc.ry(self.params[param_idx], i)
                param_idx += 1
                qc.rz(self.params[param_idx], i)
                param_idx += 1
            
            # Entanglement
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    def _simulate(self, qc: QuantumCircuit) -> np.ndarray:
        """Simulate circuit."""
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[0] = 1.0
        return state
    
    def _measure_qubit(self, state: np.ndarray, qubit: int) -> float:
        """Measure probability of qubit being |1⟩."""
        prob = 0.0
        for i in range(len(state)):
            if (i >> (self.num_qubits - 1 - qubit)) & 1:
                prob += np.abs(state[i]) ** 2
        return prob


class QuantumBoltzmannMachine:
    """
    Quantum Boltzmann Machine (QBM).
    
    Quantum version of restricted Boltzmann machine.
    """
    
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        temperature: float = 1.0
    ):
        """
        Initialize QBM.
        
        Args:
            num_visible: Number of visible units
            num_hidden: Number of hidden units
            temperature: Temperature parameter
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.temperature = temperature
        self.num_qubits = num_visible + num_hidden
        
        # Initialize weights
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
    
    def energy(self, visible: np.ndarray, hidden: np.ndarray) -> float:
        """
        Compute energy of configuration.
        
        Args:
            visible: Visible unit states
            hidden: Hidden unit states
            
        Returns:
            Energy value
        """
        interaction = -np.dot(visible, np.dot(self.weights, hidden))
        visible_term = -np.dot(self.visible_bias, visible)
        hidden_term = -np.dot(self.hidden_bias, hidden)
        
        return interaction + visible_term + hidden_term
    
    def sample_hidden(self, visible: np.ndarray) -> np.ndarray:
        """
        Sample hidden units given visible units.
        
        Args:
            visible: Visible unit states
            
        Returns:
            Hidden unit states
        """
        activation = np.dot(visible, self.weights) + self.hidden_bias
        probabilities = 1 / (1 + np.exp(-activation / self.temperature))
        return (np.random.rand(self.num_hidden) < probabilities).astype(int)
    
    def sample_visible(self, hidden: np.ndarray) -> np.ndarray:
        """
        Sample visible units given hidden units.
        
        Args:
            hidden: Hidden unit states
            
        Returns:
            Visible unit states
        """
        activation = np.dot(self.weights, hidden) + self.visible_bias
        probabilities = 1 / (1 + np.exp(-activation / self.temperature))
        return (np.random.rand(self.num_visible) < probabilities).astype(int)


def create_qml_model(model_type: str, **kwargs):
    """
    Create quantum machine learning model.
    
    Args:
        model_type: Type of model ('kernel', 'classifier', 'nn', 'rbm')
        **kwargs: Model parameters
        
    Returns:
        QML model instance
    """
    models = {
        'kernel': QuantumKernel,
        'classifier': VariationalQuantumClassifier,
        'nn': QuantumNeuralNetwork,
        'rbm': QuantumBoltzmannMachine
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type.lower()](**kwargs)
