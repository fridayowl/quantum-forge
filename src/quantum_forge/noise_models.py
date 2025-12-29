"""
Noise Model Builder Module
==========================

Tools for creating and managing quantum noise models.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class NoiseModelBuilder:
    """
    Builder for creating custom quantum noise models.
    
    Supports:
    - Depolarizing noise
    - Thermal relaxation (T1, T2)
    - Amplitude damping
    - Phase damping
    - Readout errors
    """
    
    def __init__(self):
        """Initialize noise model builder."""
        self.noise_channels = []
        self.readout_errors = {}
        
    def add_depolarizing_error(
        self,
        error_rate: float,
        gates: List[str],
        qubits: Optional[List[int]] = None
    ) -> 'NoiseModelBuilder':
        """
        Add depolarizing error channel.
        
        Args:
            error_rate: Probability of error (0 to 1)
            gates: List of gate names to apply noise to
            qubits: Specific qubits (None for all)
            
        Returns:
            Self for chaining
        """
        self.noise_channels.append({
            'type': 'depolarizing',
            'error_rate': error_rate,
            'gates': gates,
            'qubits': qubits
        })
        return self
    
    def add_thermal_relaxation(
        self,
        t1: float,
        t2: float,
        gate_time: float = 50e-9,
        qubits: Optional[List[int]] = None
    ) -> 'NoiseModelBuilder':
        """
        Add thermal relaxation noise.
        
        Args:
            t1: T1 relaxation time (seconds)
            t2: T2 dephasing time (seconds)
            gate_time: Gate execution time (seconds)
            qubits: Specific qubits (None for all)
            
        Returns:
            Self for chaining
        """
        self.noise_channels.append({
            'type': 'thermal_relaxation',
            't1': t1,
            't2': t2,
            'gate_time': gate_time,
            'qubits': qubits
        })
        return self
    
    def add_amplitude_damping(
        self,
        gamma: float,
        qubits: Optional[List[int]] = None
    ) -> 'NoiseModelBuilder':
        """
        Add amplitude damping channel.
        
        Args:
            gamma: Damping parameter (0 to 1)
            qubits: Specific qubits (None for all)
            
        Returns:
            Self for chaining
        """
        self.noise_channels.append({
            'type': 'amplitude_damping',
            'gamma': gamma,
            'qubits': qubits
        })
        return self
    
    def add_phase_damping(
        self,
        gamma: float,
        qubits: Optional[List[int]] = None
    ) -> 'NoiseModelBuilder':
        """
        Add phase damping channel.
        
        Args:
            gamma: Damping parameter (0 to 1)
            qubits: Specific qubits (None for all)
            
        Returns:
            Self for chaining
        """
        self.noise_channels.append({
            'type': 'phase_damping',
            'gamma': gamma,
            'qubits': qubits
        })
        return self
    
    def add_readout_error(
        self,
        confusion_matrix: List[List[float]],
        qubits: Optional[List[int]] = None
    ) -> 'NoiseModelBuilder':
        """
        Add readout error.
        
        Args:
            confusion_matrix: 2x2 matrix [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
            qubits: Specific qubits (None for all)
            
        Returns:
            Self for chaining
        """
        self.readout_errors = {
            'matrix': np.array(confusion_matrix),
            'qubits': qubits
        }
        return self
    
    def build(self) -> Dict:
        """
        Build the noise model.
        
        Returns:
            Noise model dictionary
        """
        return {
            'noise_channels': self.noise_channels,
            'readout_errors': self.readout_errors
        }
    
    def get_kraus_operators(self, channel_type: str, **params) -> List[np.ndarray]:
        """
        Get Kraus operators for a noise channel.
        
        Args:
            channel_type: Type of noise channel
            **params: Channel parameters
            
        Returns:
            List of Kraus operators
        """
        if channel_type == 'depolarizing':
            return self._depolarizing_kraus(params['error_rate'])
        
        elif channel_type == 'amplitude_damping':
            return self._amplitude_damping_kraus(params['gamma'])
        
        elif channel_type == 'phase_damping':
            return self._phase_damping_kraus(params['gamma'])
        
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")
    
    def _depolarizing_kraus(self, p: float) -> List[np.ndarray]:
        """
        Kraus operators for depolarizing channel.
        
        ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)
        """
        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        k0 = np.sqrt(1 - 3*p/4) * I
        k1 = np.sqrt(p/4) * X
        k2 = np.sqrt(p/4) * Y
        k3 = np.sqrt(p/4) * Z
        
        return [k0, k1, k2, k3]
    
    def _amplitude_damping_kraus(self, gamma: float) -> List[np.ndarray]:
        """
        Kraus operators for amplitude damping.
        
        Models energy relaxation (T1 decay).
        """
        k0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ])
        
        k1 = np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ])
        
        return [k0, k1]
    
    def _phase_damping_kraus(self, gamma: float) -> List[np.ndarray]:
        """
        Kraus operators for phase damping.
        
        Models dephasing (T2 decay).
        """
        k0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ])
        
        k1 = np.array([
            [0, 0],
            [0, np.sqrt(gamma)]
        ])
        
        return [k0, k1]
    
    def apply_noise_to_state(
        self,
        density_matrix: np.ndarray,
        channel_type: str,
        **params
    ) -> np.ndarray:
        """
        Apply noise channel to a density matrix.
        
        Args:
            density_matrix: Input density matrix
            channel_type: Type of noise channel
            **params: Channel parameters
            
        Returns:
            Noisy density matrix
        """
        kraus_ops = self.get_kraus_operators(channel_type, **params)
        
        # Apply Kraus operators: ρ' = Σ_i K_i ρ K_i†
        noisy_rho = np.zeros_like(density_matrix, dtype=complex)
        
        for K in kraus_ops:
            noisy_rho += K @ density_matrix @ K.conj().T
        
        return noisy_rho
    
    @staticmethod
    def estimate_error_rates_from_hardware(
        backend_properties: Dict
    ) -> Dict[str, float]:
        """
        Estimate error rates from hardware calibration data.
        
        Args:
            backend_properties: Hardware properties dictionary
            
        Returns:
            Dictionary of estimated error rates
        """
        # This would parse real hardware data
        # Simplified example:
        return {
            'single_qubit_gate_error': 0.001,
            'two_qubit_gate_error': 0.01,
            'readout_error': 0.02,
            't1': 50e-6,  # 50 microseconds
            't2': 70e-6   # 70 microseconds
        }
