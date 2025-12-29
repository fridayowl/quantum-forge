"""
Shor's Algorithm Module
=======================

Implementation of Shor's factoring algorithm.
"""

from typing import Tuple, Optional
import numpy as np
from math import gcd
import random
from .qpe import QuantumPhaseEstimation


class ShorFactoring:
    """
    Shor's algorithm for integer factorization.
    
    Provides exponential speedup over classical factoring algorithms.
    
    Features:
    - Period finding using QPE
    - Classical pre/post-processing
    - Modular exponentiation
    - Continued fractions for phase extraction
    """
    
    def __init__(self, N: int):
        """
        Initialize Shor's algorithm for factoring N.
        
        Args:
            N: Number to factor (must be composite, odd, and > 2)
        """
        self.N = N
        self.num_qubits = int(np.ceil(np.log2(N)))
        
    def classical_factor_check(self) -> Optional[Tuple[int, int]]:
        """
        Check for trivial factors before running quantum algorithm.
        
        Returns:
            Tuple of factors if found, None otherwise
        """
        # Check if even
        if self.N % 2 == 0:
            return (2, self.N // 2)
        
        # Check small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in small_primes:
            if self.N % p == 0:
                return (p, self.N // p)
        
        # Check if perfect power
        for b in range(2, int(np.log2(self.N)) + 1):
            a = int(self.N ** (1/b))
            if a ** b == self.N:
                return (a, a ** (b-1))
        
        return None
    
    def choose_random_a(self) -> int:
        """
        Choose random a coprime to N.
        
        Returns:
            Random integer a where 1 < a < N and gcd(a, N) = 1
        """
        while True:
            a = random.randint(2, self.N - 1)
            if gcd(a, self.N) == 1:
                return a
            else:
                # Found factor!
                factor = gcd(a, self.N)
                if factor != 1 and factor != self.N:
                    return a  # Can use this to extract factor
    
    def create_modular_exponentiation_circuit(
        self,
        a: int,
        num_counting_qubits: int
    ):
        """
        Create circuit for modular exponentiation: |x⟩|0⟩ → |x⟩|a^x mod N⟩
        
        Args:
            a: Base for exponentiation
            num_counting_qubits: Number of qubits for counting register
            
        Returns:
            Modular exponentiation circuit
        """
        # This is a simplified placeholder
        # Real implementation requires efficient quantum arithmetic
        
        # For demonstration, we'll use the unitary U|y⟩ = |ay mod N⟩
        dimension = 2 ** self.num_qubits
        U = np.zeros((dimension, dimension), dtype=complex)
        
        for y in range(dimension):
            new_y = (a * y) % self.N if y < self.N else y
            U[new_y, y] = 1
        
        return U
    
    def find_period(self, a: int, num_counting_qubits: int = None) -> int:
        """
        Find period r such that a^r ≡ 1 (mod N).
        
        Uses quantum phase estimation.
        
        Args:
            a: Base
            num_counting_qubits: Precision (default: 2 * num_qubits)
            
        Returns:
            Period r
        """
        if num_counting_qubits is None:
            num_counting_qubits = 2 * self.num_qubits
        
        # Create modular exponentiation unitary
        U = self.create_modular_exponentiation_circuit(a, num_counting_qubits)
        
        # Use QPE to find period
        qpe = QuantumPhaseEstimation(
            num_counting_qubits=num_counting_qubits,
            num_eigenstate_qubits=self.num_qubits
        )
        
        result = qpe.estimate_phase(U)
        
        # Convert phase to period using continued fractions
        phase = result['estimated_phase']
        period = self._phase_to_period(phase, num_counting_qubits)
        
        return period
    
    def _phase_to_period(self, phase: float, precision_bits: int) -> int:
        """
        Convert estimated phase to period using continued fractions.
        
        Args:
            phase: Estimated phase (0 to 1)
            precision_bits: Number of precision bits
            
        Returns:
            Period r
        """
        # Convert phase to fraction s/r where r is the period
        # Using continued fractions algorithm
        
        if phase == 0:
            return 1
        
        # Simple approach: find denominator of phase as fraction
        max_denominator = 2 ** precision_bits
        
        # Try to find r such that phase ≈ s/r
        best_r = 1
        best_error = float('inf')
        
        for r in range(1, min(self.N, max_denominator)):
            s = round(phase * r)
            error = abs(phase - s / r)
            
            if error < best_error:
                best_error = error
                best_r = r
            
            if error < 1 / (2 * max_denominator):
                return r
        
        return best_r
    
    def factor(self, max_attempts: int = 10) -> Tuple[int, int]:
        """
        Factor N using Shor's algorithm.
        
        Args:
            max_attempts: Maximum number of attempts
            
        Returns:
            Tuple of non-trivial factors (p, q) where N = p * q
        """
        # Check for trivial factors first
        trivial = self.classical_factor_check()
        if trivial:
            return trivial
        
        for attempt in range(max_attempts):
            # Choose random a
            a = self.choose_random_a()
            
            # Check if we got lucky with gcd
            g = gcd(a, self.N)
            if g != 1:
                return (g, self.N // g)
            
            # Find period using quantum algorithm
            r = self.find_period(a)
            
            # Check if period is even and a^(r/2) ≠ -1 (mod N)
            if r % 2 == 0:
                x = pow(a, r // 2, self.N)
                
                if x != self.N - 1:  # x ≠ -1 (mod N)
                    # Compute factors
                    factor1 = gcd(x - 1, self.N)
                    factor2 = gcd(x + 1, self.N)
                    
                    if factor1 != 1 and factor1 != self.N:
                        return (factor1, self.N // factor1)
                    
                    if factor2 != 1 and factor2 != self.N:
                        return (factor2, self.N // factor2)
        
        # If we get here, we failed to find factors
        raise ValueError(f"Failed to factor {self.N} after {max_attempts} attempts")
    
    def verify_factors(self, p: int, q: int) -> bool:
        """
        Verify that p * q = N.
        
        Args:
            p: First factor
            q: Second factor
            
        Returns:
            True if factors are correct
        """
        return p * q == self.N and p != 1 and q != 1
    
    @staticmethod
    def simulate_period_finding_classical(a: int, N: int) -> int:
        """
        Classical simulation of period finding (for testing).
        
        Args:
            a: Base
            N: Modulus
            
        Returns:
            Period r
        """
        r = 1
        current = a % N
        
        while current != 1:
            current = (current * a) % N
            r += 1
            
            if r > N:  # Safety check
                break
        
        return r
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the factoring problem.
        
        Returns:
            Dictionary with algorithm parameters
        """
        return {
            'N': self.N,
            'num_qubits_needed': self.num_qubits,
            'counting_qubits_recommended': 2 * self.num_qubits,
            'classical_complexity': f'O(exp(log N)^(1/3))',
            'quantum_complexity': f'O((log N)^3)',
            'speedup': 'Exponential'
        }
