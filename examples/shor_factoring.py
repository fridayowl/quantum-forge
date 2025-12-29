"""
Example: Shor's Factoring Algorithm
====================================

Demonstrates Shor's algorithm for integer factorization.
"""

from quantum_forge import ShorFactoring
import time


def factor_small_numbers():
    """Factor small composite numbers."""
    print("=" * 60)
    print("Shor's Factoring Algorithm")
    print("=" * 60)
    
    test_numbers = [15, 21, 35, 77, 91]
    
    for N in test_numbers:
        print(f"\n🔢 Factoring N = {N}")
        print("-" * 40)
        
        shor = ShorFactoring(N)
        
        # Show algorithm info
        info = shor.get_algorithm_info()
        print(f"Qubits needed: {info['num_qubits_needed']}")
        print(f"Quantum complexity: {info['quantum_complexity']}")
        
        # Try classical check first
        trivial = shor.classical_factor_check()
        if trivial:
            p, q = trivial
            print(f"✅ Trivial factors found: {p} × {q} = {N}")
            continue
        
        # Use quantum algorithm (simulated)
        try:
            start_time = time.time()
            p, q = shor.factor()
            elapsed = time.time() - start_time
            
            if shor.verify_factors(p, q):
                print(f"✅ Factors found: {p} × {q} = {N}")
                print(f"⏱️  Time: {elapsed:.3f}s")
            else:
                print(f"❌ Invalid factors: {p}, {q}")
        except ValueError as e:
            print(f"⚠️  {e}")


def demonstrate_period_finding():
    """Demonstrate period finding subroutine."""
    print("\n" + "=" * 60)
    print("Period Finding (Core Subroutine)")
    print("=" * 60)
    
    N = 15
    test_bases = [2, 4, 7, 11, 13]
    
    print(f"\nFinding periods for N = {N}:\n")
    
    for a in test_bases:
        if ShorFactoring(N).choose_random_a() == a or True:
            # Classical simulation for demonstration
            r = ShorFactoring.simulate_period_finding_classical(a, N)
            
            # Verify
            result = pow(a, r, N)
            check = "✅" if result == 1 else "❌"
            
            print(f"a = {a:2d}: period r = {r:2d}  ({a}^{r} mod {N} = {result}) {check}")


def analyze_complexity():
    """Analyze complexity for different problem sizes."""
    print("\n" + "=" * 60)
    print("Complexity Analysis")
    print("=" * 60)
    
    print("\nProblem Size Scaling:\n")
    print(f"{'N':>10} {'Bits':>6} {'Qubits':>8} {'Classical':>20} {'Quantum':>15}")
    print("-" * 70)
    
    for bits in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        N = 2 ** bits - 1  # Approximate
        qubits = bits
        
        # Complexity estimates
        classical = f"O(exp({bits}^(1/3)))"
        quantum = f"O({bits}^3)"
        
        print(f"{N:>10.0e} {bits:>6} {qubits:>8} {classical:>20} {quantum:>15}")


def compare_with_classical():
    """Compare quantum vs classical factoring."""
    print("\n" + "=" * 60)
    print("Quantum vs Classical Comparison")
    print("=" * 60)
    
    print("\nFor RSA-2048 (617-digit number):")
    print("-" * 40)
    print("Classical (GNFS):")
    print("  • Estimated time: ~300 trillion years")
    print("  • Complexity: O(exp((log N)^(1/3)))")
    print()
    print("Quantum (Shor's):")
    print("  • Estimated time: ~hours (with fault-tolerant QC)")
    print("  • Complexity: O((log N)^3)")
    print("  • Qubits needed: ~4000-8000 logical qubits")
    print()
    print("💡 This is why Shor's algorithm threatens RSA encryption!")


def main():
    factor_small_numbers()
    demonstrate_period_finding()
    analyze_complexity()
    compare_with_classical()
    
    print("\n" + "=" * 60)
    print("✅ Shor's algorithm demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
