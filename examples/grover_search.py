"""
Example: Grover's Search Algorithm
===================================

Demonstrates Grover's algorithm for searching marked items.
"""

from quantum_forge import GroverSearch
import numpy as np


def search_single_item():
    """Search for a single marked item."""
    print("=" * 60)
    print("Grover's Search: Single Item")
    print("=" * 60)
    
    # 3-qubit search (8 items)
    num_qubits = 3
    marked_state = ['101']  # Mark state |101⟩
    
    grover = GroverSearch(num_qubits, marked_state)
    
    print(f"\n📊 Search Space: {2**num_qubits} items")
    print(f"🎯 Marked State: {marked_state[0]}")
    print(f"🔄 Optimal Iterations: {grover.calculate_optimal_iterations(1)}")
    
    # Simulate
    result = grover.simulate()
    
    print(f"\n✅ Results:")
    print(f"Success Probability: {result['success_probability']:.2%}")
    print(f"\nTop 5 States:")
    for i, state_info in enumerate(result['top_states'], 1):
        marker = "🎯" if state_info['is_marked'] else "  "
        print(f"{marker} {i}. |{state_info['state']}⟩: {state_info['probability']:.2%}")


def search_multiple_items():
    """Search for multiple marked items."""
    print("\n" + "=" * 60)
    print("Grover's Search: Multiple Items")
    print("=" * 60)
    
    # 4-qubit search (16 items)
    num_qubits = 4
    marked_states = ['0011', '1010', '1111']  # Mark 3 states
    
    grover = GroverSearch(num_qubits, marked_states)
    
    print(f"\n📊 Search Space: {2**num_qubits} items")
    print(f"🎯 Marked States: {', '.join(marked_states)}")
    print(f"🔄 Optimal Iterations: {grover.calculate_optimal_iterations(len(marked_states))}")
    
    # Simulate
    result = grover.simulate()
    
    print(f"\n✅ Results:")
    print(f"Success Probability: {result['success_probability']:.2%}")
    print(f"\nTop 5 States:")
    for i, state_info in enumerate(result['top_states'], 1):
        marker = "🎯" if state_info['is_marked'] else "  "
        print(f"{marker} {i}. |{state_info['state']}⟩: {state_info['probability']:.2%}")


def compare_iterations():
    """Compare different numbers of iterations."""
    print("\n" + "=" * 60)
    print("Grover's Search: Iteration Comparison")
    print("=" * 60)
    
    num_qubits = 3
    marked_states = ['110']
    grover = GroverSearch(num_qubits, marked_states)
    
    optimal = grover.calculate_optimal_iterations(1)
    
    print(f"\n📊 Testing different iteration counts:")
    print(f"Optimal iterations: {optimal}\n")
    
    for num_iter in range(1, 6):
        result = grover.simulate(num_iterations=num_iter)
        success_prob = result['success_probability']
        
        marker = "⭐" if num_iter == optimal else "  "
        print(f"{marker} {num_iter} iterations: {success_prob:.2%} success")


def demonstrate_speedup():
    """Demonstrate quantum speedup."""
    print("\n" + "=" * 60)
    print("Grover's Speedup Analysis")
    print("=" * 60)
    
    print("\nClassical vs Quantum Search Complexity:\n")
    
    for n in [4, 8, 12, 16, 20]:
        N = 2 ** n
        classical_queries = N / 2  # Average case
        quantum_queries = int(np.pi / 4 * np.sqrt(N))
        speedup = classical_queries / quantum_queries
        
        print(f"n={n:2d} qubits ({N:6d} items):")
        print(f"  Classical: ~{classical_queries:8.0f} queries")
        print(f"  Quantum:   ~{quantum_queries:8.0f} queries")
        print(f"  Speedup:   ~{speedup:8.1f}x\n")


def main():
    search_single_item()
    search_multiple_items()
    compare_iterations()
    demonstrate_speedup()
    
    print("\n" + "=" * 60)
    print("✅ Grover's algorithm demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
