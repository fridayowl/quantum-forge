"""
Example: QAOA for Max-Cut
==========================

Demonstrates using QAOA to solve the Max-Cut problem on a graph.
"""

from quantum_forge import QAOASolver
import networkx as nx
import matplotlib.pyplot as plt


def create_example_graph():
    """Create an example graph for Max-Cut."""
    G = nx.Graph()
    
    # Create a graph with known max-cut
    edges = [
        (0, 1), (0, 2),
        (1, 2), (1, 3),
        (2, 3), (2, 4),
        (3, 4)
    ]
    
    G.add_edges_from(edges)
    return G


def visualize_graph(G, partition=None):
    """Visualize the graph with optional partition coloring."""
    pos = nx.spring_layout(G, seed=42)
    
    if partition is None:
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=16)
    else:
        # Color nodes based on partition
        colors = ['red' if partition[i] == '1' else 'blue' 
                  for i in range(len(partition))]
        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=500, font_size=16)
        
        # Highlight cut edges
        cut_edges = [(i, j) for i, j in G.edges() 
                     if partition[-(i+1)] != partition[-(j+1)]]
        nx.draw_networkx_edges(G, pos, cut_edges, edge_color='green',
                              width=3)
    
    plt.title("Max-Cut Problem")
    plt.show()


def brute_force_max_cut(G):
    """Find optimal max-cut by brute force."""
    n = len(G.nodes())
    max_cut = 0
    best_partition = None
    
    for i in range(2**n):
        partition = format(i, f'0{n}b')
        cut = sum(1 for edge in G.edges() 
                  if partition[-(edge[0]+1)] != partition[-(edge[1]+1)])
        
        if cut > max_cut:
            max_cut = cut
            best_partition = partition
    
    return max_cut, best_partition


def main():
    print("=" * 60)
    print("QAOA for Max-Cut Problem")
    print("=" * 60)
    
    # Create graph
    print("\n📊 Creating example graph...")
    G = create_example_graph()
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    
    # Find optimal solution by brute force
    print("\n🔍 Finding optimal solution (brute force)...")
    optimal_cut, optimal_partition = brute_force_max_cut(G)
    print(f"  Optimal cut value: {optimal_cut}")
    print(f"  Optimal partition: {optimal_partition}")
    
    # Run QAOA with different p values
    for p in [1, 2, 3]:
        print(f"\n🚀 Running QAOA with p={p} layers...")
        
        qaoa = QAOASolver(graph=G, p=p, optimizer='COBYLA')
        result = qaoa.solve()
        
        print(f"  QAOA cut value: {result['cut_value']}")
        print(f"  QAOA partition: {result['optimal_cut']}")
        print(f"  Expected cost: {result['expected_cost']:.4f}")
        
        # Calculate approximation ratio
        approx_ratio = result['cut_value'] / optimal_cut
        print(f"  Approximation ratio: {approx_ratio:.2%}")
        
        if result['cut_value'] == optimal_cut:
            print("  ✅ Found optimal solution!")
        else:
            print(f"  ⚠️  Suboptimal (off by {optimal_cut - result['cut_value']})")
    
    print("\n✅ QAOA simulation complete!")
    
    # Visualize (optional - requires matplotlib)
    # visualize_graph(G, optimal_partition)


if __name__ == "__main__":
    main()
