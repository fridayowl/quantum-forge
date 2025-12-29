# QuantumForge Web Demo 🌐⚛️

An interactive web-based demonstration of quantum computing algorithms with real-time visualizations.

## Features

### 🔍 Grover's Search Algorithm
- Interactive search space configuration (2-5 qubits)
- Customizable marked states
- Real-time probability distribution visualization
- Quantum speedup analysis
- Optimal iteration calculation

### ⚡ Circuit Optimizer
- Visual circuit comparison (before/after optimization)
- Multiple optimization levels (0-3)
- Depth and gate count reduction metrics
- Fidelity improvement estimation
- Interactive circuit generation

### 🔗 QAOA Max-Cut Solver
- Graph visualization with interactive controls
- Multiple graph types (cycle, complete, random)
- Configurable QAOA layers
- Cut value and approximation ratio display
- Visual partition highlighting

### 🧪 VQE Simulator
- Molecular Hamiltonian selection (H₂, LiH, custom)
- Multiple ansatz types (Hardware Efficient, UCCSD)
- Energy convergence visualization
- Ground state energy calculation
- Error analysis

### 🌀 Quantum State Analyzer
- Preset quantum states (Bell, GHZ, W, Product)
- Custom state vector input
- Entanglement measures (entropy, concurrence)
- Purity calculations
- State amplitude visualization

## Running the Demo

### Option 1: Local Server (Recommended)

```bash
cd web-demo

# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js
npx http-server -p 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Direct File Access

Simply open `index.html` in a modern web browser. Note: Some features may require a local server due to CORS restrictions.

## Browser Compatibility

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Opera 76+

## Technology Stack

- **Pure JavaScript** - No frameworks required
- **HTML5 Canvas** - For quantum state visualizations
- **CSS3** - Modern styling with gradients and animations
- **Vanilla CSS** - No CSS frameworks

## Features Highlights

### Real-Time Simulations
All quantum algorithms run directly in the browser using JavaScript implementations of quantum mechanics principles.

### Interactive Visualizations
- Probability distribution bar charts
- Circuit depth comparison graphs
- Graph partitioning displays
- Energy convergence plots
- State amplitude visualizations

### Educational Content
Each demo includes:
- Algorithm explanations
- Parameter controls
- Statistical analysis
- Performance metrics

## File Structure

```
web-demo/
├── index.html          # Main HTML structure
├── styles.css          # Modern dark theme styling
├── app.js              # UI event handlers and controls
├── quantum-sim.js      # Quantum algorithm simulations
└── README.md           # This file
```

## Customization

### Adding New Algorithms

1. Add a new section in `index.html`
2. Implement the algorithm in `quantum-sim.js`
3. Add event handlers in `app.js`
4. Create visualization function

### Styling

All colors and themes are defined as CSS variables in `styles.css`:

```css
:root {
    --primary: #6366f1;
    --accent: #06b6d4;
    --bg-dark: #0f172a;
    /* ... */
}
```

## Performance

- Optimized for smooth 60 FPS animations
- Efficient canvas rendering
- Minimal DOM manipulations
- Responsive design for all screen sizes

## Educational Use

This demo is perfect for:
- Quantum computing courses
- Research presentations
- Interactive tutorials
- Algorithm demonstrations
- Student projects

## Contributing

Feel free to:
- Add new quantum algorithms
- Improve visualizations
- Enhance UI/UX
- Fix bugs
- Add documentation

## License

MIT License - See main repository LICENSE file

## Links

- **Main Repository**: https://github.com/fridayowl/quantum-forge
- **Documentation**: https://github.com/fridayowl/quantum-forge#readme
- **Issues**: https://github.com/fridayowl/quantum-forge/issues

---

**Built with ⚛️ by the QuantumForge team**
