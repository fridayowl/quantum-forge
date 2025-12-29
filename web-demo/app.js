// QuantumForge Web Demo - Main Application Logic

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active button
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show corresponding demo
        const demoId = btn.dataset.demo + '-demo';
        document.querySelectorAll('.demo-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(demoId).classList.add('active');
    });
});

// Update value displays for range inputs
document.querySelectorAll('input[type="range"]').forEach(input => {
    const updateDisplay = () => {
        const display = input.nextElementSibling;
        if (display && display.classList.contains('value-display')) {
            display.textContent = input.value;
        }
    };
    
    input.addEventListener('input', updateDisplay);
    updateDisplay();
});

// ===== GROVER'S SEARCH =====
const groverQubits = document.getElementById('grover-qubits');
const searchSpaceDisplay = document.getElementById('search-space');
const optimalIterDisplay = document.getElementById('optimal-iter');
const groverIterations = document.getElementById('grover-iterations');

groverQubits.addEventListener('input', () => {
    const numQubits = parseInt(groverQubits.value);
    const searchSpace = Math.pow(2, numQubits);
    searchSpaceDisplay.textContent = searchSpace;
    
    // Update optimal iterations
    const optimal = Math.round(Math.PI / 4 * Math.sqrt(searchSpace));
    optimalIterDisplay.textContent = optimal;
    groverIterations.value = optimal;
    groverIterations.nextElementSibling.textContent = optimal;
});

document.getElementById('run-grover').addEventListener('click', () => {
    const numQubits = parseInt(groverQubits.value);
    const markedState = document.getElementById('marked-state').value;
    const numIterations = parseInt(groverIterations.value);
    
    // Validate marked state
    if (!/^[01]+$/.test(markedState) || markedState.length !== numQubits) {
        alert(`Marked state must be a ${numQubits}-bit binary string`);
        return;
    }
    
    // Run Grover's algorithm
    const result = runGroverSearch(numQubits, [markedState], numIterations);
    
    // Update stats
    document.getElementById('grover-success').textContent = 
        (result.successProbability * 100).toFixed(1) + '%';
    
    const searchSpace = Math.pow(2, numQubits);
    const classicalQueries = searchSpace / 2;
    const quantumQueries = numIterations;
    const speedup = classicalQueries / quantumQueries;
    document.getElementById('grover-speedup').textContent = 
        speedup.toFixed(1) + 'x';
    
    // Update results table
    const tbody = document.querySelector('#grover-results tbody');
    tbody.innerHTML = '';
    
    result.topStates.forEach((state, index) => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${index + 1}</td>
            <td><code>|${state.state}⟩</code></td>
            <td>${(state.probability * 100).toFixed(2)}%</td>
            <td>${state.isMarked ? '🎯' : ''}</td>
        `;
        if (state.isMarked) {
            row.style.background = 'rgba(99, 102, 241, 0.2)';
        }
    });
    
    // Draw chart
    drawGroverChart(result.probabilities, markedState);
});

// ===== CIRCUIT OPTIMIZER =====
document.getElementById('run-optimizer').addEventListener('click', () => {
    const numQubits = parseInt(document.getElementById('circuit-qubits').value);
    const depth = parseInt(document.getElementById('circuit-depth').value);
    const optLevel = parseInt(document.getElementById('opt-level').value);
    
    // Generate random circuit
    const originalCircuit = generateRandomCircuit(numQubits, depth);
    
    // Optimize
    const optimizedCircuit = optimizeCircuit(originalCircuit, optLevel);
    
    // Calculate metrics
    const depthReduction = ((originalCircuit.depth - optimizedCircuit.depth) / originalCircuit.depth * 100);
    const gateReduction = ((originalCircuit.gates - optimizedCircuit.gates) / originalCircuit.gates * 100);
    const fidelityImprovement = calculateFidelityImprovement(originalCircuit, optimizedCircuit);
    
    // Update displays
    document.getElementById('depth-reduction').textContent = depthReduction.toFixed(1) + '%';
    document.getElementById('gate-reduction').textContent = gateReduction.toFixed(1) + '%';
    document.getElementById('fidelity-improvement').textContent = fidelityImprovement.toFixed(3) + 'x';
    
    document.getElementById('orig-depth').textContent = originalCircuit.depth;
    document.getElementById('orig-gates').textContent = originalCircuit.gates;
    document.getElementById('opt-depth').textContent = optimizedCircuit.depth;
    document.getElementById('opt-gates').textContent = optimizedCircuit.gates;
    
    // Draw comparison chart
    drawOptimizationChart(originalCircuit, optimizedCircuit);
});

// ===== QAOA MAX-CUT =====
document.getElementById('run-qaoa').addEventListener('click', () => {
    const numNodes = parseInt(document.getElementById('graph-nodes').value);
    const pLayers = parseInt(document.getElementById('qaoa-layers').value);
    const graphType = document.getElementById('graph-type').value;
    
    // Generate graph
    const graph = generateGraph(numNodes, graphType);
    
    // Run QAOA
    const result = runQAOA(graph, pLayers);
    
    // Update stats
    document.getElementById('cut-value').textContent = result.cutValue;
    document.getElementById('approx-ratio').textContent = 
        (result.approximationRatio * 100).toFixed(1) + '%';
    document.getElementById('partition').textContent = result.partition;
    
    // Draw graph
    drawGraph(graph, result.partition);
});

// ===== VQE SIMULATOR =====
document.getElementById('run-vqe').addEventListener('click', () => {
    const molecule = document.getElementById('molecule').value;
    const ansatz = document.getElementById('ansatz-type').value;
    const maxIter = parseInt(document.getElementById('vqe-iterations').value);
    
    // Get Hamiltonian
    const hamiltonian = getMolecularHamiltonian(molecule);
    
    // Run VQE
    const result = runVQE(hamiltonian, ansatz, maxIter);
    
    // Update stats
    document.getElementById('vqe-energy').textContent = result.energy.toFixed(6) + ' Ha';
    document.getElementById('exact-energy').textContent = result.exactEnergy.toFixed(6) + ' Ha';
    document.getElementById('vqe-error').textContent = 
        (Math.abs(result.energy - result.exactEnergy) * 1000).toFixed(3) + ' mHa';
    
    // Draw convergence chart
    drawVQEChart(result.energyHistory);
});

// ===== STATE ANALYZER =====
document.getElementById('preset-state').addEventListener('change', (e) => {
    const customGroup = document.getElementById('custom-state-group');
    if (e.target.value === 'custom') {
        customGroup.style.display = 'block';
    } else {
        customGroup.style.display = 'none';
    }
});

document.getElementById('analyze-state').addEventListener('click', () => {
    const preset = document.getElementById('preset-state').value;
    let stateVector;
    
    if (preset === 'custom') {
        try {
            stateVector = JSON.parse(document.getElementById('custom-state').value);
        } catch (e) {
            alert('Invalid JSON format for state vector');
            return;
        }
    } else {
        stateVector = getPresetState(preset);
    }
    
    // Analyze state
    const result = analyzeQuantumState(stateVector);
    
    // Update stats
    document.getElementById('state-purity').textContent = result.purity.toFixed(4);
    document.getElementById('state-entropy').textContent = result.entropy.toFixed(4);
    document.getElementById('state-concurrence').textContent = 
        result.concurrence !== null ? result.concurrence.toFixed(4) : 'N/A';
    
    // Draw state visualization
    drawStateChart(stateVector);
});

// Initialize with default values
window.addEventListener('load', () => {
    groverQubits.dispatchEvent(new Event('input'));
});
