// QuantumForge Web Demo - Quantum Simulation Library

// ===== GROVER'S SEARCH =====
function runGroverSearch(numQubits, markedStates, numIterations) {
    const searchSpace = Math.pow(2, numQubits);
    
    // Initialize uniform superposition
    let state = new Array(searchSpace).fill(1 / Math.sqrt(searchSpace));
    
    // Apply Grover iterations
    for (let iter = 0; iter < numIterations; iter++) {
        // Oracle: flip phase of marked states
        for (let markedState of markedStates) {
            const idx = parseInt(markedState, 2);
            state[idx] *= -1;
        }
        
        // Diffusion operator (inversion about average)
        const avg = state.reduce((sum, amp) => sum + amp, 0) / searchSpace;
        state = state.map(amp => 2 * avg - amp);
    }
    
    // Calculate probabilities
    const probabilities = state.map(amp => amp * amp);
    
    // Get top states
    const topStates = probabilities
        .map((prob, idx) => ({
            state: idx.toString(2).padStart(numQubits, '0'),
            probability: prob,
            isMarked: markedStates.includes(idx.toString(2).padStart(numQubits, '0'))
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 5);
    
    // Calculate success probability
    const successProbability = markedStates.reduce((sum, state) => {
        const idx = parseInt(state, 2);
        return sum + probabilities[idx];
    }, 0);
    
    return {
        probabilities,
        topStates,
        successProbability
    };
}

// ===== CIRCUIT OPTIMIZATION =====
function generateRandomCircuit(numQubits, depth) {
    const gates = ['H', 'X', 'Y', 'Z', 'T', 'S', 'CX'];
    let gateCount = 0;
    let circuit = [];
    
    for (let d = 0; d < depth; d++) {
        for (let q = 0; q < numQubits; q++) {
            if (Math.random() > 0.3) {
                const gate = gates[Math.floor(Math.random() * gates.length)];
                circuit.push({ gate, qubit: q, layer: d });
                gateCount++;
            }
        }
    }
    
    return {
        depth,
        gates: gateCount,
        circuit
    };
}

function optimizeCircuit(circuit, optLevel) {
    let optimizedDepth = circuit.depth;
    let optimizedGates = circuit.gates;
    
    // Simulate optimization based on level
    const reductions = [0, 0.15, 0.35, 0.55];
    const reduction = reductions[optLevel];
    
    optimizedDepth = Math.max(1, Math.floor(circuit.depth * (1 - reduction)));
    optimizedGates = Math.max(1, Math.floor(circuit.gates * (1 - reduction * 0.8)));
    
    return {
        depth: optimizedDepth,
        gates: optimizedGates
    };
}

function calculateFidelityImprovement(original, optimized) {
    // Simplified fidelity calculation
    const singleQubitError = 0.001;
    const twoQubitError = 0.01;
    
    const origFidelity = Math.pow(1 - singleQubitError, original.gates);
    const optFidelity = Math.pow(1 - singleQubitError, optimized.gates);
    
    return optFidelity / origFidelity;
}

// ===== QAOA MAX-CUT =====
function generateGraph(numNodes, type) {
    const graph = {
        nodes: Array.from({ length: numNodes }, (_, i) => i),
        edges: []
    };
    
    if (type === 'cycle') {
        for (let i = 0; i < numNodes; i++) {
            graph.edges.push([i, (i + 1) % numNodes]);
        }
    } else if (type === 'complete') {
        for (let i = 0; i < numNodes; i++) {
            for (let j = i + 1; j < numNodes; j++) {
                graph.edges.push([i, j]);
            }
        }
    } else { // random
        const numEdges = Math.floor(numNodes * 1.5);
        for (let i = 0; i < numEdges; i++) {
            const a = Math.floor(Math.random() * numNodes);
            const b = Math.floor(Math.random() * numNodes);
            if (a !== b && !graph.edges.some(e => 
                (e[0] === a && e[1] === b) || (e[0] === b && e[1] === a))) {
                graph.edges.push([a, b]);
            }
        }
    }
    
    return graph;
}

function runQAOA(graph, pLayers) {
    // Simplified QAOA simulation
    // In reality, this would run quantum circuits
    
    // Find optimal cut by trying random partitions
    let bestCut = 0;
    let bestPartition = '';
    
    for (let trial = 0; trial < 100; trial++) {
        const partition = Array.from({ length: graph.nodes.length }, 
            () => Math.random() > 0.5 ? '1' : '0').join('');
        
        const cutValue = calculateCutValue(graph, partition);
        
        if (cutValue > bestCut) {
            bestCut = cutValue;
            bestPartition = partition;
        }
    }
    
    // Calculate optimal cut (brute force for small graphs)
    let optimalCut = bestCut;
    if (graph.nodes.length <= 6) {
        for (let i = 0; i < Math.pow(2, graph.nodes.length); i++) {
            const partition = i.toString(2).padStart(graph.nodes.length, '0');
            const cutValue = calculateCutValue(graph, partition);
            optimalCut = Math.max(optimalCut, cutValue);
        }
    }
    
    return {
        cutValue: bestCut,
        partition: bestPartition,
        approximationRatio: bestCut / optimalCut
    };
}

function calculateCutValue(graph, partition) {
    return graph.edges.reduce((count, [a, b]) => {
        return count + (partition[a] !== partition[b] ? 1 : 0);
    }, 0);
}

// ===== VQE SIMULATOR =====
function getMolecularHamiltonian(molecule) {
    const hamiltonians = {
        h2: {
            c0: -0.8105,
            c1: 0.1721,
            c2: 0.1721,
            c3: -0.2228,
            c4: 0.1686
        },
        lih: {
            c0: -7.8823,
            c1: 0.2252,
            c2: 0.2252,
            c3: -0.3425,
            c4: 0.1854
        },
        custom: {
            c0: -1.0,
            c1: 0.5,
            c2: 0.5,
            c3: -0.25,
            c4: 0.15
        }
    };
    
    return hamiltonians[molecule];
}

function runVQE(hamiltonian, ansatz, maxIter) {
    // Exact ground state energy
    const exactEnergy = hamiltonian.c0 + hamiltonian.c3 - Math.abs(hamiltonian.c4);
    
    // Simulate VQE optimization
    const energyHistory = [];
    let currentEnergy = hamiltonian.c0 + 0.5; // Start above ground state
    
    for (let iter = 0; iter < maxIter; iter++) {
        // Simulate convergence
        const progress = iter / maxIter;
        const noise = (Math.random() - 0.5) * 0.01 * (1 - progress);
        currentEnergy = exactEnergy + (currentEnergy - exactEnergy) * (1 - 0.1) + noise;
        energyHistory.push(currentEnergy);
    }
    
    return {
        energy: currentEnergy,
        exactEnergy,
        energyHistory
    };
}

// ===== STATE ANALYZER =====
function getPresetState(preset) {
    const states = {
        bell: [1, 0, 0, 1].map(x => x / Math.sqrt(2)),
        ghz: [1, 0, 0, 0, 0, 0, 0, 1].map(x => x / Math.sqrt(2)),
        w: [0, 1, 1, 0, 1, 0, 0, 0].map(x => x / Math.sqrt(3)),
        product: [1, 0, 0, 0]
    };
    
    return states[preset];
}

function analyzeQuantumState(stateVector) {
    const numQubits = Math.log2(stateVector.length);
    
    // Normalize
    const norm = Math.sqrt(stateVector.reduce((sum, amp) => sum + amp * amp, 0));
    const normalized = stateVector.map(amp => amp / norm);
    
    // Calculate purity
    const purity = normalized.reduce((sum, amp) => sum + Math.pow(amp, 4), 0);
    
    // Calculate von Neumann entropy
    const probabilities = normalized.map(amp => amp * amp).filter(p => p > 1e-10);
    const entropy = -probabilities.reduce((sum, p) => sum + p * Math.log2(p), 0);
    
    // Calculate concurrence (for 2-qubit states)
    let concurrence = null;
    if (numQubits === 2) {
        // Simplified concurrence calculation
        const rho = normalized.map((amp, i) => 
            normalized.map((_, j) => amp * normalized[j])
        );
        concurrence = Math.max(0, 2 * Math.abs(normalized[0] * normalized[3] - 
                                                normalized[1] * normalized[2]));
    }
    
    return {
        purity,
        entropy,
        concurrence
    };
}

// ===== VISUALIZATION HELPERS =====
function drawGroverChart(probabilities, markedState) {
    const canvas = document.getElementById('grover-chart');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const barWidth = width / probabilities.length;
    const maxProb = Math.max(...probabilities);
    const markedIdx = parseInt(markedState, 2);
    
    probabilities.forEach((prob, idx) => {
        const barHeight = (prob / maxProb) * (height - 40);
        const x = idx * barWidth;
        const y = height - barHeight - 20;
        
        // Draw bar
        ctx.fillStyle = idx === markedIdx ? '#10b981' : '#6366f1';
        ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
        
        // Draw label
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(idx.toString(2).padStart(Math.log2(probabilities.length), '0'), 
                     x + barWidth / 2, height - 5);
    });
}

function drawOptimizationChart(original, optimized) {
    const canvas = document.getElementById('optimization-chart');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const metrics = [
        { label: 'Depth', orig: original.depth, opt: optimized.depth },
        { label: 'Gates', orig: original.gates, opt: optimized.gates }
    ];
    
    const barHeight = 40;
    const spacing = 80;
    
    metrics.forEach((metric, idx) => {
        const y = idx * spacing + 50;
        const maxVal = Math.max(metric.orig, metric.opt);
        
        // Original bar
        const origWidth = (metric.orig / maxVal) * (width - 200);
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(100, y, origWidth, barHeight);
        ctx.fillStyle = '#f1f5f9';
        ctx.fillText(metric.orig, 100 + origWidth + 10, y + 25);
        
        // Optimized bar
        const optWidth = (metric.opt / maxVal) * (width - 200);
        ctx.fillStyle = '#10b981';
        ctx.fillRect(100, y + barHeight + 10, optWidth, barHeight);
        ctx.fillText(metric.opt, 100 + optWidth + 10, y + barHeight + 35);
        
        // Label
        ctx.fillStyle = '#cbd5e1';
        ctx.textAlign = 'right';
        ctx.fillText(metric.label, 90, y + 25);
    });
}

function drawGraph(graph, partition) {
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Calculate node positions in a circle
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 50;
    
    const positions = graph.nodes.map((_, i) => {
        const angle = (i / graph.nodes.length) * 2 * Math.PI - Math.PI / 2;
        return {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
        };
    });
    
    // Draw edges
    graph.edges.forEach(([a, b]) => {
        const isCut = partition[a] !== partition[b];
        ctx.strokeStyle = isCut ? '#10b981' : '#475569';
        ctx.lineWidth = isCut ? 3 : 1;
        ctx.beginPath();
        ctx.moveTo(positions[a].x, positions[a].y);
        ctx.lineTo(positions[b].x, positions[b].y);
        ctx.stroke();
    });
    
    // Draw nodes
    graph.nodes.forEach((node, i) => {
        ctx.fillStyle = partition[i] === '1' ? '#ef4444' : '#6366f1';
        ctx.beginPath();
        ctx.arc(positions[i].x, positions[i].y, 20, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node, positions[i].x, positions[i].y);
    });
}

function drawVQEChart(energyHistory) {
    const canvas = document.getElementById('vqe-chart');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    
    const minEnergy = Math.min(...energyHistory);
    const maxEnergy = Math.max(...energyHistory);
    const energyRange = maxEnergy - minEnergy;
    
    // Draw axes
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw energy curve
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    energyHistory.forEach((energy, iter) => {
        const x = padding + (iter / (energyHistory.length - 1)) * chartWidth;
        const y = height - padding - ((energy - minEnergy) / energyRange) * chartHeight;
        
        if (iter === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Iteration', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy (Ha)', 0, 0);
    ctx.restore();
}

function drawStateChart(stateVector) {
    const canvas = document.getElementById('state-chart');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const barWidth = width / stateVector.length;
    const maxAmp = Math.max(...stateVector.map(Math.abs));
    
    stateVector.forEach((amp, idx) => {
        const barHeight = (Math.abs(amp) / maxAmp) * (height - 40);
        const x = idx * barWidth;
        const y = height - barHeight - 20;
        
        ctx.fillStyle = amp >= 0 ? '#6366f1' : '#ef4444';
        ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
        
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        const numQubits = Math.log2(stateVector.length);
        ctx.fillText(idx.toString(2).padStart(numQubits, '0'), 
                     x + barWidth / 2, height - 5);
    });
}
