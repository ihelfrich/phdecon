// Canvas setup
const fieldCanvas = document.getElementById('fieldCanvas');
const kernelCanvas = document.getElementById('kernelCanvas');
const dispersionCanvas = document.getElementById('dispersionCanvas');
const spectrumCanvas = document.getElementById('spectrumCanvas');

const fieldCtx = fieldCanvas.getContext('2d');
const kernelCtx = kernelCanvas.getContext('2d');
const dispersionCtx = dispersionCanvas.getContext('2d');
const spectrumCtx = spectrumCanvas.getContext('2d');

const N = 128;
fieldCanvas.width = N;
fieldCanvas.height = N;

// Simulation state
let field = new Float32Array(N * N);
let fieldNext = new Float32Array(N * N);
let targetField = null; // For parameter estimation
let simTime = 0;
let simRunning = true;
let dataMode = 'synthetic';

// Parameters
const params = {
    lambda: 0.15,
    kappa: 0.003,
    alpha: 1.0,
    M: 1.0,
    y0: 0.0,
    sigma1: 2.0,
    sigma2: 5.0,
    dt: 0.05
};

// Kernel
const kernelSize = 21;
let kernel = new Float32Array(kernelSize * kernelSize);

function computeKernel() {
    const half = Math.floor(kernelSize / 2);
    const s1 = params.sigma1;
    const s2 = params.sigma2;

    let sum = 0;
    for (let i = 0; i < kernelSize; i++) {
        for (let j = 0; j < kernelSize; j++) {
            const x = (i - half);
            const y = (j - half);
            const r2 = x*x + y*y;

            const gauss1 = Math.exp(-r2 / (2 * s1 * s1));
            const gauss2 = Math.exp(-r2 / (2 * s2 * s2));
            const val = gauss1 - 0.5 * gauss2;

            kernel[i * kernelSize + j] = val;
            sum += val;
        }
    }

    const mean = sum / (kernelSize * kernelSize);
    for (let i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] -= mean;
    }

    let l1 = 0;
    for (let i = 0; i < kernelSize * kernelSize; i++) {
        l1 += Math.abs(kernel[i]);
    }
    for (let i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= l1;
    }

    renderKernel();
}

function initField() {
    if (targetField) {
        // Start from target for estimation
        for (let i = 0; i < N * N; i++) {
            field[i] = targetField[i] + (Math.random() - 0.5) * 0.05;
        }
    } else {
        // Random initial condition
        for (let i = 0; i < N * N; i++) {
            field[i] = (Math.random() - 0.5) * 0.01;
        }
    }
    simTime = 0;
}

// Convolution
let convResult = new Float32Array(N * N);
function convolve() {
    const half = Math.floor(kernelSize / 2);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let ki = -half; ki <= half; ki++) {
                for (let kj = -half; kj <= half; kj++) {
                    const ii = (i + ki + N) % N;
                    const jj = (j + kj + N) % N;
                    sum += field[ii * N + jj] * kernel[(ki + half) * kernelSize + (kj + half)];
                }
            }
            convResult[i * N + j] = sum;
        }
    }
}

// Laplacian
let lapResult = new Float32Array(N * N);
function laplacian() {
    const factor = 100;
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const ip = (i + 1) % N;
            const im = (i - 1 + N) % N;
            const jp = (j + 1) % N;
            const jm = (j - 1 + N) % N;

            lapResult[i * N + j] = (
                field[ip * N + j] + field[im * N + j] +
                field[i * N + jp] + field[i * N + jm] -
                4 * field[i * N + j]
            ) * factor;
        }
    }
}

// Time step
function stepSimulation() {
    if (!simRunning) return;

    convolve();
    laplacian();

    const dt = params.dt;
    const k = params.kappa;
    const l = params.lambda;
    const a = params.alpha;
    const M = params.M;
    const y0 = params.y0;

    for (let i = 0; i < N * N; i++) {
        const y = field[i];
        const dydt = M * (k * lapResult[i] + l * convResult[i] - a * Math.pow(y - y0, 3));
        fieldNext[i] = y + dt * dydt;
    }

    [field, fieldNext] = [fieldNext, field];
    simTime += dt;
}

// Render field
const fieldImageData = fieldCtx.createImageData(N, N);
function renderField() {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < N * N; i++) {
        if (field[i] < min) min = field[i];
        if (field[i] > max) max = field[i];
    }

    const range = max - min || 1;

    for (let i = 0; i < N * N; i++) {
        const val = field[i];
        const norm = (val - min) / range;

        // Colormap
        let r, g, b;
        if (norm < 0.25) {
            const t = norm / 0.25;
            r = 0;
            g = t * 120;
            b = 80 + t * 175;
        } else if (norm < 0.5) {
            const t = (norm - 0.25) / 0.25;
            r = t * 87;
            g = 120 + t * 94;
            b = 255 - t * 48;
        } else if (norm < 0.75) {
            const t = (norm - 0.5) / 0.25;
            r = 87 + t * 168;
            g = 214 - t * 74;
            b = 207 - t * 207;
        } else {
            const t = (norm - 0.75) / 0.25;
            r = 255;
            g = 140 - t * 33;
            b = 0;
        }

        fieldImageData.data[i * 4] = r;
        fieldImageData.data[i * 4 + 1] = g;
        fieldImageData.data[i * 4 + 2] = b;
        fieldImageData.data[i * 4 + 3] = 255;
    }

    fieldCtx.putImageData(fieldImageData, 0, 0);
}

// Render kernel
function renderKernel() {
    kernelCanvas.width = 400;
    kernelCanvas.height = 400;
    kernelCtx.fillStyle = '#000';
    kernelCtx.fillRect(0, 0, 400, 400);

    const half = Math.floor(kernelSize / 2);
    let minK = Infinity, maxK = -Infinity;
    for (let i = 0; i < kernelSize * kernelSize; i++) {
        if (kernel[i] < minK) minK = kernel[i];
        if (kernel[i] > maxK) maxK = kernel[i];
    }

    const cellSize = 400 / kernelSize;
    for (let i = 0; i < kernelSize; i++) {
        for (let j = 0; j < kernelSize; j++) {
            const val = kernel[i * kernelSize + j];

            if (val > 0) {
                const norm = val / maxK;
                const intensity = Math.floor(norm * 139);
                kernelCtx.fillStyle = `rgb(87, 94, ${207 + intensity})`;
            } else {
                const norm = -val / Math.abs(minK);
                const intensity = Math.floor(norm * 148);
                kernelCtx.fillStyle = `rgb(${107 + intensity}, 0, 0)`;
            }

            kernelCtx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }

    kernelCtx.fillStyle = '#8b92ff';
    kernelCtx.font = '11px JetBrains Mono, monospace';
    kernelCtx.fillText('Attract (σ₁)', 10, 385);
    kernelCtx.fillStyle = '#ff6b6b';
    kernelCtx.fillText('Repel (σ₂)', 290, 385);
}

// Dispersion relation
function renderDispersion() {
    dispersionCanvas.width = 400;
    dispersionCanvas.height = 400;
    dispersionCtx.fillStyle = '#000';
    dispersionCtx.fillRect(0, 0, 400, 400);

    const kmax_approx = Math.PI / 2;
    const sigma_approx = params.sigma1;
    const Khat_peak = 0.15;

    const kValues = [];
    const sigmaValues = [];
    let maxSigma = -Infinity;
    let kMaxIdx = 0;

    for (let i = 0; i < 100; i++) {
        const k = i * kmax_approx / 50;
        const Khat = Khat_peak * Math.exp(-k * k * sigma_approx * sigma_approx / 2);
        const sigma = params.M * (-params.kappa * k * k + params.lambda * Khat);

        kValues.push(k);
        sigmaValues.push(sigma);

        if (sigma > maxSigma) {
            maxSigma = sigma;
            kMaxIdx = i;
        }
    }

    const kMax = kValues[kMaxIdx];
    document.getElementById('dominant-k').textContent = kMax.toFixed(3);
    document.getElementById('char-scale').textContent = (2 * Math.PI / kMax).toFixed(2);

    // Plot
    dispersionCtx.strokeStyle = '#575ECF';
    dispersionCtx.lineWidth = 2.5;
    dispersionCtx.beginPath();

    const margin = 40;
    const plotWidth = 400 - 2 * margin;
    const plotHeight = 400 - 2 * margin;

    const minSigma = Math.min(...sigmaValues);
    const sigmaRange = maxSigma - minSigma || 1;
    const kRange = kValues[kValues.length - 1];

    for (let i = 0; i < kValues.length; i++) {
        const x = margin + (kValues[i] / kRange) * plotWidth;
        const y = 400 - margin - ((sigmaValues[i] - minSigma) / sigmaRange) * plotHeight;

        if (i === 0) dispersionCtx.moveTo(x, y);
        else dispersionCtx.lineTo(x, y);
    }
    dispersionCtx.stroke();

    // Zero line
    dispersionCtx.strokeStyle = '#333';
    dispersionCtx.setLineDash([5, 5]);
    dispersionCtx.lineWidth = 1;
    dispersionCtx.beginPath();
    const y0 = 400 - margin - ((-minSigma) / sigmaRange) * plotHeight;
    dispersionCtx.moveTo(margin, y0);
    dispersionCtx.lineTo(400 - margin, y0);
    dispersionCtx.stroke();
    dispersionCtx.setLineDash([]);

    // Mark k_max
    const xMax = margin + (kMax / kRange) * plotWidth;
    const yMax = 400 - margin - ((maxSigma - minSigma) / sigmaRange) * plotHeight;
    dispersionCtx.fillStyle = '#ff6b6b';
    dispersionCtx.beginPath();
    dispersionCtx.arc(xMax, yMax, 6, 0, 2 * Math.PI);
    dispersionCtx.fill();

    dispersionCtx.fillStyle = '#9a9691';
    dispersionCtx.font = '11px JetBrains Mono, monospace';
    dispersionCtx.fillText('k', 360, 390);
    dispersionCtx.fillText('σ(k)', 10, 50);
}

// Power spectrum
function renderSpectrum() {
    spectrumCanvas.width = 400;
    spectrumCanvas.height = 400;
    spectrumCtx.fillStyle = '#000';
    spectrumCtx.fillRect(0, 0, 400, 400);

    // Compute FFT power spectrum
    const fftReal = new Float32Array(N * N);
    const fftImag = new Float32Array(N * N);

    for (let i = 0; i < N * N; i++) {
        fftReal[i] = field[i];
        fftImag[i] = 0;
    }

    // Simple 1D radial average for visualization
    const bars = 50;
    const barWidth = 360 / bars;
    const spectrum = new Float32Array(bars);

    // Compute radial spectrum
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const ki = i < N/2 ? i : i - N;
            const kj = j < N/2 ? j : j - N;
            const k = Math.sqrt(ki*ki + kj*kj);
            const bin = Math.min(Math.floor(k / N * bars * 2), bars - 1);
            const val = field[i * N + j];
            spectrum[bin] += val * val;
        }
    }

    // Normalize
    let maxSpec = 0;
    for (let i = 0; i < bars; i++) {
        if (spectrum[i] > maxSpec) maxSpec = spectrum[i];
    }

    // Find peak
    let peakIdx = 0;
    let peakVal = 0;
    for (let i = 1; i < bars; i++) {
        if (spectrum[i] > peakVal) {
            peakVal = spectrum[i];
            peakIdx = i;
        }
    }

    // Draw bars
    for (let i = 0; i < bars; i++) {
        const height = (spectrum[i] / maxSpec) * 300;
        spectrumCtx.fillStyle = i === peakIdx ? '#ff6b6b' : '#575ECF';
        spectrumCtx.fillRect(20 + i * barWidth, 400 - 40 - height, barWidth - 2, height);
    }

    spectrumCtx.fillStyle = '#9a9691';
    spectrumCtx.font = '11px JetBrains Mono, monospace';
    spectrumCtx.fillText('k', 270, 390);
    spectrumCtx.fillText('|ŷ(k)|²', 10, 50);
}

// Stability
function updateStability() {
    const kmax = Math.PI / 2;
    const Khat_peak = 0.15;
    const lambdaC = params.kappa * kmax * kmax / Khat_peak;

    const ratio = params.lambda / lambdaC;
    document.getElementById('lambda-ratio').textContent = ratio.toFixed(3);

    const indicator = document.querySelector('.status-indicator');
    const status = document.getElementById('stability-status');

    if (ratio < 0.95) {
        indicator.className = 'status-indicator status-stable';
        status.textContent = 'Stable';
    } else if (ratio < 1.05) {
        indicator.className = 'status-indicator status-critical';
        status.textContent = 'Critical';
    } else {
        indicator.className = 'status-indicator status-unstable';
        status.textContent = 'Unstable';
    }

    renderDispersion();
}

// Energy
function computeEnergy() {
    let total = 0;
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const idx = i * N + j;
            const ip = ((i + 1) % N) * N + j;
            const jp = i * N + ((j + 1) % N);

            const dx = field[ip] - field[idx];
            const dy = field[jp] - field[idx];
            total += 0.5 * params.kappa * (dx*dx + dy*dy);

            const dev = field[idx] - params.y0;
            total += 0.25 * params.alpha * dev * dev * dev * dev;
        }
    }

    convolve();
    for (let i = 0; i < N * N; i++) {
        total -= 0.5 * params.lambda * field[i] * convResult[i];
    }

    document.getElementById('energy').textContent = total.toFixed(4);

    // If in estimation mode, compute L2 error
    if (targetField) {
        let error = 0;
        for (let i = 0; i < N * N; i++) {
            const diff = field[i] - targetField[i];
            error += diff * diff;
        }
        error = Math.sqrt(error / (N * N));
        console.log('L2 error:', error.toFixed(6));
    }
}

// GeoTIFF loading (simplified - would need geotiff.js library in production)
document.getElementById('geotiffUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const dataMode = document.getElementById('dataMode');
    dataMode.innerHTML = '<strong>Mode: Data Loading...</strong>Processing GeoTIFF';

    // In a real implementation, use geotiff.js:
    // const arrayBuffer = await file.arrayBuffer();
    // const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    // const image = await tiff.getImage();
    // const rasters = await image.readRasters();

    // For now, simulate with a placeholder
    setTimeout(() => {
        targetField = new Float32Array(N * N);

        // Placeholder: create a realistic-looking distribution
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const x = (i - N/2) / 20;
                const y = (j - N/2) / 20;
                const r2 = x*x + y*y;
                targetField[i * N + j] = Math.exp(-r2/10) * (0.5 + 0.5*Math.random());
            }
        }

        // Normalize
        let sum = 0;
        for (let i = 0; i < N * N; i++) sum += targetField[i];
        for (let i = 0; i < N * N; i++) targetField[i] /= sum;

        dataMode.innerHTML = `<strong>Mode: Parameter Estimation</strong>${file.name} loaded. Adjust parameters to match observed distribution.`;
        initField();
    }, 500);
});

// Controls
function setupControls() {
    ['lambda', 'kappa', 'alpha', 'sigma1', 'sigma2'].forEach(id => {
        const slider = document.getElementById(id);
        const display = document.getElementById(`${id}-val`);
        slider.addEventListener('input', () => {
            params[id] = parseFloat(slider.value);
            display.textContent = slider.value;
            if (id === 'sigma1' || id === 'sigma2') {
                computeKernel();
            }
            updateStability();
        });
    });

    document.getElementById('toggleSim').addEventListener('click', (e) => {
        simRunning = !simRunning;
        e.target.textContent = simRunning ? 'Pause' : 'Resume';
        e.target.classList.toggle('active');
    });

    document.getElementById('reset').addEventListener('click', initField);
}

// Main loop
let frameCount = 0;
function animate() {
    requestAnimationFrame(animate);

    stepSimulation();
    renderField();

    if (Math.floor(simTime * 10) % 5 === 0) {
        computeEnergy();
        renderSpectrum();
    }

    frameCount++;
    document.getElementById('time').textContent = simTime.toFixed(2);
}

// Initialize
computeKernel();
initField();
setupControls();
updateStability();
renderDispersion();
renderSpectrum();
animate();
