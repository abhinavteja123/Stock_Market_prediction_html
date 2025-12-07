// Globals
let analysisData = null;

// DOM Elements
const views = {
    welcome: document.getElementById('welcomeState'),
    loading: document.getElementById('loadingState'),
    results: document.getElementById('resultsState')
};
const form = document.getElementById('analysisForm');
const logs = document.getElementById('buildLogs');
const progressBar = document.querySelector('.progress-fill');
const valSplitSlider = document.getElementById('validationSplit');
const valSplitDisplay = document.getElementById('valSplitValue');

// Utils
function switchView(viewName) {
    Object.values(views).forEach(el => el.classList.remove('active'));
    views[viewName].classList.add('active');
}

function log(msg) {
    logs.innerHTML += `> ${msg}<br>`;
    logs.scrollTop = logs.scrollHeight;
}

// Event Listeners
valSplitSlider.addEventListener('input', (e) => {
    valSplitDisplay.textContent = e.target.value;
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const symbol = document.getElementById('symbol').value;
    const split = parseInt(document.getElementById('validationSplit').value);
    const epochs = parseInt(document.getElementById('lstmEpochs').value);
    const batch = parseInt(document.getElementById('lstmBatchSize').value);
    const windowSize = parseInt(document.getElementById('lstmWindow').value);

    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    // Start UI Flow
    switchView('loading');
    logs.innerHTML = '> Initializing training sequence...<br>';
    progressBar.style.width = '10%';

    setTimeout(() => { log(`Fetching market data for ${symbol}...`); progressBar.style.width = '30%'; }, 500);
    setTimeout(() => { log(`Engineering technical indicators...`); progressBar.style.width = '50%'; }, 1500);
    setTimeout(() => { log(`Training ensemble models (Logistic, XGBoost, LSTM)...`); progressBar.style.width = '70%'; }, 3500);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                validation_split: split,
                lstm_epochs: epochs,
                lstm_batch_size: batch,
                lstm_window_size: windowSize,
                start_date: startDate,
                end_date: endDate
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Analysis failed');
        }

        analysisData = await response.json();

        progressBar.style.width = '100%';
        log('Analysis complete. Rendering dashboard...');

        setTimeout(() => {
            try {
                renderDashboard(analysisData);
                switchView('results');
            } catch (err) {
                console.error(err);
                alert("Rendering Error: " + err.message);
                log("RENDER ERROR: " + err.message);
            }
        }, 1000);

    } catch (error) {
        log(`ERROR: ${error.message}`);
        alert(`Error: ${error.message}`);
        switchView('welcome');
    }
});

function renderDashboard(data) {
    document.getElementById('resSymbol').textContent = data.symbol;

    // --- 1. EDA Section ---
    if (data.eda) renderEDA(data.eda, data.chart_data);

    // --- 2. Metrics Table ---
    const tbody = document.querySelector('#metricsTable tbody');
    tbody.innerHTML = '';
    if (data.results_table) {
        data.results_table.forEach(model => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color:white; font-weight:600">${model.Model}</td>
                <td>${model['Validation F1']}</td>
                <td>${model['Validation Accuracy']}</td>
                <td>${model['Valid AUC']}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // --- 3. Best Model Reports & Diagnostics ---
    renderDetailedReports(data.detailed_results, data.best_model_name);

}

function renderEDA(eda, chartData) {
    // A. Pie Chart Removed

    // B. Price Chart (Enhanced)
    const dates = chartData.map(d => d.Date);
    const closes = chartData.map(d => d.Close);
    const tracePrice = {
        x: dates,
        y: closes,
        type: 'scatter',
        mode: 'lines',
        name: 'Close Price',
        line: { color: '#3b82f6', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(59, 130, 246, 0.1)'
    };

    Plotly.newPlot('priceChart', [tracePrice], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            title: 'Price (INR)'
        },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            rangeselector: {
                buttons: [
                    { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
                    { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
                    { count: 1, label: 'YTD', step: 'year', stepmode: 'todate' },
                    { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
                    { step: 'all', label: 'All' }
                ],
                bgcolor: '#1e293b',
                activecolor: '#3b82f6',
                font: { color: '#94a3b8' }
            },
            rangeslider: { visible: true, borderwidth: 0, bgcolor: 'rgba(0,0,0,0.1)' }
        },
        margin: { t: 20, b: 50, l: 50, r: 20 },
        showlegend: false
    }, { responsive: true, displayModeBar: false }).then(() => {
        // Force resize
        setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 100);
    });

    // C. Histograms
    const histContainer = document.getElementById('featureHistograms');
    histContainer.innerHTML = '';
    Object.entries(eda.histograms).forEach(([feature, data]) => {
        const div = document.createElement('div');
        div.className = 'small-hist';
        div.id = `hist_${feature}`;
        histContainer.appendChild(div);

        const edges = data.bin_edges;
        const centers = [];
        for (let i = 0; i < edges.length - 1; i++) centers.push((edges[i] + edges[i + 1]) / 2);

        const traceHist = {
            x: centers,
            y: data.counts,
            type: 'bar',
            marker: { color: '#6366f1' }
        };

        Plotly.newPlot(div.id, [traceHist], {
            title: { text: feature, font: { size: 12, color: '#94a3b8' } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', size: 10 },
            margin: { t: 25, b: 20, l: 25, r: 10 },
            xaxis: { showgrid: false },
            yaxis: { showgrid: false }
        }, { responsive: true });
    });


    // D. Correlation Heatmap
    if (eda.correlation) {
        const traceHeat = {
            x: eda.correlation.x,
            y: eda.correlation.y,
            z: eda.correlation.z,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmin: -1,
            zmax: 1,
            texttemplate: '%{z:.2f}',
            textfont: { color: 'white' }
        };
        Plotly.newPlot('corrHeatmap', [traceHeat], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8' },
            margin: { t: 20, b: 80, l: 120, r: 20 },
            xaxis: { tickangle: -45 }
        }, { responsive: true, displayModeBar: false });
    }
}

function renderDetailedReports(results, bestModelName) {
    const bestModel = results.find(m => m.name === bestModelName);

    // 1. Classification Report
    let reportHtml = '<p style="color:#64748b">No comprehensive report available for this model type.</p>';
    if (bestModel && bestModel.classification_report) {
        const cr = bestModel.classification_report;
        let rows = '';

        // Helper to safely get nested val
        const getVal = (k, metric) => cr[k] ? cr[k][metric] : 0;

        // Dynamic row generation
        Object.keys(cr).forEach(key => {
            if (key === 'accuracy') {
                rows += `
                    <tr style="border-top: 1px solid rgba(255,255,255,0.2)">
                        <td><b>Accuracy</b></td>
                        <td colspan="2"></td>
                        <td><b>${(cr['accuracy']).toFixed(4)}</b></td>
                        <td>${cr['macro avg'] ? cr['macro avg']['support'] : '-'}</td>
                    </tr>`;
            } else if (typeof cr[key] === 'object') {
                const rowName = key === '0' ? 'Down (0)' : key === '1' ? 'Up (1)' : key;
                rows += `
                    <tr>
                        <td>${rowName}</td>
                        <td>${(getVal(key, 'precision')).toFixed(4)}</td>
                        <td>${(getVal(key, 'recall')).toFixed(4)}</td>
                        <td>${(getVal(key, 'f1-score')).toFixed(4)}</td>
                        <td>${getVal(key, 'support')}</td>
                    </tr>`;
            }
        });

        reportHtml = `
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;
    }
    document.getElementById('clfReport').innerHTML = reportHtml;

    // 2. Feature Importance Table
    const tbody = document.querySelector('#featImpTable tbody');
    tbody.innerHTML = '';

    let modelWithFeats = bestModel;
    if (!modelWithFeats || !modelWithFeats.feature_importances) {
        modelWithFeats = results.find(m => m.feature_importances);
    }

    if (modelWithFeats && modelWithFeats.feature_importances) {
        Object.entries(modelWithFeats.feature_importances).forEach(([feat, score]) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${feat}</td><td>${score.toFixed(4)}</td>`;
            tbody.appendChild(tr);
        });
    } else {
        tbody.innerHTML = '<tr><td colspan="2">Feature importance not available.</td></tr>';
    }

    // 3. Advanced Diagnostics (Pred vs Actual & Confusion Matrix)
    const predContainer = document.getElementById('predVsActualContainer');
    const cmContainer = document.getElementById('confMatricesContainer');

    predContainer.innerHTML = '';
    cmContainer.innerHTML = '';

    results.forEach((model, idx) => {
        // A. Pred vs Actual
        if (model.pred_vs_actual) {
            const div = document.createElement('div');
            div.className = 'chart-container';
            div.style.height = '300px';
            div.id = `predCurve_${idx}`;
            predContainer.appendChild(div);

            const pva = model.pred_vs_actual;
            const xVals = pva.dates || Array.from({ length: pva.actual.length }, (_, i) => i);
            const probs = pva.probability;
            const actuals = pva.actual;
            const threshold = 0.5;

            // Categories for Scatter Plot
            const actUpCorr = { x: [], y: [], text: [] };
            const actUpWrong = { x: [], y: [], text: [] };
            const actDownCorr = { x: [], y: [], text: [] };
            const actDownWrong = { x: [], y: [], text: [] };

            xVals.forEach((x, i) => {
                const p = probs[i];
                const act = actuals[i];
                const pred = p >= threshold ? 1 : 0;
                const isCorrect = act === pred;

                const txt = `Date: ${x}<br>Actual: ${act === 1 ? 'UP' : 'DOWN'}<br>Prob: ${(p * 100).toFixed(2)}%`;

                if (act === 1) { // Actual UP
                    if (isCorrect) {
                        actUpCorr.x.push(x); actUpCorr.y.push(p); actUpCorr.text.push(txt);
                    } else {
                        actUpWrong.x.push(x); actUpWrong.y.push(p); actUpWrong.text.push(txt);
                    }
                } else { // Actual DOWN
                    if (isCorrect) {
                        actDownCorr.x.push(x); actDownCorr.y.push(p); actDownCorr.text.push(txt);
                    } else {
                        actDownWrong.x.push(x); actDownWrong.y.push(p); actDownWrong.text.push(txt);
                    }
                }
            });

            // Add Traces
            const markerSize = 8;
            Plotly.newPlot(div.id, [
                {
                    x: actUpCorr.x, y: actUpCorr.y, mode: 'markers', name: 'Actual UP (Correct)',
                    text: actUpCorr.text, hoverinfo: 'text',
                    marker: { symbol: 'triangle-up', color: '#10b981', size: markerSize, line: { color: 'black', width: 0.5 } }
                },
                {
                    x: actUpWrong.x, y: actUpWrong.y, mode: 'markers', name: 'Actual UP (Wrong)',
                    text: actUpWrong.text, hoverinfo: 'text',
                    marker: { symbol: 'triangle-up', color: '#ef4444', size: markerSize, line: { color: 'black', width: 0.5 } }
                },
                {
                    x: actDownCorr.x, y: actDownCorr.y, mode: 'markers', name: 'Actual DOWN (Correct)',
                    text: actDownCorr.text, hoverinfo: 'text',
                    marker: { symbol: 'triangle-down', color: '#10b981', size: markerSize, line: { color: 'black', width: 0.5 } }
                },
                {
                    x: actDownWrong.x, y: actDownWrong.y, mode: 'markers', name: 'Actual DOWN (Wrong)',
                    text: actDownWrong.text, hoverinfo: 'text',
                    marker: { symbol: 'triangle-down', color: '#ef4444', size: markerSize, line: { color: 'black', width: 0.5 } }
                }
            ], {
                title: { text: `${model.name} - Probabilities`, font: { size: 14, color: '#94a3b8' } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8' },
                margin: { t: 40, b: 30, l: 50, r: 10 },
                xaxis: { showgrid: false },
                yaxis: {
                    showgrid: true, gridcolor: 'rgba(255,255,255,0.05)',
                    title: 'Probability (UP)', range: [0, 1.05]
                },
                shapes: [
                    { type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: threshold, y1: 1.05, yref: 'y', fillcolor: 'rgba(16, 185, 129, 0.1)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: threshold, yref: 'y', fillcolor: 'rgba(239, 68, 68, 0.1)', line: { width: 0 }, layer: 'below' },
                    { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: threshold, y1: threshold, yref: 'y', line: { color: 'blue', width: 2, dash: 'dash' } }
                ],
                showlegend: true,
                legend: { orientation: 'h', y: -0.2, font: { size: 10 } }
            }, { responsive: true, displayModeBar: false });
        }

        // B. Confusion Matrix
        if (model.confusion_matrix) {
            const div = document.createElement('div');
            div.className = 'small-hist';
            div.style.height = '300px';
            div.id = `cm_${idx}`;
            cmContainer.appendChild(div);

            Plotly.newPlot(div.id, [{
                z: model.confusion_matrix,
                x: ['Pred Down', 'Pred Up'],
                y: ['Act Down', 'Act Up'],
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: false,
                textTemplate: "%{z}",
                textfont: { color: 'white', size: 16, weight: 'bold' }
            }], {
                title: { text: `Confusion Matrix - ${model.name}`, font: { size: 14, color: '#fff' } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8' },
                margin: { t: 40, b: 40, l: 80, r: 20 },
                xaxis: { side: 'bottom' },
                yaxis: { autorange: 'reversed' }
            }, { responsive: true, displayModeBar: false });
        }
    });
}

// Prediction Flow
document.getElementById('predictBtn').addEventListener('click', () => {
    const card = document.getElementById('predictionCard');
    const result = analysisData.prediction_result;

    card.classList.remove('hidden');

    // Update Content
    const isUp = result.direction === 'UP';
    document.getElementById('predIcon').textContent = isUp ? '▲' : '▼';
    document.getElementById('predIcon').style.color = isUp ? '#10b981' : '#ef4444';
    card.style.borderColor = isUp ? '#10b981' : '#ef4444';
    card.style.background = isUp ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';

    document.getElementById('predSignal').textContent = result.direction;
    document.getElementById('predSignal').style.color = isUp ? '#10b981' : '#ef4444';

    // Show Best Model Name & Metrics
    document.getElementById('predModel').textContent = analysisData.best_model_name;

    if (analysisData.best_model_metrics) {
        const metrics = analysisData.best_model_metrics;
        document.getElementById('predF1').textContent = (metrics['Validation F1'] || 0).toFixed(4);
        document.getElementById('predAcc').textContent = (metrics['Validation Accuracy'] || 0).toFixed(4);
    }

    document.getElementById('predConf').textContent = `${result.confidence.toFixed(2)}%`;
    document.getElementById('predProb').textContent = `${result.probability_up.toFixed(2)}%`;

    // Animate Bar and Card
    card.classList.add('animate-pop');
    setTimeout(() => {
        document.getElementById('predConfBar').style.width = `${result.confidence}%`;
        document.getElementById('predConfBar').style.backgroundColor = isUp ? '#10b981' : '#ef4444';
    }, 100);

    // Show Shortcut
    document.getElementById('predictionShortcut').classList.remove('hidden');

    // Scroll to it
    card.scrollIntoView({ behavior: 'smooth' });
});

// Scroll Shortcut Function
function scrollToPrediction() {
    const card = document.getElementById('predictionCard');
    if (card) {
        card.scrollIntoView({ behavior: 'smooth' });
        card.classList.remove('animate-pop');
        void card.offsetWidth;
        card.classList.add('animate-pop');
    }
}
