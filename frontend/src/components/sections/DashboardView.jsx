import { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { PlotlyChart } from '../charts/PlotlyChart';
import { ArrowLeft, CheckCircle, TrendingUp, BarChart2, Zap, Target, Sliders, Activity, Grid, List } from 'lucide-react';

export function DashboardView({ data, onBack }) {
    // Extract Data
    const { symbol, eda, chart_data, results_table, detailed_results, best_model_name, prediction_result } = data;

    const [showPrediction, setShowPrediction] = useState(false);
    const [chartType, setChartType] = useState('candle'); // 'candle' or 'line'
    const predictionRef = useRef(null);
    const actionRef = useRef(null);

    // --- 1. CHART DATA (Toggleable) ---
    const dates = chart_data.map(d => d.Date);
    const closes = chart_data.map(d => d.Close);

    // Simplified Data for Line Chart
    const mainChartData = [
        {
            x: dates,
            y: closes,
            type: 'scatter',
            mode: 'lines',
            name: 'Price',
            line: { color: '#06b6d4', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(6, 182, 212, 0.05)',
            hovertemplate: '$%{y:.2f}'
        }
    ];

    const mainChartLayout = {
        grid: { rows: 1, columns: 1 },
        xaxis: {
            rangeselector: {
                buttons: [
                    { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
                    { count: 6, label: '6M', step: 'month', stepmode: 'backward' },
                    { count: 1, label: '1Y', step: 'year', stepmode: 'backward' },
                    { count: 5, label: '5Y', step: 'year', stepmode: 'backward' },
                    { step: 'all', label: 'MAX' }
                ],
                bgcolor: 'rgba(255,255,255,0.05)',
                activecolor: '#06b6d4',
                font: { color: '#fff', size: 10 }
            },
            rangeslider: { visible: false },
            type: 'date',
            gridcolor: '#1e293b',
            linecolor: '#1e293b',
            automargin: true,
            hoverformat: '%B %d, %Y'
        },
        yaxis: { title: 'Price', gridcolor: '#1e293b', side: 'right', fixedrange: false, automargin: true },
        margin: { t: 50, b: 40, l: 20, r: 60 },
        showlegend: false,
        hovermode: 'x unified',
        hoverlabel: {
            bgcolor: 'rgba(15, 23, 42, 1.0)',
            bordercolor: '#6366f1',
            font: { color: '#ffffff', size: 14, family: 'JetBrains Mono, monospace' }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'JetBrains Mono, monospace', color: '#94a3b8' }
    };

    // --- 2. HELPERS ---
    const renderHistograms = () => {
        if (!eda || !eda.histograms) return null;
        return Object.entries(eda.histograms).map(([feature, hData]) => {
            const edges = hData.bin_edges;
            const centers = [];
            for (let i = 0; i < edges.length - 1; i++) centers.push((edges[i] + edges[i + 1]) / 2);

            return (
                <div key={feature} className="glass-card" style={{ padding: '15px', height: '180px', display: 'flex', flexDirection: 'column' }}>
                    <div className="label-text" style={{ fontSize: '0.8rem', marginBottom: '8px', color: 'var(--text-main)' }}>{feature}</div>
                    <div style={{ flex: 1 }}>
                        <PlotlyChart
                            data={[{ x: centers, y: hData.counts, type: 'bar', marker: { color: '#6366f1' } }]}
                            layout={{
                                margin: { t: 0, b: 20, l: 0, r: 0 },
                                xaxis: { showticklabels: false, showgrid: false },
                                yaxis: { showticklabels: false, showgrid: false },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)'
                            }}
                        />
                    </div>
                </div>
            );
        });
    };

    // Scroll to action
    const scrollToPrediction = () => {
        actionRef.current?.scrollIntoView({ behavior: 'smooth' });
        // Also trigger the prediction view if they click the header button
        setShowPrediction(true);
    };

    // Scroll to prediction result
    useEffect(() => {
        if (showPrediction && predictionRef.current) {
            predictionRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [showPrediction]);

    // Animation variants
    const container = { show: { transition: { staggerChildren: 0.1 } } };
    const item = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0 } };

    return (
        <div className="dashboard-wrapper font-mono" style={{ paddingBottom: '100px' }}>
            {/* HEADER */}
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', borderBottom: '1px solid var(--border-glass)', paddingBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <button onClick={onBack} className="glass-card" style={{ padding: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', letterSpacing: '2px' }}>TICKER</div>
                        <h2 style={{ fontSize: '2.5rem', fontWeight: 800, margin: 0, lineHeight: 1, letterSpacing: '-1px' }}>{symbol}</h2>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                    <button
                        onClick={scrollToPrediction}
                        className="glass-card"
                        style={{
                            padding: '0.8rem 1.5rem',
                            fontSize: '0.9rem',
                            fontWeight: 700,
                            color: '#fff',
                            background: 'linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)',
                            border: 'none',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            boxShadow: '0 0 15px rgba(99, 102, 241, 0.5)',
                            transition: 'all 0.3s ease',
                            letterSpacing: '1px',
                            borderRadius: '8px'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 0 25px rgba(6, 182, 212, 0.8)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 0 15px rgba(99, 102, 241, 0.5)';
                        }}
                    >
                        <Zap size={16} fill="currentColor" />
                        PREDICT NEXT DAY
                    </button>
                    <div className="text-right">
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>LATEST CLOSE</div>
                        <div style={{ fontWeight: 700, fontSize: '1.5rem', color: '#06b6d4', textShadow: '0 0 20px rgba(6, 182, 212, 0.3)' }}>
                            {chart_data[chart_data.length - 1]?.Close.toFixed(2)}
                        </div>
                    </div>
                </div>
            </header>

            <motion.div variants={container} initial="hidden" animate="show" className="bento-grid">

                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem 2rem', borderBottom: '1px solid var(--border-glass)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>MARKET STRUCTURE ANALYSIS</h3>
                    </div>
                    <div style={{ flex: 1, padding: '0.5rem' }}>
                        <PlotlyChart
                            data={mainChartData}
                            layout={mainChartLayout}
                        />
                    </div>
                </motion.div>

                {/* 2. METRICS TABLE (Stacked Full Width) */}
                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem 2rem', borderBottom: '1px solid var(--border-glass)' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>MODEL PERFORMANCE LEADERBOARD</h3>
                    </div>
                    <div style={{ padding: '0', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead style={{ background: 'rgba(255,255,255,0.03)' }}>
                                <tr>
                                    <th style={{ textAlign: 'left', padding: '15px 2rem', color: 'var(--text-muted)', fontWeight: 600 }}>MODEL TYPE</th>
                                    <th style={{ textAlign: 'right', padding: '15px 2rem', color: 'var(--text-muted)', fontWeight: 600 }}>F1 SCORE</th>
                                    <th style={{ textAlign: 'right', padding: '15px 2rem', color: 'var(--text-muted)', fontWeight: 600 }}>ACCURACY</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results_table.map((row, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid var(--border-glass)', transition: 'background 0.2s' }}>
                                        <td style={{ padding: '15px 2rem', fontWeight: 600, color: 'var(--text-main)' }}>{row.Model}</td>
                                        <td style={{ padding: '15px 2rem', textAlign: 'right', color: 'var(--accent)', fontWeight: 700 }}>{row['Validation F1']}</td>
                                        <td style={{ padding: '15px 2rem', textAlign: 'right', color: 'var(--success)', fontWeight: 700 }}>{row['Validation Accuracy']}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>

                {/* 3. CONFUSION MATRICES (Stacked Full Width - Grid Inside) */}
                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-glass)' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>CONFUSION MATRICES (ALL MODELS)</h3>
                    </div>
                    <div style={{ padding: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
                        {detailed_results.map((model, idx) => {
                            if (!model.confusion_matrix) return null;
                            const cm = model.confusion_matrix;
                            return (
                                <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-glass)', height: '300px', display: 'flex', flexDirection: 'column' }}>
                                    <div style={{ textAlign: 'center', marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--accent)' }}>{model.name}</div>
                                    <div style={{ flex: 1 }}>
                                        <PlotlyChart
                                            data={[{
                                                z: cm,
                                                x: ['Pred DOWN', 'Pred UP'],
                                                y: ['Act DOWN', 'Act UP'],
                                                type: 'heatmap',
                                                colorscale: 'Blues',
                                                showscale: false,
                                                text: cm.map(row => row.map(String)),
                                                texttemplate: "%{text}",
                                                textfont: { size: 16, color: 'white' }
                                            }]}
                                            layout={{
                                                margin: { t: 0, b: 30, l: 30, r: 0 },
                                                xaxis: { side: 'bottom', color: '#94a3b8' },
                                                yaxis: { color: '#94a3b8' },
                                                paper_bgcolor: 'rgba(0,0,0,0)',
                                                plot_bgcolor: 'rgba(0,0,0,0)',
                                                font: { family: 'JetBrains Mono, monospace' }
                                            }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </motion.div>

                {/* 3.5. ACTUAL VS PREDICTED (Forecast Timeline) */}
                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-glass)' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>FORECAST TIMELINE (ACTUAL VS PREDICTED)</h3>
                    </div>
                    <div style={{ padding: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '1.5rem' }}>
                        {detailed_results.map((model, idx) => {
                            if (!model.pred_vs_actual) return null;
                            const pva = model.pred_vs_actual;
                            const threshold = 0.5;

                            // Calculate marker properties
                            const markerColors = [];
                            const markerSymbols = [];
                            const markerSizes = [];

                            pva.actual.forEach((act, i) => {
                                const prob = pva.probability[i];
                                const pred = prob >= threshold ? 1 : 0;
                                const isCorrect = pred === act;

                                // Color: Green if correct, Red if wrong
                                markerColors.push(isCorrect ? '#10b981' : '#ef4444');

                                // Symbol: Triangle Up if Actual is UP, Triangle Down if Actual is DOWN
                                markerSymbols.push(act === 1 ? 'triangle-up' : 'triangle-down');
                                markerSizes.push(10);
                            });

                            return (
                                <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-glass)', height: '400px', display: 'flex', flexDirection: 'column' }}>
                                    <div style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                                        {model.name}
                                        <span style={{ fontSize: '0.8rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: '10px' }}>
                                            | F1: {model.metrics['Validation F1']} | Acc: {model.metrics['Validation Accuracy']}
                                        </span>
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <PlotlyChart
                                            data={[
                                                // 1. Probability Line
                                                {
                                                    x: pva.dates,
                                                    y: pva.probability,
                                                    type: 'scatter',
                                                    mode: 'lines',
                                                    name: 'Probability',
                                                    line: { color: 'rgba(99, 102, 241, 0.5)', width: 1, dash: 'dot' },
                                                    hoverinfo: 'skip'
                                                },
                                                // 2. Markers
                                                {
                                                    x: pva.dates,
                                                    y: pva.probability,
                                                    type: 'scatter',
                                                    mode: 'markers',
                                                    name: 'Outcome',
                                                    marker: {
                                                        color: markerColors,
                                                        symbol: markerSymbols,
                                                        size: markerSizes,
                                                        line: { color: 'rgba(0,0,0,0.5)', width: 1 }
                                                    },
                                                    hovertemplate:
                                                        '<b>Date</b>: %{x}<br>' +
                                                        '<b>Prob</b>: %{y:.2f}<br>' +
                                                        '<extra></extra>'
                                                }
                                            ]}
                                            layout={{
                                                margin: { t: 30, b: 40, l: 40, r: 20 },
                                                xaxis: { showgrid: false, color: '#94a3b8', type: 'category', tickmode: 'auto', nticks: 5 },
                                                yaxis: { range: [0, 1.05], showgrid: true, gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8', title: 'Probability' },
                                                showlegend: false,
                                                paper_bgcolor: 'rgba(0,0,0,0)',
                                                plot_bgcolor: 'rgba(0,0,0,0)',
                                                font: { family: 'JetBrains Mono, monospace' },
                                                hovermode: 'closest',
                                                shapes: [
                                                    // Predicted UP Zone (Green)
                                                    {
                                                        type: 'rect', xref: 'paper', yref: 'y',
                                                        x0: 0, x1: 1, y0: 0.5, y1: 1,
                                                        fillcolor: 'rgba(16, 185, 129, 0.1)', line: { width: 0 }, layer: 'below'
                                                    },
                                                    // Predicted DOWN Zone (Red)
                                                    {
                                                        type: 'rect', xref: 'paper', yref: 'y',
                                                        x0: 0, x1: 1, y0: 0, y1: 0.5,
                                                        fillcolor: 'rgba(239, 68, 68, 0.1)', line: { width: 0 }, layer: 'below'
                                                    },
                                                    // Threshold Line
                                                    {
                                                        type: 'line', xref: 'paper', yref: 'y',
                                                        x0: 0, x1: 1, y0: 0.5, y1: 0.5,
                                                        line: { color: '#6366f1', width: 2, dash: 'dash' }
                                                    }
                                                ],
                                                annotations: [
                                                    {
                                                        x: 0.5, y: 0.9, xref: 'paper', yref: 'y',
                                                        text: 'PREDICTED UP ZONE',
                                                        showarrow: false,
                                                        font: { size: 12, color: 'rgba(16, 185, 129, 0.8)', weight: 900 }
                                                    },
                                                    {
                                                        x: 0.5, y: 0.1, xref: 'paper', yref: 'y',
                                                        text: 'PREDICTED DOWN ZONE',
                                                        showarrow: false,
                                                        font: { size: 12, color: 'rgba(239, 68, 68, 0.8)', weight: 900 }
                                                    }
                                                ]
                                            }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </motion.div>

                {/* 4. KEY DRIVERS (Stacked Full Width) */}
                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-glass)' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>KEY MARKET DRIVERS</h3>
                    </div>
                    <div style={{ padding: '2rem' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                            {(() => {
                                let model = detailed_results.find(m => m.name === best_model_name);
                                if (!model || !model.feature_importances) model = detailed_results.find(m => m.feature_importances);
                                if (!model || !model.feature_importances) return <div>Data Unavailable</div>;

                                return Object.entries(model.feature_importances)
                                    .sort(([, a], [, b]) => b - a)
                                    .slice(0, 8)
                                    .map(([feat, imp], idx) => (
                                        <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            <div style={{ flex: 1, textAlign: 'right', fontSize: '0.8rem', color: 'var(--text-muted)' }}>{feat}</div>
                                            <div style={{ flex: 1.5 }}>
                                                <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px' }}>
                                                    <div style={{ width: `${imp * 100}%`, height: '100%', background: idx < 3 ? 'var(--primary)' : 'var(--text-muted)', borderRadius: '3px' }} />
                                                </div>
                                            </div>
                                        </div>
                                    ));
                            })()}
                        </div>
                    </div>
                </motion.div>

                {/* 5. CORRELATION (Stacked Full Width) */}
                <motion.div variants={item} className="bento-item col-span-12 glass-card" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-glass)' }}>
                        <h3 className="label-text" style={{ fontSize: '1rem', margin: 0, letterSpacing: '2px' }}>CORRELATION MATRIX</h3>
                    </div>
                    <div style={{ flex: 1, padding: '1rem', overflow: 'hidden' }}>
                        {eda && eda.correlation && (
                            <PlotlyChart
                                data={[{
                                    x: eda.correlation.x, y: eda.correlation.y, z: eda.correlation.z,
                                    type: 'heatmap',
                                    colorscale: 'RdBu',
                                    zmin: -1, zmax: 1,
                                    showscale: true,
                                    colorbar: { thickness: 10, len: 0.8, tickcolor: '#94a3b8', tickfont: { color: '#94a3b8' } }
                                }]}
                                layout={{
                                    margin: { t: 20, b: 80, l: 80, r: 20 },
                                    xaxis: { tickangle: -45, color: '#94a3b8', showgrid: false },
                                    yaxis: { color: '#94a3b8', showgrid: false },
                                    paper_bgcolor: 'rgba(0,0,0,0)',
                                    plot_bgcolor: 'rgba(0,0,0,0)',
                                    font: { family: 'JetBrains Mono, monospace' }
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '100%' }}
                            />
                        )}
                    </div>
                </motion.div>

                {/* 6. HISTOGRAMS (Stacked Full Width) */}
                <motion.div variants={item} className="bento-item col-span-12">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                        {renderHistograms()}
                    </div>
                </motion.div>


                {/* 7. PREDICTION ACTION */}
                <motion.div ref={actionRef} variants={item} className="bento-item col-span-12" style={{ marginTop: '2rem', display: 'flex', justifyContent: 'center' }}>
                    {!showPrediction && (
                        <button
                            onClick={() => setShowPrediction(true)}
                            className="btn-cyber glitch-wrapper"
                            style={{
                                fontSize: '1.2rem', padding: '1.5rem 4rem',
                                maxWidth: '500px', letterSpacing: '0.2em',
                                background: 'var(--bg-card)', backdropFilter: 'blur(10px)',
                                borderRadius: '50px', border: '1px solid var(--primary)'
                            }}
                        >
                            <Zap size={20} style={{ display: 'inline', marginRight: '10px', verticalAlign: 'text-bottom' }} />
                            GENERATE SIGNAL
                        </button>
                    )}
                </motion.div>

                <div className="col-span-12">
                    {showPrediction && prediction_result && (
                        <motion.div
                            ref={predictionRef}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="glass-card"
                            style={{
                                borderRadius: '20px', padding: '0', overflow: 'hidden',
                                background: 'linear-gradient(180deg, rgba(10,10,30,0.9) 0%, rgba(5,5,10,0.95) 100%)',
                                border: `1px solid ${prediction_result.direction === 'UP' ? 'var(--success)' : 'var(--danger)'}`,
                                maxWidth: '800px', margin: '0 auto',
                                position: 'relative'
                            }}
                        >
                            {/* Holographic Header */}
                            <div style={{
                                background: prediction_result.direction === 'UP' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                borderBottom: '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <Target size={18} />
                                    <span style={{ fontSize: '0.8rem', letterSpacing: '2px' }}>SIGNAL DETECTED</span>
                                </div>
                                <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', opacity: 0.7 }}>ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</div>
                            </div>

                            <div style={{ padding: '3rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4rem' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>DIRECTION</div>
                                    <div style={{
                                        fontSize: '4rem', fontWeight: 800,
                                        color: prediction_result.direction === 'UP' ? 'var(--success)' : 'var(--danger)',
                                        textShadow: `0 0 30px ${prediction_result.direction === 'UP' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
                                    }}>
                                        {prediction_result.direction}
                                    </div>
                                </div>

                                <div style={{ width: '1px', height: '100px', background: 'var(--border-glass)' }}></div>

                                <div style={{ flex: 1, maxWidth: '300px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                        <span style={{ fontSize: '0.9rem' }}>CONFIDENCE</span>
                                        <span style={{ fontWeight: 700 }}>{prediction_result.confidence.toFixed(2)}%</span>
                                    </div>
                                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden', marginBottom: '1.5rem' }}>
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${prediction_result.confidence}%` }}
                                            transition={{ duration: 1, ease: "easeOut" }}
                                            style={{ height: '100%', background: prediction_result.direction === 'UP' ? 'var(--success)' : 'var(--danger)' }}
                                        />
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                                        <span style={{ color: 'var(--text-muted)' }}>MODEL</span>
                                        <span>{best_model_name}</span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </div>

            </motion.div>
        </div>
    );
}
