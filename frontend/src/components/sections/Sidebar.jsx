import { Share2, Calendar, Settings, Activity, Zap, Server, Layers, BarChart, TrendingUp } from 'lucide-react';

export function Sidebar({ config, setConfig, onAnalyze, isAnalyzing }) {

    const handleChange = (e) => {
        const { name, value } = e.target;
        setConfig(prev => ({
            ...prev,
            [name]: value
        }));
    };

    return (
        <aside className="sidebar-dock font-mono">
            <div className="brand" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '3rem' }}>
                <div style={{
                    width: '48px', height: '48px',
                    background: 'linear-gradient(135deg, var(--primary), var(--accent))',
                    borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: 'white', boxShadow: '0 0 25px var(--primary-glow)',
                    transform: 'rotate(-5deg)'
                }}>
                    <TrendingUp size={28} fill="none" strokeWidth={2.5} />
                </div>
                <div>
                    <h1 style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1 }}>MARKET</h1>
                    <span style={{ fontSize: '0.75rem', color: 'var(--accent)', letterSpacing: '0.2em' }}>INSIGHT</span>
                </div>
            </div>

            <form onSubmit={onAnalyze} className="no-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', flex: 1, overflowY: 'auto', overflowX: 'hidden', paddingRight: '0.5rem', paddingBottom: '2rem' }}>

                {/* Section 1: Data Source */}
                <div className="control-group">
                    <div className="group-header" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                        <Server size={14} />
                        <span className="label-text" style={{ margin: 0 }}>DATA FEED</span>
                    </div>

                    <div style={{ marginBottom: '1rem' }}>
                        <div style={{ position: 'relative' }}>
                            <Share2 size={16} style={{ position: 'absolute', left: '12px', top: '16px', color: 'var(--text-muted)' }} />
                            <input
                                className="input-field"
                                type="text"
                                name="symbol"
                                value={config.symbol}
                                onChange={handleChange}
                                style={{ paddingLeft: '2.5rem' }}
                                placeholder="SYMBOL"
                                required
                            />
                        </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                        <div>
                            <label className="label-text" style={{ fontSize: '0.7rem' }}>START</label>
                            <input
                                className="input-field"
                                type="date"
                                name="startDate"
                                value={config.startDate}
                                onChange={handleChange}
                                style={{ fontSize: '0.8rem', padding: '0.8rem' }}
                            />
                        </div>
                        <div>
                            <label className="label-text" style={{ fontSize: '0.7rem' }}>END</label>
                            <input
                                className="input-field"
                                type="date"
                                name="endDate"
                                value={config.endDate}
                                onChange={handleChange}
                                style={{ fontSize: '0.8rem', padding: '0.8rem' }}
                            />
                        </div>
                    </div>
                </div>

                {/* Section 2: Validation Strategy */}
                <div className="control-group">
                    <div className="group-header" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                        <Layers size={14} />
                        <span className="label-text" style={{ margin: 0 }}>VALIDATION STRATEGY</span>
                    </div>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span className="label-text">TEST SPLIT</span>
                            <span style={{ color: 'var(--accent)', fontWeight: 600, fontSize: '0.9rem' }}>{config.validationSplit}%</span>
                        </div>
                        <input
                            type="range"
                            name="validationSplit"
                            min="10" max="30"
                            value={config.validationSplit}
                            onChange={handleChange}
                            style={{ width: '100%', accentColor: 'var(--accent)', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', appearance: 'none' }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                            <span>10%</span>
                            <span>30%</span>
                        </div>
                    </div>
                </div>

                {/* Section 3: LSTM Config */}
                <div className="control-group">
                    <div className="group-header" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                        <Activity size={14} />
                        <span className="label-text" style={{ margin: 0 }}>LSTM HYPERPARAMS</span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div>
                            <label className="label-text">EPOCHS</label>
                            <input
                                className="input-field"
                                type="number"
                                name="epochs"
                                value={config.epochs}
                                onChange={handleChange}
                                min="10"
                            />
                        </div>
                        <div>
                            <label className="label-text">BATCH</label>
                            <input
                                className="input-field"
                                type="number"
                                name="batchSize"
                                value={config.batchSize}
                                onChange={handleChange}
                                min="1"
                            />
                        </div>
                        <div className="col-span-2" style={{ gridColumn: 'span 2' }}>
                            <label className="label-text">WINDOW SIZE (DAYS)</label>
                            <input
                                className="input-field"
                                type="number"
                                name="windowSize"
                                value={config.windowSize}
                                onChange={handleChange}
                                min="10"
                            />
                        </div>
                    </div>
                </div>

                <button type="submit" className="btn-cyber" disabled={isAnalyzing} style={{ marginTop: 'auto' }}>
                    {isAnalyzing ? 'PROCESSING...' : 'INITIALIZE SYSTEM'}
                </button>
            </form>

            <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.7rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-glass)', paddingTop: '1rem' }}>
                <div style={{ width: '6px', height: '6px', background: 'var(--success)', borderRadius: '50%', boxShadow: '0 0 10px var(--success)' }}></div>
                SYSTEM ONLINE // V5.1.0
            </div>
        </aside>
    );
}
