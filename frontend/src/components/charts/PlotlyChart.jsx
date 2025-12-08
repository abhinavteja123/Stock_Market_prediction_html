import Plot from 'react-plotly.js';

export function PlotlyChart({ data, layout, style, config }) {
    const defaultLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Outfit, sans-serif' },
        margin: { t: 30, b: 40, l: 50, r: 20 },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zerolinecolor: 'rgba(255,255,255,0.05)'
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zerolinecolor: 'rgba(255,255,255,0.05)'
        },
        showlegend: false,
        hovermode: 'x unified',
        ...layout
    };

    const defaultConfig = {
        responsive: true,
        displayModeBar: false,
        ...config
    };

    return (
        <Plot
            data={data}
            layout={defaultLayout}
            config={defaultConfig}
            style={{ width: '100%', height: '100%', ...style }}
            useResizeHandler={true}
        />
    );
}
