import { useState } from 'react';
import { Sidebar } from './components/sections/Sidebar';
import { WelcomeView } from './components/sections/WelcomeView';
import { LoadingView } from './components/sections/LoadingView';
import { DashboardView } from './components/sections/DashboardView';
import { AnimatePresence } from 'framer-motion';

function App() {
  // Config State
  const [config, setConfig] = useState({
    symbol: 'INFY.NS',
    startDate: '2015-01-01', // Updated Logic default
    endDate: '',
    validationSplit: 20,
    epochs: 50,
    batchSize: 32,
    windowSize: 60
  });

  // App Phase State
  const [view, setView] = useState('welcome');
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [analysisData, setAnalysisData] = useState(null);

  const addLog = (msg) => {
    setLogs(prev => [...prev, `NOVA_CORE >> ${msg}`]);
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setView('loading');
    setLogs(['NOVA_CORE >> INITIALIZING SEQUENCE...']);
    setProgress(5);

    // Simulated progress logs for UX
    setTimeout(() => { addLog(`ACQUIRING TARGET FEED: ${config.symbol}...`); setProgress(25); }, 500);
    setTimeout(() => { addLog(`CALCULATING TECHNICAL VECTORS...`); setProgress(45); }, 1500);
    setTimeout(() => { addLog(`TRAINING NEURAL ENSEMBLE...`); setProgress(65); }, 3000);
    setTimeout(() => { addLog(`OPTIMIZING HYPERPARAMETERS...`); setProgress(85); }, 4500);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: config.symbol,
          validation_split: config.validationSplit,
          lstm_epochs: config.epochs,
          lstm_batch_size: config.batchSize,
          lstm_window_size: config.windowSize,
          start_date: config.startDate,
          end_date: config.endDate
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown Server Error' }));
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setAnalysisData(data);
      setProgress(100);
      addLog('ANALYSIS COMPLETE. RENDERING INTERFACE...');

      setTimeout(() => {
        setView('results');
      }, 800);

    } catch (err) {
      console.error(err);
      addLog(`FATAL ERROR: ${err.message}`);
      // alert(err.message); // Don't alert, just log
      setView('welcome');
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        config={config}
        setConfig={setConfig}
        onAnalyze={handleAnalyze}
        isAnalyzing={view === 'loading'}
      />

      <main className="main-viewport">
        <AnimatePresence mode="wait">
          {view === 'welcome' && (
            <WelcomeView key="welcome" />
          )}

          {view === 'loading' && (
            <LoadingView key="loading" logs={logs} progress={progress} />
          )}

          {view === 'results' && analysisData && (
            <DashboardView key="results" data={analysisData} onBack={() => setView('welcome')} />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
