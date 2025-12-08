import { motion } from 'framer-motion';
import { Terminal } from 'lucide-react';

export function LoadingView({ logs, progress }) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
                height: '100%', display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center'
            }}
        >
            <div className="loader-dna"></div>

            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Processing Market Data...</h3>

            <div style={{ width: '400px', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden', marginBottom: '2rem' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ ease: "easeInOut" }}
                    style={{ height: '100%', background: 'var(--accent)' }}
                />
            </div>

            <div className="glass-panel" style={{ width: '500px', height: '200px', padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>
                    <Terminal size={14} color="var(--text-secondary)" />
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>System Logs</span>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {logs.map((log, i) => (
                        <div key={i}>{log}</div>
                    ))}
                </div>
            </div>
        </motion.div>
    );
}
