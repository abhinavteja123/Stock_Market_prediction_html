import { motion } from 'framer-motion';
import heroBg from '../../assets/hero-bg.png';

export function WelcomeView() {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="view-container"
            style={{
                height: '100%', display: 'flex', flexDirection: 'column',
                justifyContent: 'center', position: 'relative', overflow: 'hidden'
            }}
        >
            <div
                style={{
                    position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                    backgroundImage: `url(${heroBg})`, backgroundSize: 'cover', backgroundPosition: 'center',
                    opacity: 0.4, mixBlendMode: 'screen', pointerEvents: 'none'
                }}
            />

            <div style={{ position: 'relative', zIndex: 10, maxWidth: '800px', marginLeft: '4rem' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: '100px' }}
                    style={{ height: '4px', background: 'var(--accent)', marginBottom: '2rem' }}
                />

                <motion.h1
                    initial={{ y: 50, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2, duration: 0.8 }}
                    className="gradient-text"
                    style={{ fontSize: '5rem', fontWeight: 800, lineHeight: 1.1, marginBottom: '1.5rem', letterSpacing: '-0.03em' }}
                >
                    PREDICT THE<br />UNPREDICTABLE.
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    style={{ fontSize: '1.25rem', color: '#ffffff', maxWidth: '600px', lineHeight: 1.6, textShadow: '0 2px 4px rgba(0,0,0,0.5)' }}
                >
                    Master the markets with institutional-grade AI.
                    Synthesize complex technicals, sentiment, and volatility into crystal-clear trading signals using our advanced LSTM & XGBoost ensemble.
                </motion.p>

                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="font-mono"
                    style={{ marginTop: '3rem', display: 'flex', gap: '2rem', fontSize: '0.8rem', color: 'var(--accent)' }}
                >
                    <span>// LSTM RECURRENT NETWORKS</span>
                    <span>// XGBOOST GRADIENT BOOSTING</span>
                    <span>// LOGISTIC REGRESSION</span>
                </motion.div>
            </div>
        </motion.div>
    );
}
