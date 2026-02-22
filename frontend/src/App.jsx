import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, ClipboardList, Activity, Bus, AlertCircle } from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

function App() {
    const [logs, setLogs] = useState([]);
    const [status, setStatus] = useState('System Ready');
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('feed'); // 'feed' or 'logs'
    const [useMobileCamera, setUseMobileCamera] = useState(false);
    const [processorImage, setProcessorImage] = useState(null);
    const videoRef = React.useRef(null);
    const canvasRef = React.useRef(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [logsRes, statusRes] = await Promise.all([
                    axios.get(`${API_BASE}/api/logs`),
                    axios.get(`${API_BASE}/api/status`)
                ]);
                setLogs(logsRes.data);
                if (!useMobileCamera) setStatus(statusRes.data.status);
                setLoading(false);
            } catch (err) {
                console.error("Error fetching data:", err);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, [useMobileCamera]);

    // Mobile Camera logic
    useEffect(() => {
        let stream = null;
        let captureInterval = null;

        if (useMobileCamera) {
            const startCamera = async () => {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment' }
                    });
                    if (videoRef.current) videoRef.current.srcObject = stream;

                    // Start frame capture loop
                    captureInterval = setInterval(async () => {
                        if (videoRef.current && canvasRef.current) {
                            const canvas = canvasRef.current;
                            const video = videoRef.current;
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            const ctx = canvas.getContext('2d');
                            ctx.drawImage(video, 0, 0);

                            const base64Image = canvas.toDataURL('image/jpeg', 0.6);
                            try {
                                const res = await axios.post(`${API_BASE}/api/process_frame`, {
                                    image: base64Image
                                });
                                setProcessorImage(res.data.image);
                                setStatus(res.data.status);
                            } catch (err) {
                                console.error("OCR Error:", err);
                            }
                        }
                    }, 1000); // Process once per second for stability
                } catch (err) {
                    console.error("Camera access error:", err);
                    setUseMobileCamera(false);
                }
            };
            startCamera();
        }

        return () => {
            if (stream) stream.getTracks().forEach(t => t.stop());
            if (captureInterval) clearInterval(captureInterval);
        };
    }, [useMobileCamera]);

    return (
        <div className="dashboard">
            <header className="header">
                <div className="title-section">
                    <motion.h1
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        Bus Monitoring System
                    </motion.h1>
                    <p>Real-time Entry & Exit Tracking Dashboard</p>
                </div>

                <div className="glass-card status-card">
                    <div className={`status-indicator ${status.includes('Processing') ? 'processing' : ''}`}></div>
                    <span className="status-text">{status}</span>
                </div>
            </header>

            <nav className="mobile-tabs">
                <button
                    className={activeTab === 'feed' ? 'active' : ''}
                    onClick={() => setActiveTab('feed')}
                >
                    <Camera size={20} />
                    <span>Live Feed</span>
                </button>
                <button
                    className={activeTab === 'logs' ? 'active' : ''}
                    onClick={() => setActiveTab('logs')}
                >
                    <ClipboardList size={20} />
                    <span>Activity Logs</span>
                </button>
            </nav>

            <main className="main-grid">
                <section className={`feed-section ${activeTab === 'feed' ? 'active' : ''}`}>
                    <motion.div
                        className="glass-card feed-container"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                    >
                        <div className="feed-header">
                            <div className="flex-row">
                                <Camera size={20} color="#00f2fe" />
                                <span>{useMobileCamera ? 'MOBILE BACK CAMERA' : 'GATE CCTV FEED'}</span>
                            </div>
                            <button
                                className={`camera-toggle ${useMobileCamera ? 'active' : ''}`}
                                onClick={() => setUseMobileCamera(!useMobileCamera)}
                            >
                                {useMobileCamera ? 'Switch to CCTV' : 'Use Mobile Camera'}
                            </button>
                        </div>

                        {useMobileCamera ? (
                            <div className="mobile-cam-preview">
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    style={{ display: 'none' }}
                                />
                                <img
                                    src={processorImage || 'https://via.placeholder.com/1280x720/0a0a0c/ffffff?text=Initializing+Mobile+Vision...'}
                                    alt="Processed Mobile Feed"
                                    className="feed-image"
                                />
                                <canvas ref={canvasRef} style={{ display: 'none' }} />
                            </div>
                        ) : (
                            <img
                                src={`${API_BASE}/video_feed`}
                                alt="Live Feed"
                                className="feed-image"
                                onError={(e) => {
                                    e.target.src = 'https://via.placeholder.com/1280x720/0a0a0c/ffffff?text=Camera+Feed+Offline';
                                }}
                            />
                        )}

                        <div className="feed-overlay">
                            <Activity size={16} className="pulse" />
                            <span>{useMobileCamera ? 'Edge Processing' : 'Live Integration'}</span>
                        </div>
                    </motion.div>
                </section>

                <aside className={`sidebar ${activeTab === 'logs' ? 'active' : ''}`}>
                    <motion.div
                        className="glass-card log-table-container"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <div className="card-header">
                            <ClipboardList size={20} color="#00f2fe" />
                            <span>ENTRY / EXIT LOGS</span>
                        </div>

                        <div className="table-scroll">
                            <table className="log-table">
                                <thead>
                                    <tr>
                                        <th>Registration</th>
                                        <th>Status</th>
                                        <th>Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <AnimatePresence>
                                        {logs.map((log, index) => (
                                            <motion.tr
                                                key={log._id || index}
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0 }}
                                                transition={{ delay: index * 0.05 }}
                                            >
                                                <td className="plate-text">{log.registration_number}</td>
                                                <td>
                                                    <span className={`status-pill ${log.status}`}>
                                                        {log.status}
                                                    </span>
                                                </td>
                                                <td className="time-text">{log.timestamp}</td>
                                            </motion.tr>
                                        ))}
                                    </AnimatePresence>
                                </tbody>
                            </table>
                            {logs.length === 0 && !loading && (
                                <div className="no-data">
                                    <Activity size={40} opacity={0.2} />
                                    <p>No activity detected yet</p>
                                </div>
                            )}
                        </div>
                    </motion.div>
                </aside>
            </main>

            <footer className="footer">
                <p>© 2026 AI Transportation Solutions - IEEE Research Demo</p>
            </footer>
        </div>
    );
}

export default App;
