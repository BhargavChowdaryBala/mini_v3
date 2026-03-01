import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, ClipboardList, Activity, Bus, AlertCircle, RotateCcw, X, ChevronDown, Smartphone, FileVideo, Construction } from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

function App() {
    const [logs, setLogs] = useState([]);
    const [status, setStatus] = useState('System Ready');
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('feed'); // 'feed' or 'logs'
    const [useMobileCamera, setUseMobileCamera] = useState(false);
    const [processorImage, setProcessorImage] = useState(null);
    const [lastUpdateTime, setLastUpdateTime] = useState(new Date().toLocaleTimeString());
    const [availableCameras, setAvailableCameras] = useState([]);
    const [showCameraMenu, setShowCameraMenu] = useState(false);
    const [currentCamId, setCurrentCamId] = useState(0);
    const videoRef = React.useRef(null);
    const canvasRef = React.useRef(null);
    const fileInputRef = React.useRef(null);
    const [isVideoMode, setIsVideoMode] = useState(false); // Legacy - keep for minimal compatibility if needed elsewhere, but effectively unused
    const [currentPage, setCurrentPage] = useState('dashboard'); // 'dashboard' or 'image-checker'
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [uploadPreview, setUploadPreview] = useState(null);
    const [connectionError, setConnectionError] = useState(false);
    const [appMode, setAppMode] = useState('live'); // 'live' or 'upload'
    const [videoFile, setVideoFile] = useState(null);
    const [videoProcessId, setVideoProcessId] = useState(null);
    const [vProcessingStatus, setVProcessingStatus] = useState('idle');
    const [vProgress, setVProgress] = useState(0);
    const [vPlatesDetected, setVPlatesDetected] = useState(0);

    const videoPreviewUrl = React.useMemo(() => {
        if (!videoFile) return null;
        return URL.createObjectURL(videoFile);
    }, [videoFile]);

    const aiFeedUrl = React.useMemo(() => {
        if (vProcessingStatus !== 'processing') return null;
        return `${API_BASE}/upload_video_feed?run=${Date.now()}`;
    }, [vProcessingStatus]);

    useEffect(() => {
        const fetchCameras = async () => {
            try {
                const res = await axios.get(`${API_BASE}/api/list_cameras`);
                setAvailableCameras(res.data);
            } catch (err) {
                console.error("Error fetching cameras:", err);
            }
        };
        fetchCameras();

        const fetchData = async () => {
            try {
                // Determine source filter based on appMode
                const sourceQuery = appMode === 'upload' ? 'upload' : 'live';
                const [logsRes, statusRes] = await Promise.all([
                    axios.get(`${API_BASE}/api/logs?source=${sourceQuery}`),
                    axios.get(`${API_BASE}/api/status`)
                ]);
                setLogs(logsRes.data);
                setLastUpdateTime(new Date().toLocaleTimeString());
                if (!useMobileCamera) setStatus(statusRes.data.status);
                setConnectionError(false);
                setLoading(false);
            } catch (err) {
                console.error("Error fetching data:", err);
                setConnectionError(true);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 3000); // Relaxed interval
        return () => clearInterval(interval);
    }, [useMobileCamera, appMode]);

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


    const handleImageCheck = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show preview immediately
        const reader = new FileReader();
        reader.onload = () => setUploadPreview(reader.result);
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('image', file);

        setIsAnalyzing(true);
        setAnalysisResult(null);
        try {
            const res = await axios.post(`${API_BASE}/api/check_image`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setAnalysisResult(res.data);
        } catch (err) {
            console.error("Analysis Error:", err);
            alert("Failed to analyze image");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleResetCamera = async (index = 0) => {
        try {
            await axios.post(`${API_BASE}/api/reset_camera`, { index });
            setIsVideoMode(false);
            setUseMobileCamera(false);
            setCurrentCamId(index);
            setShowCameraMenu(false);
            setStatus(`Switched to Camera ${index}`);
        } catch (err) {
            console.error("Reset Error:", err);
        }
    };

    const handleVideoFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setVideoFile(file);
        setVProcessingStatus('idle');
        setVProgress(0);
    };

    const startVideoAnalysis = async () => {
        if (!videoFile) return;

        const formData = new FormData();
        formData.append('video', videoFile);

        try {
            setVProcessingStatus('uploading');
            setVPlatesDetected(0);
            setVProgress(0);

            // Actual file upload with progress tracking
            const uploadRes = await axios.post(`${API_BASE}/api/upload_video`, formData, {
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setVProgress(percentCompleted);
                }
            });

            const filePath = uploadRes.data.file_path;
            setVProcessingStatus('processing');
            setVProgress(0); // Reset for analysis phase

            await axios.post(`${API_BASE}/api/process_uploaded_video`, { file_path: filePath });

            // Poll for analysis status
            const poll = setInterval(async () => {
                try {
                    const [statusRes, logsRes] = await Promise.all([
                        axios.get(`${API_BASE}/api/upload_status`),
                        axios.get(`${API_BASE}/api/logs?source=upload`)
                    ]);

                    setVProgress(statusRes.data.progress);

                    // Update global logs state with the latest upload logs
                    setLogs(logsRes.data);

                    // Count plates specifically for the current viewing context
                    const currentLogs = logsRes.data.filter(l => l.source && l.source.includes('upload'));
                    setVPlatesDetected(currentLogs.length);

                    if (statusRes.data.status === 'complete') {
                        setVProcessingStatus('complete');
                        clearInterval(poll);
                    } else if (statusRes.data.status === 'error') {
                        setVProcessingStatus('error');
                        clearInterval(poll);
                    }
                } catch (e) {
                    console.error("Poll error:", e);
                }
            }, 1000);
        } catch (err) {
            console.error("Video analysis error:", err);
            setVProcessingStatus('error');
        }
    };

    return (
        <div className="dashboard">
            <AnimatePresence mode="wait">
                {currentPage === 'dashboard' ? (
                    <motion.div
                        key="dashboard"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <header className="header">
                            {connectionError && (
                                <motion.div
                                    className="connection-warning"
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                >
                                    <AlertCircle size={16} />
                                    <span>Disconnected from Backend - Attempting to reconnect...</span>
                                </motion.div>
                            )}
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

                            <div className="mode-tabs">
                                <button
                                    className={`mode-tab ${appMode === 'live' ? 'active' : ''}`}
                                    onClick={() => setAppMode('live')}
                                >
                                    LIVE MONITORING
                                </button>
                                <button
                                    className={`mode-tab ${appMode === 'upload' ? 'active' : ''}`}
                                    onClick={() => setAppMode('upload')}
                                >
                                    VIDEO ANALYSIS
                                </button>
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
                            {appMode === 'live' ? (
                                <section className={`feed-section ${activeTab === 'feed' ? 'active' : ''}`}>
                                    <motion.div
                                        className="glass-card feed-container"
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ duration: 0.5 }}
                                    >
                                        <div className="feed-header">
                                            <div className="flex-row">
                                                <div className="source-info">
                                                    <span className="source-label">
                                                        CCTV FEED (CAM 0)
                                                    </span>
                                                    <div className="activity-tag">
                                                        <Activity size={10} className="pulse" />
                                                        <span>Live Analytics</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="wip-container">
                                            <div className="wip-content">
                                                <Construction size={64} color="#00f2fe" opacity={0.5} />
                                                <h2>Work in Progress</h2>
                                                <p>Live Monitoring is currently being optimized.</p>
                                                <div className="wip-hint">Please use VIDEO ANALYSIS tab for processing</div>
                                            </div>
                                        </div>
                                    </motion.div>
                                </section>
                            ) : (
                                <section className="upload-mode-section">
                                    <motion.div
                                        className="glass-card upload-video-card"
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                    >
                                        <div className="card-header">
                                            <Bus size={20} color="#00f2fe" />
                                            <span>VIDEO UPLOAD ANALYSIS</span>
                                        </div>

                                        <div className="upload-controls">
                                            <div className="file-zone" onClick={() => fileInputRef.current.click()}>
                                                {videoFile ? (
                                                    <div className="file-info">
                                                        <Activity className="pulse" />
                                                        <span>{videoFile.name} ({(videoFile.size / (1024 * 1024)).toFixed(1)} MB)</span>
                                                    </div>
                                                ) : (
                                                    <div className="drop-hint">
                                                        <RotateCcw size={40} opacity={0.3} />
                                                        <p>Select video file to analyze (.mp4, .avi)</p>
                                                    </div>
                                                )}
                                                <input
                                                    type="file"
                                                    accept="video/*"
                                                    ref={fileInputRef}
                                                    style={{ display: 'none' }}
                                                    onChange={handleVideoFileUpload}
                                                />
                                            </div>

                                            <button
                                                className={`start-btn ${vProcessingStatus === 'processing' ? 'loading' : ''}`}
                                                disabled={!videoFile || vProcessingStatus === 'processing'}
                                                onClick={startVideoAnalysis}
                                            >
                                                {vProcessingStatus === 'processing' ? 'Analysing Video...' : 'Start Intelligence Analysis'}
                                            </button>
                                        </div>

                                        {vProcessingStatus !== 'idle' && (
                                            <div className="progress-container">
                                                <div className="status-label">
                                                    <span>
                                                        {vProcessingStatus === 'complete'
                                                            ? `ANALYSIS COMPLETE - ${vPlatesDetected} PLATES DETECTED`
                                                            : `Status: ${vProcessingStatus.toUpperCase()}`}
                                                    </span>
                                                    <span>{vProgress}%</span>
                                                </div>
                                                <div className="progress-bar">
                                                    <motion.div
                                                        className="progress-fill"
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${vProgress}%` }}
                                                    />
                                                </div>
                                                {vProcessingStatus === 'processing' && (
                                                    <div className="live-metric-hint">
                                                        <Activity size={10} className="pulse" />
                                                        <span>{vPlatesDetected} identified so far...</span>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </motion.div>

                                    {videoFile && (
                                        <motion.div
                                            className="glass-card video-preview-card"
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.5 }}
                                        >
                                            <div className="preview-header">
                                                <div className="preview-label">
                                                    {vProcessingStatus === 'processing' ? (
                                                        <span className="flex-row gap-2">
                                                            <div className="pulse-dot"></div>
                                                            LIVE AI ANALYTICS FEED
                                                        </span>
                                                    ) : 'VIDEO SOURCE PREVIEW'}
                                                    ```
                                                </div>
                                            </div>

                                            <div className="preview-container">
                                                {vProcessingStatus === 'processing' ? (
                                                    <img
                                                        className="upload-preview-player ai-feed"
                                                        src={aiFeedUrl}
                                                        alt="AI Analysis Feed"
                                                        onError={(e) => {
                                                            // Auto-retry once if stream isn't ready
                                                            setTimeout(() => {
                                                                if (e.target) e.target.src = `${aiFeedUrl}&retry=${Date.now()}`;
                                                            }, 1000);
                                                        }}
                                                    />
                                                ) : (
                                                    <video
                                                        className="upload-preview-player"
                                                        src={videoPreviewUrl}
                                                        controls
                                                        autoPlay
                                                        muted
                                                    />
                                                )}
                                            </div>
                                        </motion.div>
                                    )}
                                </section>
                            )}

                            <aside className={`sidebar ${activeTab === 'logs' ? 'active' : ''}`}>
                                <motion.div
                                    className="glass-card log-table-container"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                >
                                    <div className="card-header">
                                        <div className="flex-row">
                                            <ClipboardList size={20} color="#00f2fe" />
                                            <span>ENTRY / EXIT LOGS</span>
                                        </div>
                                        <span className="last-sync">Updated: {lastUpdateTime}</span>
                                    </div>

                                    <div className="table-scroll">
                                        <table className="log-table">
                                            <thead>
                                                <tr>
                                                    <th>Bus ID</th>
                                                    <th>Registration</th>
                                                    <th>Status</th>
                                                    <th>Time</th>
                                                    {appMode === 'upload' && <th>Source</th>}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <AnimatePresence>
                                                    {logs
                                                        .filter(log => {
                                                            const source = log.source || '';
                                                            return appMode === 'upload' ? source.includes('upload') : !source.includes('upload');
                                                        })
                                                        .map((log, index) => (
                                                            <motion.tr
                                                                key={log._id || index}
                                                                initial={{ opacity: 0, x: -10 }}
                                                                animate={{ opacity: 1, x: 0 }}
                                                                exit={{ opacity: 0 }}
                                                                transition={{ delay: index * 0.05 }}
                                                            >
                                                                <td className="bus-id-text">#{log.bus_id || 'N/A'}</td>
                                                                <td className={`plate-text ${(log.registration_number || '').includes('missed') ? 'warning-text' : ''}`}>
                                                                    {log.registration_number}
                                                                </td>
                                                                <td>
                                                                    <span className={`status-pill ${log.status}`}>
                                                                        {log.status}
                                                                    </span>
                                                                </td>
                                                                <td className="time-text">{log.timestamp}</td>
                                                                {appMode === 'upload' && <td className="source-tag">{log.source}</td>}
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
                    </motion.div>
                ) : (
                    <motion.div
                        key="image-checker"
                        className="image-checker-page"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                    >
                        <header className="page-header">
                            <div className="title-block">
                                <h1>Image Checker</h1>
                                <p>Static License Plate Analysis</p>
                            </div>
                            <button className="top-back-btn" onClick={() => setCurrentPage('dashboard')}>
                                <X size={18} />
                                <span>Exit to Home</span>
                            </button>
                        </header>

                        <div className="checker-grid">
                            <motion.div
                                className="glass-card upload-card"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                            >
                                <div className="card-header">
                                    <Camera size={18} color="#00f2fe" />
                                    <span>UPLOAD FOR ANALYSIS</span>
                                </div>
                                <div
                                    className="upload-dropzone"
                                    onClick={() => fileInputRef.current.click()}
                                >
                                    {uploadPreview ? (
                                        <img src={uploadPreview} alt="Preview" className="preview-img" />
                                    ) : (
                                        <div className="dropzone-content">
                                            <FileVideo size={40} opacity={0.3} />
                                            <p>Drop image here or click to browse</p>
                                        </div>
                                    )}
                                </div>
                                <input
                                    type="file"
                                    accept="image/*"
                                    ref={fileInputRef}
                                    style={{ display: 'none' }}
                                    onChange={handleImageCheck}
                                />
                                <button
                                    className="analyze-btn"
                                    disabled={isAnalyzing}
                                    onClick={() => fileInputRef.current.click()}
                                >
                                    {isAnalyzing ? 'Analyzing...' : 'Select Image'}
                                </button>
                            </motion.div>

                            {analysisResult ? (
                                <motion.div
                                    className="glass-card result-card"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                >
                                    <div className="card-header-flex">
                                        <h2>Analysis Result</h2>
                                        {analysisResult.confidence > 70 && (
                                            <span className="conf-pill high">
                                                <Activity size={12} /> High Confidence
                                            </span>
                                        )}
                                    </div>

                                    <div className="result-content">
                                        <div className="result-sub-card">
                                            <span className="sub-label">ORIGINAL UPLOAD</span>
                                            <div className="img-holder">
                                                <img src={uploadPreview} alt="Original" />
                                            </div>
                                        </div>

                                        <div className="result-sub-card">
                                            <span className="sub-label">DETECTED PLATE</span>
                                            <div className="img-holder plate-crop">
                                                {analysisResult.plate_image ? (
                                                    <img src={`data:image/jpeg;base64,${analysisResult.plate_image}`} alt="Plate Crop" />
                                                ) : (
                                                    <div className="error-placeholder">No Plate Detected</div>
                                                )}
                                            </div>
                                        </div>

                                        <div className="info-box">
                                            <span className="sub-label">LICENSE NO.</span>
                                            <div className="plate-id-big">{analysisResult.plate_text}</div>
                                        </div>

                                        <div className="info-box">
                                            <div className="flex-between">
                                                <span className="sub-label">CONFIDENCE SCORE</span>
                                                <span className="conf-value">{analysisResult.confidence}%</span>
                                            </div>
                                            <div className="progress-bar">
                                                <motion.div
                                                    className="progress-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${analysisResult.confidence}%` }}
                                                />
                                            </div>
                                        </div>

                                        <div className="metrics-box">
                                            <span className="sub-label">RAW OCR OUTPUT</span>
                                            <div className="raw-text-list">
                                                {analysisResult.raw_texts && analysisResult.raw_texts.length > 0 ? (
                                                    analysisResult.raw_texts.map((t, i) => (
                                                        <span key={i} className="raw-text-tag">{t}</span>
                                                    ))
                                                ) : (
                                                    <span className="metric-val">No raw text found</span>
                                                )}
                                            </div>
                                        </div>

                                        <div className="metrics-box">
                                            <span className="sub-label">PERFORMANCE METRICS</span>
                                            <div className="metric-row">
                                                <span>YOLO Processing:</span>
                                                <span className="metric-val">{analysisResult.metrics.yolo} ms</span>
                                            </div>
                                            <div className="metric-row">
                                                <span>OCR Processing:</span>
                                                <span className="metric-val purple">{analysisResult.metrics.ocr} ms</span>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            ) : (
                                <div className="glass-card empty-result">
                                    <Activity size={48} opacity={0.1} />
                                    <p>{isAnalyzing ? 'Processing AI models...' : 'Results will appear after analysis'}</p>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <footer className="footer">
                <p>© 2026 AI Transportation Solutions - IEEE Research Demo</p>
            </footer>

            <button
                className="floating-next-btn"
                onClick={() => setCurrentPage(currentPage === 'dashboard' ? 'image-checker' : 'dashboard')}
            >
                <span>{currentPage === 'dashboard' ? 'Image Checker' : 'Back to Dashboard'}</span>
                <ChevronDown size={16} style={{ transform: 'rotate(-90deg)' }} />
            </button>
        </div >

    );
}

export default App;
