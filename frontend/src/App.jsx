import { useState, useEffect } from 'react'

function App() {
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [modelName, setModelName] = useState('Loading...');
  const [stats, setStats] = useState({ detections_count: 0, unique_classes: [] });

  useEffect(() => {
    // Check backend status periodically
    const checkStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/status');
        if (response.ok) {
          const data = await response.json();
          setIsBackendOnline(true);
          setModelName(data.model);
        } else {
          setIsBackendOnline(false);
        }
      } catch (error) {
        setIsBackendOnline(false);
      }
    };

    checkStatus();
    
    // Status can be checked every 5s
    const statusInterval = setInterval(checkStatus, 5000);
    
    // SSE for real-time detections
    const eventSource = new EventSource('http://localhost:5000/detections');
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStats(data);
    };

    eventSource.onerror = () => {
      console.error("SSE connection failed.");
      eventSource.close();
    };
    
    return () => {
      clearInterval(statusInterval);
      eventSource.close();
    };
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <div>
          <h1 className="title">YOLOv26 Analytics</h1>
          <p style={{ color: '#9ca3af', marginTop: '0.5rem' }}>Real-time Object Detection Pipeline</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className={`status-badge ${isBackendOnline ? 'online' : 'offline'}`}>
            <span className=""></span>
            {isBackendOnline ? 'System Online' : 'System Offline'}
          </div>
          <div className="controls" style={{ marginTop: 0 }}>
            {isStreaming ? (
              <button
                onClick={() => setIsStreaming(false)}
                style={{ width: '100%', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.5)', color: '#ef4444', borderRadius: '9999px' }}
              >
                Stop Streaming
              </button>
            ) : (
              <button
                onClick={() => setIsStreaming(true)}
                style={{ width: '100%', background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.5)', color: '#10b981', borderRadius: '9999px' }}
              >
                Start Streaming
              </button>
            )}
          </div>
        </div>
      </header>


      <main className="main-content">
        <div className="video-container">
          {isBackendOnline && isStreaming ? (
            <img
              src="http://localhost:5000/video_feed"
              alt="YOLO Video Stream"
              className="video-stream"
              onError={() => setIsBackendOnline(false)}
            />
          ) : (
            <div className="video-placeholder">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p>{!isStreaming ? 'Stream Paused' : 'Waiting for video stream...'}</p>
              <p style={{ fontSize: '0.875rem', opacity: 0.7, marginTop: '0.5rem' }}>
                {!isStreaming ? 'Click Start Stream to resume' : 'Please ensure backend server is running.'}
              </p>
            </div>
          )}
        </div>

        <div className="info-panel">
          <div className="card">
            <h3>Active Model</h3>
            <div className="stat-value">{isBackendOnline ? modelName : '---'}</div>
            <div className="stat-label" style={{ marginTop: '0.5rem' }}>Ultralytics Engine</div>
          </div>
          {/*           
          <div className="card">
            <h3>Current Detections</h3>
            <div className="stat-value" style={{ color: stats.detections_count > 0 ? '#4facfe' : '#fff' }}>
              {isBackendOnline ? stats.detections_count : '0'}
            </div>
            <div className="stat-label" style={{ marginTop: '0.5rem' }}>Objects in frame</div>
          </div> */}

          <div className="card">
            <h3>Detections</h3>
            <div className="stat-value" style={{ fontSize: '1.25rem' }}>
              {isBackendOnline ? (stats.summary || 'None') : 'None'}
            </div>
            <div className="stat-label" style={{ marginTop: '0.5rem' }}>Real-time breakdown</div>
          </div>

          <div className="card">
            <h3>Inference Status</h3>
            <div className="stat-value" style={{ color: isBackendOnline && isStreaming ? '#10b981' : '#6b7280', fontSize: '1.5rem' }}>
              {isBackendOnline ? (isStreaming ? 'Active' : 'Paused') : 'Idle'}
            </div>
            <div className="stat-label" style={{ marginTop: '0.5rem' }}>Stream processing</div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
