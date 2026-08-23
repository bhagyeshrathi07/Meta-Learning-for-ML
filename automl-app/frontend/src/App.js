import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Leaderboard from './Leaderboard';

// In production (Docker), use same origin. In development, use localhost:5000
const API_BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:5000';

const MODELS_CONFIG = {
  "Classification": ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost", "Decision Tree"],
  "Regression": ["Linear Regression", "Random Forest", "SVM", "XGBoost", "Decision Tree"]
};

function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');

  // API Key State
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('apiKey') || '');
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [isValidatingKey, setIsValidatingKey] = useState(false);
  const [keyError, setKeyError] = useState('');

  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [columns, setColumns] = useState([]);
  const [dataStats, setDataStats] = useState(null);

  // Default to null so we can hide the section initially
  const [detectedTaskType, setDetectedTaskType] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false); // UI state for analysis loading

  const [selectedModels, setSelectedModels] = useState([]);

  // Task Management State
  const [taskId, setTaskId] = useState(null);
  const [completedTaskId, setCompletedTaskId] = useState(null);

  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Create axios instance with API key header
  const api = axios.create({
    baseURL: API_BASE_URL,
    headers: apiKey ? { 'X-API-Key': apiKey } : {}
  });

  // Validate API key on app load if one exists in localStorage
  useEffect(() => {
    if (apiKey) {
      validateApiKey(apiKey);
    }
  }, []);

  const validateApiKey = async (key) => {
    setIsValidatingKey(true);
    setKeyError('');
    try {
      const res = await axios.post(`${API_BASE_URL}/validate-key`, {}, {
        headers: { 'X-API-Key': key }
      });
      if (res.data.valid) {
        setApiKey(key);
        localStorage.setItem('apiKey', key);
        setKeyError('');
      }
    } catch (err) {
      setKeyError(err.response?.data?.message || 'Invalid API key');
      setApiKey('');
      localStorage.removeItem('apiKey');
    } finally {
      setIsValidatingKey(false);
    }
  };

  const handleApiKeySubmit = (e) => {
    e.preventDefault();
    if (apiKeyInput.trim()) {
      validateApiKey(apiKeyInput.trim());
    }
  };

  const handleLogout = () => {
    setApiKey('');
    setApiKeyInput('');
    localStorage.removeItem('apiKey');
    // Reset app state
    setFile(null);
    setTarget('');
    setColumns([]);
    setDataStats(null);
    setDetectedTaskType(null);
    setResults(null);
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // --- AUTOMATIC TASK TYPE DETECTION ---
  useEffect(() => {
    if (!file || !target) {
      setDetectedTaskType(null);
      setSelectedModels([]);
      return;
    }

    setIsDetecting(true);

    // Parse the first 1000 rows to analyze the target column
    Papa.parse(file, {
      header: true,
      preview: 1000,
      complete: (results) => {
        const rows = results.data;
        // Extract target values, filtering out empty/undefined
        const values = rows.map(r => r[target]).filter(v => v !== null && v !== undefined && v !== '');

        if (values.length === 0) {
          setIsDetecting(false);
          return;
        }

        // 1. Check if values are numeric
        const isNumeric = values.every(val => !isNaN(parseFloat(val)) && isFinite(val));

        // 2. Check cardinality (number of unique values)
        const uniqueValues = new Set(values).size;

        // Heuristic: If numeric and has many unique values relative to sample -> Regression
        // Otherwise -> Classification
        let type = "Classification";
        if (isNumeric && uniqueValues > 10) {
          type = "Regression";
        }

        setDetectedTaskType(type);
        setSelectedModels(MODELS_CONFIG[type]); // Auto-select relevant models
        setIsDetecting(false);
      }
    });

  }, [target, file]);


  // --- POLLING LOGIC ---
  useEffect(() => {
    let interval = null;
    if (taskId) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE_URL}/status/${taskId}`, {
            headers: { 'X-API-Key': apiKey }
          });
          const data = res.data;
          setProgress(data.progress);
          setLogs(data.logs || []);

          if (data.status === 'completed') {
            setResults(data.results);
            setCompletedTaskId(taskId);
            setLoading(false);
            setTaskId(null);
          } else if (data.status === 'failed') {
            setError(data.error);
            setLoading(false);
            setTaskId(null);
          }
        } catch (err) {
          console.error("Polling Error:", err);
          if (err.response?.status === 401) {
            setError('API key invalid or expired. Please re-authenticate.');
            setLoading(false);
            setTaskId(null);
            handleLogout();
          }
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [taskId, apiKey]);

  // --- FILE HANDLING ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    // Reset states on new file
    setFile(selectedFile);
    setColumns([]);
    setTarget('');
    setDataStats(null);
    setDetectedTaskType(null); // Hide model zoo initially

    if (selectedFile) {
      let sizeString = selectedFile.size / 1024 < 1024 ? `${(selectedFile.size / 1024).toFixed(1)} KB` : `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB`;
      Papa.parse(selectedFile, {
        header: true, preview: 1,
        complete: (res) => {
          if (res.meta && res.meta.fields) {
            setColumns(res.meta.fields);
            setDataStats({ rows: "Calculating...", cols: res.meta.fields.length, size: sizeString });

            // Fast Row Estimation Logic
            if (selectedFile.size < 50 * 1024) {
              Papa.parse(selectedFile, { header: true, complete: (r) => setDataStats(p => ({ ...p, rows: r.data.length.toLocaleString() })) });
            } else {
              const reader = new FileReader();
              reader.onload = (ev) => {
                const lines = ev.target.result.split('\n').length;
                const estimated = Math.floor(selectedFile.size / (50 * 1024 / lines));
                setDataStats(p => ({ ...p, rows: `~${estimated.toLocaleString()}` }));
              };
              reader.readAsText(selectedFile.slice(0, 50 * 1024));
            }
          }
        }
      });
    }
  };

  const handleModelToggle = (m) => setSelectedModels(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);
    // Send the task type explicitly to backend if needed, or just models
    formData.append('task_type', detectedTaskType);
    formData.append('models', JSON.stringify(selectedModels));

    setLoading(true);
    setError('');
    setResults(null);
    setCompletedTaskId(null);
    setLogs(["🚀 Initializing Upload..."]);
    setProgress(0);

    try {
      const res = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'X-API-Key': apiKey }
      });
      setTaskId(res.data.task_id);
    } catch (err) {
      if (err.response?.status === 401) {
        setError('API key invalid or expired. Please re-authenticate.');
        handleLogout();
      } else {
        setError(err.response?.data?.error || 'Upload Failed');
      }
      setLoading(false);
    }
  };

  // Allow manual override of task type
  const handleTaskTypeChange = (e) => {
    const newType = e.target.value;
    setDetectedTaskType(newType);
    setSelectedModels(MODELS_CONFIG[newType]);
  }

  // --- API KEY LOGIN SCREEN ---
  if (!apiKey) {
    return (
      <div className="container py-5">
        <div className="row justify-content-center">
          <div className="col-md-6 col-lg-5">
            <div className="glass-card p-4 p-md-5 animate__animated animate__fadeInUp">
              <div className="text-center mb-4">
                <h1 className="fw-bold mb-2" style={{
                  background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  AutoML
                </h1>
                <p className="text-muted">Enter your API key to continue</p>
              </div>

              <form onSubmit={handleApiKeySubmit}>
                <div className="mb-3">
                  <label className="form-label fw-bold text-uppercase small text-muted">API Key</label>
                  <input
                    type="password"
                    className="form-control"
                    placeholder="Enter your API key"
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    disabled={isValidatingKey}
                    autoFocus
                  />
                </div>

                {keyError && (
                  <div className="alert alert-danger py-2 rounded-3 border-0 shadow-sm mb-3">
                    {keyError}
                  </div>
                )}

                <button
                  type="submit"
                  className="btn btn-primary w-100 py-2"
                  disabled={isValidatingKey || !apiKeyInput.trim()}
                >
                  {isValidatingKey ? (
                    <span><span className="spinner-border spinner-border-sm me-2"></span> Validating...</span>
                  ) : (
                    'Authenticate'
                  )}
                </button>
              </form>

              <div className="text-center mt-4">
                <div onClick={() => setDarkMode(!darkMode)} className="d-inline-block" style={{ cursor: 'pointer' }}>
                  <span style={{ fontSize: '1.2rem' }}>{darkMode ? '🌙' : '☀️'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container py-5">
      {/* HEADER */}
      <div className="d-flex justify-content-between align-items-center mb-5">
        <div>
          <h1 className="fw-bold mb-0" style={{
            background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            AutoML <span className="fw-light text-muted" style={{ WebkitTextFillColor: darkMode ? '#ccc' : '#555' }}>Comparator</span>
          </h1>
          <p className="text-muted mb-0">Upload CSV, Select Target, Relax.</p>
        </div>

        <div className="d-flex align-items-center gap-3">
          <button onClick={handleLogout} className="btn btn-outline-secondary btn-sm" title="Logout">
            Logout
          </button>
          <div onClick={() => setDarkMode(!darkMode)} className="glass-card d-flex align-items-center justify-content-center"
            style={{ width: '50px', height: '50px', cursor: 'pointer' }}>
            <span style={{ fontSize: '1.5rem' }}>{darkMode ? '🌙' : '☀️'}</span>
          </div>
        </div>
      </div>

      {/* MAIN INPUT CARD */}
      <div className="glass-card p-4 p-md-5 mb-5 animate__animated animate__fadeInUp">
        <form onSubmit={handleSubmit}>
          <div className="row g-4">
            {/* 1. File Upload */}
            <div className="col-md-6">
              <label className="form-label fw-bold text-uppercase small text-muted tracking-wide">1. Dataset Source</label>
              <div className="input-group">
                <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
              </div>
            </div>

            {/* 2. Target Selection */}
            <div className="col-md-6">
              <label className="form-label fw-bold text-uppercase small text-muted tracking-wide">2. Target Variable</label>
              <input type="text" className="form-control" list="cols" value={target} onChange={e => setTarget(e.target.value)} disabled={!file} placeholder="Which column to predict?" />
              <datalist id="cols">{columns.map(c => <option key={c} value={c} />)}</datalist>
            </div>
          </div>

          {/* STATS BADGES */}
          {dataStats && (
            <div className="d-flex gap-3 mt-4 justify-content-start">
              <div className="px-3 py-2 rounded-3 bg-primary bg-opacity-10 border border-primary text-primary">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Rows</small>
                <strong className="font-monospace">{dataStats.rows}</strong>
              </div>
              <div className="px-3 py-2 rounded-3 bg-success bg-opacity-10 border border-success text-success">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Columns</small>
                <strong className="font-monospace">{dataStats.cols}</strong>
              </div>
              <div className="px-3 py-2 rounded-3 bg-info bg-opacity-10 border border-info text-info">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Size</small>
                <strong className="font-monospace">{dataStats.size}</strong>
              </div>
            </div>
          )}

          <hr className="my-4 opacity-25" />

          {/* LOADING STATE FOR DETECTION */}
          {isDetecting && (
            <div className="text-center py-4">
              <div className="spinner-border text-primary mb-2" role="status"></div>
              <p className="text-muted small">Analyzing target column to determine task type...</p>
            </div>
          )}

          {/* 3. MODEL SELECTION (HIDDEN UNTIL DETECTED) */}
          {detectedTaskType && !isDetecting && (
            <div className="mb-4 animate__animated animate__fadeIn">
              <div className="d-flex justify-content-between align-items-center mb-3">
                <div className='d-flex align-items-center gap-2'>
                  <label className="form-label fw-bold text-uppercase small text-muted tracking-wide mb-0">3. Model Zoo</label>
                  <span className="badge bg-secondary bg-opacity-25 text-body border">Detected: {detectedTaskType}</span>
                </div>

                <select className="form-select w-auto form-select-sm py-1" value={detectedTaskType} onChange={handleTaskTypeChange}>
                  <option value="Classification">🎯 Classification</option>
                  <option value="Regression">📈 Regression</option>
                </select>
              </div>

              <div className="d-flex flex-wrap gap-2">
                {MODELS_CONFIG[detectedTaskType].map(m => (
                  <div key={m} onClick={() => handleModelToggle(m)}
                    className={`px-3 py-2 rounded-pill border cursor-pointer transition-all ${selectedModels.includes(m)
                      ? 'bg-primary text-white border-primary shadow-sm'
                      : 'bg-transparent text-muted border-secondary'
                      }`}
                    style={{ cursor: 'pointer', fontSize: '0.9rem', transition: '0.2s' }}>
                    {selectedModels.includes(m) && <span className="me-2">✓</span>}
                    {m}
                  </div>
                ))}
              </div>
            </div>
          )}

          <button type="submit" className="btn btn-primary w-100 py-3 shadow-lg" disabled={loading || !file || !target || !detectedTaskType}>
            {loading ? (
              <span><span className="spinner-border spinner-border-sm me-2"></span> Processing Pipeline...</span>
            ) : (
              <span className="h5 mb-0">🚀 Launch Experiments</span>
            )}
          </button>
        </form>

        {/* LOGS & PROGRESS */}
        {loading && (
          <div className="mt-4 animate__animated animate__fadeIn">
            <div className="progress" style={{ height: '6px', borderRadius: '10px', backgroundColor: 'rgba(255,255,255,0.1)' }}>
              <div className="progress-bar bg-gradient-primary" style={{ width: `${progress}%`, transition: 'width 0.5s ease' }}></div>
            </div>
            <div className="mt-3 p-3 rounded bg-black bg-opacity-25 font-monospace small text-muted" style={{ maxHeight: '120px', overflowY: 'auto' }}>
              {logs.map((l, i) => <div key={i} className="mb-1"> {l}</div>)}
              <div className="text-primary blink">_</div>
            </div>
          </div>
        )}

        {error && <div className="alert alert-danger mt-3 rounded-3 border-0 shadow-sm">{error}</div>}
      </div>

      {results && <Leaderboard results={results} darkMode={darkMode} taskId={completedTaskId} apiKey={apiKey} />}
    </div>
  );
}

export default App;
