import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ConfusionMatrix from './ConfusionMatrix';
import ROCChart from './ROCChart';
import RegressionChart from './RegressionChart';

// In production (Docker), use same origin. In development, use localhost:5000
const API_BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:5000';

const Leaderboard = ({ results, darkMode, taskId, apiKey }) => {

  // Determine default sort key based on task type
  const defaultSort = (results && results.length > 0 && results[0]["Task Type"] === "Regression")
    ? 'R2 Score' : 'Accuracy';

  const [sortKey, setSortKey] = useState(defaultSort);
  const [sortedResults, setSortedResults] = useState([]);

  // Helper to format large regression errors (e.g., 5040733 -> 5.04M)
  const formatError = (num) => {
    if (!num) return "-";
    if (num > 1000000) return (num / 1000000).toFixed(2) + "M";
    if (num > 1000) return (num / 1000).toFixed(2) + "k";
    return num.toFixed(2);
  };

  // Reset sort key if the dataset/task type changes
  useEffect(() => {
    if (results && results.length > 0) {
      const type = results[0]["Task Type"];
      setSortKey(type === "Classification" ? 'Accuracy' : 'R2 Score');
    }
  }, [results]);

  // Sort logic: handles both ascending (errors/time) and descending (scores)
  useEffect(() => {
    if (!results || results.length === 0) return;
    const sorted = [...results].sort((a, b) => {
      const valA = a[sortKey] !== undefined ? a[sortKey] : -Infinity;
      const valB = b[sortKey] !== undefined ? b[sortKey] : -Infinity;

      // Metrics where LOWER is better
      if (['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)', 'Max CPU (%)'].includes(sortKey)) {
        return valA - valB;
      }
      // Metrics where HIGHER is better
      return valB - valA;
    });
    setSortedResults(sorted);
  }, [results, sortKey]);

  if (!results || results.length === 0 || sortedResults.length === 0) return null;

  const bestModel = sortedResults[0];
  const taskType = bestModel["Task Type"] || "Classification";
  const isClassification = taskType === "Classification";

  // Handle authenticated downloads
  const handleDownload = async (url, filename) => {
    try {
      const response = await axios.get(url, {
        headers: { 'X-API-Key': apiKey },
        responseType: 'blob'
      });
      const blob = new Blob([response.data]);
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Download failed. Please check your API key.');
    }
  };

  const handleDownloadModel = (modelName) => {
    handleDownload(
      `${API_BASE_URL}/download?model=${encodeURIComponent(modelName)}`,
      `${modelName.replace(/ /g, '_')}.pkl`
    );
  };

  const handleDownloadCode = (modelName) => {
    handleDownload(
      `${API_BASE_URL}/download-code?model=${encodeURIComponent(modelName)}&task_id=${taskId}`,
      `train_${modelName.replace(/ /g, '_')}.py`
    );
  };

  // Reusable Header Component with sorting arrow logic
  const SortableHeader = ({ label, metricKey }) => {
    const isActive = sortKey === metricKey;
    return (
      <th
        className={`align-middle ${isActive ? 'active-head' : ''}`}
        style={{ cursor: 'pointer', minWidth: '120px' }}
        onClick={() => setSortKey(metricKey)}
      >
        <div className="d-flex justify-content-between align-items-center">
          <span>{label}</span>
          <span style={{ opacity: isActive ? 1 : 0.3, transition: 'opacity 0.2s' }}>
            {['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)', 'Max CPU (%)'].includes(metricKey) ? '▲' : '▼'}
          </span>
        </div>
      </th>
    );
  };

  return (
    <div className="animate__animated animate__fadeIn mt-5">
      {/* Header Section */}
      <div className="d-flex align-items-center justify-content-between mb-4">
        <h3 className="m-0 fw-bold" style={{ background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          🏆 Leaderboard <span className={`badge ms-2 fs-6 ${isClassification ? 'bg-primary bg-opacity-25 text-primary border border-primary' : 'bg-warning bg-opacity-25 text-warning border border-warning'}`} style={{ WebkitTextFillColor: 'initial' }}>{taskType}</span>
        </h3>

        {/* Sort Dropdown */}
        <div className="d-flex align-items-center glass-card px-3 py-2">
          <label className="me-2 fw-bold text-muted small text-uppercase">Sort By:</label>
          <select className="form-select form-select-sm w-auto border-0 bg-transparent py-0 fw-bold text-primary" style={{ boxShadow: 'none' }} value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
            {isClassification ? (
              <>
                <option value="Accuracy">Accuracy</option>
                <option value="F1 Score">F1 Score</option>
                <option value="Precision">Precision</option>
                <option value="Recall">Recall</option>
              </>
            ) : (
              <>
                <option value="R2 Score">R2 Score</option>
                <option value="RMSE">RMSE (Low is good)</option>
              </>
            )}
            <option value="Training Time (s)">Time (Fastest)</option>
            <option value="Max RAM (MB)">RAM (Efficiency)</option>
            <option value="Max CPU (%)">CPU (Efficiency)</option>
          </select>
        </div>
      </div>

      {/* Results Table */}
      <div className="table-responsive" style={{ overflowX: 'visible' }}>
        <table className="table table-modern align-middle mb-0">
          <thead>
            <tr>
              <th className="text-center ps-4" style={{ width: '70px' }}>#</th>
              <th>Model Name</th>

              {isClassification ? (
                <>
                  <SortableHeader label="Accuracy" metricKey="Accuracy" />
                  <SortableHeader label="F1 Score" metricKey="F1 Score" />
                  <SortableHeader label="Precision" metricKey="Precision" />
                  <SortableHeader label="Recall" metricKey="Recall" />
                </>
              ) : (
                <>
                  <SortableHeader label="R2 Score" metricKey="R2 Score" />
                  <SortableHeader label="MAE" metricKey="MAE" />
                  <SortableHeader label="RMSE" metricKey="RMSE" />
                </>
              )}

              <SortableHeader label="Time" metricKey="Training Time (s)" />
              <SortableHeader label="RAM" metricKey="Max RAM (MB)" />
              <SortableHeader label="CPU" metricKey="Max CPU (%)" />
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((r, i) => {
              const isWinner = i === 0;
              return (
                <tr key={i} className={isWinner ? 'fw-bold' : 'text-muted'}>
                  {/* Rank Column */}
                  <td className="text-center">
                    {isWinner ? <span className="fs-3" style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))' }}>🥇</span> : <span className="badge bg-secondary bg-opacity-25 text-secondary rounded-circle p-2" style={{ width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{i + 1}</span>}
                  </td>

                  {/* Model Name Column */}
                  <td>
                    <div className="d-flex align-items-center">
                      <span className={isWinner ? 'fs-5' : ''}>{r.Model}</span>
                      {isWinner && <span className="badge badge-best-glass ms-3 shadow-sm" style={{ fontSize: '0.65rem' }}>BEST</span>}
                    </div>
                  </td>

                  {/* Metrics Columns */}
                  {isClassification ? (
                    <>
                      <td className={sortKey === 'Accuracy' ? 'text-primary fw-bolder fs-5' : ''}>{(r.Accuracy * 100).toFixed(2)}%</td>
                      <td className={sortKey === 'F1 Score' ? 'text-primary fw-bolder fs-5' : ''}>{(r["F1 Score"] * 100).toFixed(2)}%</td>
                      <td className={sortKey === 'Precision' ? 'text-primary fw-bolder fs-5' : ''}>{(r["Precision"] * 100).toFixed(2)}%</td>
                      <td className={sortKey === 'Recall' ? 'text-primary fw-bolder fs-5' : ''}>{(r["Recall"] * 100).toFixed(2)}%</td>
                    </>
                  ) : (
                    <>
                      <td className={sortKey === 'R2 Score' ? 'text-primary fw-bolder fs-5' : ''}>{r["R2 Score"]}</td>
                      <td>{formatError(r["MAE"])}</td>
                      <td>{formatError(r["RMSE"])}</td>
                    </>
                  )}

                  <td className={sortKey === 'Training Time (s)' ? 'text-primary fw-bold' : ''}>{r["Training Time (s)"]}s</td>
                  <td className={sortKey === 'Max RAM (MB)' ? 'text-primary fw-bold' : ''}>{r["Max RAM (MB)"]} MB</td>
                  <td className={sortKey === 'Max CPU (%)' ? 'text-primary fw-bold' : ''}>{Number(r["Max CPU (%)"]).toFixed(2)}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Detail Card for Best Model */}
      {bestModel && (
        <div className="glass-card mt-5 p-0 overflow-hidden">
          <div className="p-4 bg-success bg-gradient text-white fw-bold d-flex justify-content-between align-items-center" style={{ backdropFilter: 'none' }}>
            <span className="fs-5">✨ Recommended Model: {bestModel.Model}</span>
            <span className="badge bg-white text-success shadow-sm py-2 px-3">Rank #1 Overall</span>
          </div>
          <div className="p-4">
            <div className="row g-4">
              {/* Hyperparameters */}
              <div className="col-md-7">
                <h6 className="fw-bold text-success text-uppercase small letter-spacing-1">⚙️ Best Hyperparameters</h6>
                <pre className="p-3 rounded-3 border border-secondary border-opacity-25 shadow-inset" style={{ backgroundColor: darkMode ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)', fontSize: '0.85rem', maxHeight: '150px', overflowY: 'auto' }}>
                  {JSON.stringify(bestModel["Best Params"], null, 2)}
                </pre>
              </div>

              {/* Action Buttons */}
              <div className="col-md-5 d-flex flex-column gap-3 justify-content-center font-monospace">
                <button onClick={() => handleDownloadModel(bestModel.Model)} className="btn btn-success w-100 shadow-lg py-3">📦 Download Model (.pkl)</button>
                <button onClick={() => handleDownloadCode(bestModel.Model)} className="btn btn-outline-primary w-100 py-3" style={{ borderWidth: '2px' }}>📜 Generate Python Script</button>
              </div>
            </div>

            {/* Visualizations */}
            <div className="row mt-5 pt-4 border-top border-secondary border-opacity-10">
              <div className="col-12 mb-4 fw-bold text-uppercase small text-muted letter-spacing-1">📊 Visual Analysis ({bestModel.Model})</div>
              {isClassification ? (
                <>
                  <div className="col-md-5 d-flex justify-content-center">
                    <ConfusionMatrix data={bestModel.ConfusionMatrix} darkMode={darkMode} />
                  </div>
                  <div className="col-md-7">
                    <ROCChart data={bestModel.ROCData} auc={bestModel.AUC} darkMode={darkMode} />
                  </div>
                </>
              ) : (
                <div className="col-12">
                  <RegressionChart data={bestModel.ScatterData} r2={bestModel["R2 Score"]} darkMode={darkMode} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Leaderboard;