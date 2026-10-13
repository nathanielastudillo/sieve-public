import React from 'react';

function formatElapsed(seconds) {
  if (seconds == null || seconds <= 0) return '0m';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  const s = Math.floor(seconds % 60);
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatLoss(val) {
  if (val == null) return '--';
  return val.toFixed(4);
}

function formatDiversity(val) {
  if (val == null) return '--';
  return val.toFixed(3);
}

export default function Header({ status, isConnected, loading }) {
  const s = status || {};
  const isRunning = s.is_running;

  return (
    <div className="header-bar">
      <div className="header-left">
        <div className="app-title">
          <span>SIEVE</span> Evolutionary Ensemble Trainer
        </div>
      </div>

      <div className="header-stats">
        <div className="header-stat header-stat-primary">
          <div className="stat-value large">{s.current_generation ?? '--'}</div>
          <div className="stat-label">Generation</div>
        </div>

        <div className="header-stat-divider" />

        <div className="header-stat header-stat-primary">
          <div className="stat-value large" style={{ color: 'var(--accent-cyan)' }}>
            {formatLoss(s.best_ensemble_loss)}
          </div>
          <div className="stat-label">Best Ensemble Loss</div>
        </div>

        <div className="header-stat-divider" />

        <div className="header-stat">
          <div className="stat-value small">{formatLoss(s.best_individual_loss)}</div>
          <div className="stat-label">Best Individual Loss</div>
        </div>

        <div className="header-stat">
          <div className="stat-value small">{formatElapsed(s.elapsed_seconds)}</div>
          <div className="stat-label">Elapsed</div>
        </div>

        <div className="header-stat">
          <div className="stat-value small">{s.total_members ?? '--'}</div>
          <div className="stat-label">Members</div>
        </div>

        <div className="header-stat">
          <div className="stat-value small">{s.total_ensembles ?? '--'}</div>
          <div className="stat-label">Ensembles</div>
        </div>

        <div className="header-stat">
          <div className="stat-value small">{formatDiversity(s.population_diversity)}</div>
          <div className="stat-label">Diversity</div>
        </div>

        <div className="header-stat header-stat-status">
          <div className="status-indicator">
            <span className={`status-dot ${isRunning ? 'running' : 'idle'}`} />
            <span className="status-text">{isRunning ? 'Running' : 'Idle'}</span>
          </div>
          <div className="stat-label">Status</div>
        </div>
      </div>

      {loading && (
        <div className="header-loading">
          <div className="header-loading-bar" />
        </div>
      )}
    </div>
  );
}
