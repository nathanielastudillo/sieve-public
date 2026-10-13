import React from 'react';

const PILL_COLORS = ['pill-cyan', 'pill-blue', 'pill-purple', 'pill-green', 'pill-yellow', 'pill-orange', 'pill-red'];

function getPillColor(index) {
  return PILL_COLORS[index % PILL_COLORS.length];
}

function truncateId(id) {
  if (!id) return '??';
  return String(id).slice(0, 6);
}

export default function EnsembleLeaderboard({ ensembles, generation, loading }) {
  const sorted = ensembles && ensembles.length > 0
    ? [...ensembles].sort((a, b) => (a.ensemble_val_loss ?? Infinity) - (b.ensemble_val_loss ?? Infinity))
    : [];

  return (
    <div className="card leaderboard-card">
      <div className="card-title">
        Top Ensembles {generation != null ? `(Gen ${generation})` : ''}
      </div>

      {loading && !sorted.length ? (
        <div className="no-data loading">Loading...</div>
      ) : sorted.length === 0 ? (
        <div className="no-data">No ensemble data yet</div>
      ) : (
        <div className="leaderboard-list">
          {sorted.map((ens, i) => {
            const isFirst = i === 0;
            return (
              <div
                key={ens.id || i}
                className={`leaderboard-row ${isFirst ? 'leaderboard-row-best' : ''} ${i % 2 === 0 ? 'leaderboard-row-even' : ''}`}
              >
                <div className="leaderboard-rank">
                  {isFirst ? (
                    <span className="rank-badge rank-gold">1</span>
                  ) : (
                    <span className="rank-badge">{i + 1}</span>
                  )}
                </div>

                <div className="leaderboard-info">
                  <div className="leaderboard-loss">
                    <span className="mono" style={{
                      color: isFirst ? 'var(--accent-cyan)' : 'var(--text-primary)',
                      fontWeight: isFirst ? 600 : 400,
                      fontSize: isFirst ? '0.95rem' : '0.85rem',
                    }}>
                      {ens.ensemble_val_loss != null ? ens.ensemble_val_loss.toFixed(4) : '--'}
                    </span>
                    <span className="leaderboard-meta">
                      div: {ens.diversity_score != null ? ens.diversity_score.toFixed(3) : '--'}
                    </span>
                  </div>

                  <div className="leaderboard-members">
                    {(ens.member_ids || []).slice(0, 8).map((mid, j) => (
                      <span key={mid} className={`pill ${getPillColor(j)}`}>
                        {truncateId(mid)}
                      </span>
                    ))}
                    {(ens.member_ids || []).length > 8 && (
                      <span className="pill">+{ens.member_ids.length - 8}</span>
                    )}
                    {(!ens.member_ids || ens.member_ids.length === 0) && (
                      <span className="leaderboard-meta">
                        {ens.member_ids?.length ?? 0} members
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
