import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ZAxis,
} from 'recharts';

const OPTIMIZER_COLORS = {
  adam: 'var(--accent-cyan)',
  adamw: 'var(--accent-blue)',
  sgd: 'var(--accent-green)',
  rmsprop: 'var(--accent-purple)',
  adagrad: 'var(--accent-yellow)',
};

function getOptimizerColor(config) {
  if (!config) return 'var(--accent-cyan)';
  const opt = (config.optimizer || config.optim || '').toLowerCase();
  return OPTIMIZER_COLORS[opt] || 'var(--accent-orange)';
}

function truncateId(id) {
  if (!id) return '??';
  return String(id).slice(0, 8);
}

function ScatterTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-light)',
      borderRadius: 'var(--radius-sm)',
      padding: '10px 14px',
      boxShadow: 'var(--shadow-md)',
      fontSize: '0.8rem',
      maxWidth: 260,
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        color: 'var(--text-primary)',
        fontWeight: 600,
        marginBottom: 6,
      }}>
        {truncateId(d.id)}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Row label="Val Loss" value={d.val_loss?.toFixed(6)} />
        <Row label="Ensemble Contrib" value={d.ensemble_contribution?.toFixed(4)} />
        {d.config && Object.entries(d.config).slice(0, 5).map(([k, v]) => (
          <Row key={k} label={k} value={typeof v === 'number' ? v.toPrecision(4) : String(v)} />
        ))}
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      gap: 12,
      fontSize: '0.72rem',
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{
        color: 'var(--text-primary)',
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {value ?? '--'}
      </span>
    </div>
  );
}

function MembersTable({ members }) {
  const sorted = useMemo(() =>
    [...members].sort((a, b) => (a.val_loss ?? Infinity) - (b.val_loss ?? Infinity)),
    [members]
  );

  return (
    <div className="members-table-wrap">
      <table className="members-table">
        <thead>
          <tr>
            <th>#</th>
            <th>ID</th>
            <th>Val Loss</th>
            <th>Optimizer</th>
            <th>LR</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => (
            <tr key={m.id} className={i === 0 ? 'best-row' : ''}>
              <td className="rank">{i + 1}</td>
              <td className="mono">{truncateId(m.id)}</td>
              <td className="mono">{m.val_loss?.toFixed(6) ?? '--'}</td>
              <td>{m.config?.optimizer || m.config?.optim || '--'}</td>
              <td className="mono">{m.config?.lr?.toExponential(2) || m.config?.learning_rate?.toExponential(2) || '--'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PopulationView({ scatter, members, generation, loading }) {
  const hasScatter = scatter && scatter.length > 0 && scatter.some(s => s.ensemble_contribution != null);

  const scatterData = useMemo(() => {
    if (!hasScatter) return [];
    return scatter.map(s => ({
      ...s,
      color: getOptimizerColor(s.config),
    }));
  }, [scatter, hasScatter]);

  return (
    <div className="card population-card">
      <div className="card-title">
        Population {generation != null ? `(Gen ${generation})` : ''}
      </div>

      {loading && (!members || !members.length) ? (
        <div className="no-data loading">Loading...</div>
      ) : hasScatter ? (
        <ResponsiveContainer width="100%" height={310}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="val_loss"
              name="Val Loss"
              stroke="var(--border-light)"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              label={{
                value: 'Val Loss',
                position: 'insideBottom',
                offset: -2,
                fill: 'var(--text-muted)',
                fontSize: 11,
              }}
              tickFormatter={v => v.toFixed(3)}
            />
            <YAxis
              dataKey="ensemble_contribution"
              name="Ensemble Contribution"
              stroke="var(--border-light)"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              label={{
                value: 'Ensemble Contrib',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                fill: 'var(--text-muted)',
                fontSize: 11,
              }}
              tickFormatter={v => v.toFixed(2)}
              width={50}
            />
            <ZAxis range={[30, 80]} />
            <Tooltip content={<ScatterTooltip />} />

            {/* Render one Scatter per optimizer for color */}
            {Object.entries(
              scatterData.reduce((acc, d) => {
                const opt = d.config?.optimizer || d.config?.optim || 'unknown';
                if (!acc[opt]) acc[opt] = [];
                acc[opt].push(d);
                return acc;
              }, {})
            ).map(([opt, points]) => (
              <Scatter
                key={opt}
                name={opt}
                data={points}
                fill={OPTIMIZER_COLORS[opt.toLowerCase()] || 'var(--accent-orange)'}
                opacity={0.8}
                animationDuration={400}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      ) : members && members.length > 0 ? (
        <MembersTable members={members} />
      ) : (
        <div className="no-data">No population data yet</div>
      )}
    </div>
  );
}
