import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-light)',
      borderRadius: 'var(--radius-sm)',
      padding: '10px 14px',
      boxShadow: 'var(--shadow-md)',
      fontSize: '0.8rem',
    }}>
      <div style={{ color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 4 }}>
        Generation {label}
      </div>
      {payload.map((entry, i) => (
        <div key={i} style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '2px 0',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '0.75rem',
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: entry.color, flexShrink: 0,
          }} />
          <span style={{ color: 'var(--text-secondary)' }}>{entry.name}:</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
            {entry.value != null ? entry.value.toFixed(6) : '--'}
          </span>
        </div>
      ))}
    </div>
  );
}

function LatestLabel({ viewBox, value, color }) {
  if (!viewBox || value == null) return null;
  return (
    <text
      x={viewBox.x + 8}
      y={viewBox.y}
      fill={color}
      fontSize={11}
      fontFamily="'JetBrains Mono', monospace"
      fontWeight={600}
      dominantBaseline="middle"
    >
      {value.toFixed(4)}
    </text>
  );
}

export default function LossCurve({ generations, loading }) {
  const data = useMemo(() => {
    if (!generations || !generations.length) return [];
    return generations.map(g => ({
      generation: g.num,
      ensemble: g.best_ensemble_loss,
      individual: g.best_individual_loss,
    }));
  }, [generations]);

  const latest = data.length > 0 ? data[data.length - 1] : null;

  const yDomain = useMemo(() => {
    if (!data.length) return [0, 1];
    const allVals = data.flatMap(d => [d.ensemble, d.individual].filter(v => v != null));
    if (!allVals.length) return [0, 1];
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);
    const pad = (max - min) * 0.1 || 0.01;
    return [Math.max(0, min - pad), max + pad];
  }, [data]);

  return (
    <div className="card">
      <div className="card-title">Loss Curve</div>
      {!data.length ? (
        <div className={`no-data ${loading ? 'loading' : ''}`}>
          {loading ? 'Loading...' : 'No generation data yet'}
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={data} margin={{ top: 8, right: 60, bottom: 4, left: 8 }}>
            <defs>
              <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity={0.12} />
                <stop offset="100%" stopColor="var(--accent-cyan)" stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />

            <XAxis
              dataKey="generation"
              stroke="var(--border-light)"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
            />
            <YAxis
              domain={yDomain}
              stroke="var(--border-light)"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              tickFormatter={v => v.toFixed(3)}
              width={55}
            />

            <Tooltip content={<CustomTooltip />} />

            <Legend
              wrapperStyle={{ fontSize: '0.75rem', paddingTop: 8 }}
              iconType="circle"
              iconSize={8}
            />

            <Area
              type="monotone"
              dataKey="ensemble"
              fill="url(#lossFill)"
              stroke="none"
              name="Ensemble Loss"
              animationDuration={600}
              isAnimationActive={true}
            />

            <Line
              type="monotone"
              dataKey="ensemble"
              stroke="var(--accent-cyan)"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4, fill: 'var(--accent-cyan)', strokeWidth: 0 }}
              name="Ensemble Loss"
              animationDuration={600}
              isAnimationActive={true}
            />

            <Line
              type="monotone"
              dataKey="individual"
              stroke="var(--accent-blue)"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              activeDot={{ r: 3, fill: 'var(--accent-blue)', strokeWidth: 0 }}
              name="Individual Loss"
              animationDuration={600}
              isAnimationActive={true}
              opacity={0.7}
            />

            {latest && (
              <ReferenceLine
                y={latest.ensemble}
                stroke="var(--accent-cyan)"
                strokeDasharray="2 4"
                strokeOpacity={0.4}
                label={<LatestLabel value={latest.ensemble} color="var(--accent-cyan)" />}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
