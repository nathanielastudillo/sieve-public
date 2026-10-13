import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';

const DANGER_THRESHOLD = 0.1;

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const val = payload[0]?.value;
  const isDanger = val != null && val < DANGER_THRESHOLD;
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: `1px solid ${isDanger ? 'var(--accent-red)' : 'var(--border-light)'}`,
      borderRadius: 'var(--radius-sm)',
      padding: '10px 14px',
      boxShadow: 'var(--shadow-md)',
      fontSize: '0.8rem',
    }}>
      <div style={{ color: 'var(--text-secondary)', fontWeight: 600, marginBottom: 4 }}>
        Generation {label}
      </div>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '0.85rem',
        fontWeight: 600,
        color: isDanger ? 'var(--accent-red)' : 'var(--accent-green)',
      }}>
        {val != null ? val.toFixed(4) : '--'}
      </div>
      {isDanger && (
        <div style={{ color: 'var(--accent-red)', fontSize: '0.7rem', marginTop: 4 }}>
          Below diversity threshold
        </div>
      )}
    </div>
  );
}

export default function DiversityChart({ generations, loading }) {
  const data = useMemo(() => {
    if (!generations || !generations.length) return [];
    return generations.map(g => ({
      generation: g.num,
      diversity: g.population_diversity,
    }));
  }, [generations]);

  const latest = data.length > 0 ? data[data.length - 1] : null;
  const latestVal = latest?.diversity;
  const isDanger = latestVal != null && latestVal < DANGER_THRESHOLD;

  return (
    <div className="card">
      <div className="card-title" style={{ justifyContent: 'space-between' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            display: 'inline-block', width: 3, height: 12,
            borderRadius: 2, background: 'var(--accent-green)',
          }} />
          Population Diversity
        </span>
        {latestVal != null && (
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '0.85rem',
            fontWeight: 600,
            color: isDanger ? 'var(--accent-red)' : 'var(--accent-green)',
            textTransform: 'none',
            letterSpacing: 'normal',
          }}>
            {latestVal.toFixed(3)}
          </span>
        )}
      </div>

      {!data.length ? (
        <div className={`no-data ${loading ? 'loading' : ''}`}>
          {loading ? 'Loading...' : 'No data yet'}
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
            <defs>
              <linearGradient id="diversityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-green)" stopOpacity={0.3} />
                <stop offset="60%" stopColor="var(--accent-yellow)" stopOpacity={0.1} />
                <stop offset="100%" stopColor="var(--accent-red)" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="diversityStroke" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-green)" />
                <stop offset="70%" stopColor="var(--accent-yellow)" />
                <stop offset="100%" stopColor="var(--accent-red)" />
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
              domain={[0, 'auto']}
              stroke="var(--border-light)"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              tickFormatter={v => v.toFixed(2)}
              width={40}
            />

            <Tooltip content={<CustomTooltip />} />

            <ReferenceLine
              y={DANGER_THRESHOLD}
              stroke="var(--accent-red)"
              strokeDasharray="6 3"
              strokeOpacity={0.6}
              label={{
                value: 'Danger',
                position: 'right',
                fill: 'var(--accent-red)',
                fontSize: 10,
                fontFamily: "'JetBrains Mono', monospace",
                opacity: 0.7,
              }}
            />

            <Area
              type="monotone"
              dataKey="diversity"
              stroke="url(#diversityStroke)"
              strokeWidth={2}
              fill="url(#diversityGradient)"
              dot={false}
              activeDot={{ r: 4, fill: 'var(--accent-green)', strokeWidth: 0 }}
              animationDuration={600}
              isAnimationActive={true}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
