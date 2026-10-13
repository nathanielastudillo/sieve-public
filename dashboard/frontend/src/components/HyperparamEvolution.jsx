import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Legend,
} from 'recharts';

const STACK_COLORS = [
  'var(--accent-cyan)',
  'var(--accent-blue)',
  'var(--accent-purple)',
  'var(--accent-green)',
  'var(--accent-yellow)',
  'var(--accent-orange)',
  'var(--accent-red)',
  '#ec4899',
];

function NumericParamChart({ name, data }) {
  return (
    <div className="hp-chart-card">
      <div className="hp-chart-title">{name}</div>
      <ResponsiveContainer width="100%" height={140}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id={`hp-fill-${name}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity={0.15} />
              <stop offset="100%" stopColor="var(--accent-cyan)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="generation"
            tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={v => {
              if (Math.abs(v) < 0.01 || Math.abs(v) >= 1000) return v.toExponential(1);
              return v.toPrecision(3);
            }}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.72rem',
              fontFamily: "'JetBrains Mono', monospace",
            }}
            labelFormatter={v => `Gen ${v}`}
            formatter={(v) => v != null ? v.toPrecision(4) : '--'}
          />
          {/* Range area (min to max) */}
          <Area
            type="monotone"
            dataKey="range"
            fill={`url(#hp-fill-${name})`}
            stroke="none"
            animationDuration={400}
            isAnimationActive={true}
          />
          {/* Min line */}
          <Line
            type="monotone"
            dataKey="min"
            stroke="var(--accent-cyan)"
            strokeWidth={1}
            strokeOpacity={0.3}
            dot={false}
            animationDuration={400}
          />
          {/* Max line */}
          <Line
            type="monotone"
            dataKey="max"
            stroke="var(--accent-cyan)"
            strokeWidth={1}
            strokeOpacity={0.3}
            dot={false}
            animationDuration={400}
          />
          {/* Mean line */}
          <Line
            type="monotone"
            dataKey="mean"
            stroke="var(--accent-cyan)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: 'var(--accent-cyan)', strokeWidth: 0 }}
            animationDuration={400}
          />
          {/* Median line */}
          <Line
            type="monotone"
            dataKey="median"
            stroke="var(--accent-blue)"
            strokeWidth={1.5}
            strokeDasharray="3 2"
            dot={false}
            opacity={0.6}
            animationDuration={400}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function CategoricalParamChart({ name, data, categories }) {
  return (
    <div className="hp-chart-card">
      <div className="hp-chart-title">{name}</div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="generation"
            tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
            tickLine={false}
            axisLine={false}
            width={30}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.72rem',
              fontFamily: "'JetBrains Mono', monospace",
            }}
            labelFormatter={v => `Gen ${v}`}
          />
          {categories.map((cat, i) => (
            <Bar
              key={cat}
              dataKey={cat}
              stackId="stack"
              fill={STACK_COLORS[i % STACK_COLORS.length]}
              opacity={0.8}
              animationDuration={400}
            />
          ))}
          {categories.length <= 6 && (
            <Legend
              wrapperStyle={{ fontSize: '0.65rem', paddingTop: 4 }}
              iconType="square"
              iconSize={8}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function HyperparamEvolution({ hyperparams, loading }) {
  const numericCharts = useMemo(() => {
    if (!hyperparams?.numeric) return [];
    return Object.entries(hyperparams.numeric).map(([name, points]) => {
      const data = points.map(p => ({
        generation: p.generation,
        min: p.min,
        max: p.max,
        mean: p.mean,
        median: p.median,
        std: p.std,
        // Area needs array [min, max] for range display
        range: [p.min, p.max],
      }));
      return { name, data };
    });
  }, [hyperparams]);

  const categoricalCharts = useMemo(() => {
    if (!hyperparams?.categorical) return [];
    return Object.entries(hyperparams.categorical).map(([name, points]) => {
      // Collect all category values across generations
      const allCategories = new Set();
      points.forEach(p => {
        if (p.counts) Object.keys(p.counts).forEach(k => allCategories.add(k));
      });
      const categories = [...allCategories].sort();

      const data = points.map(p => {
        const row = { generation: p.generation };
        categories.forEach(cat => {
          row[cat] = p.counts?.[cat] || 0;
        });
        return row;
      });

      return { name, data, categories };
    });
  }, [hyperparams]);

  const hasData = numericCharts.length > 0 || categoricalCharts.length > 0;

  return (
    <div className="card hp-card">
      <div className="card-title">Hyperparameter Evolution</div>

      {loading && !hasData ? (
        <div className="no-data loading">Loading...</div>
      ) : !hasData ? (
        <div className="no-data">No hyperparameter data yet</div>
      ) : (
        <div className="hp-grid">
          {numericCharts.map(({ name, data }) => (
            <NumericParamChart key={name} name={name} data={data} />
          ))}
          {categoricalCharts.map(({ name, data, categories }) => (
            <CategoricalParamChart key={name} name={name} data={data} categories={categories} />
          ))}
        </div>
      )}
    </div>
  );
}
