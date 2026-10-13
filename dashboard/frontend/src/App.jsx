import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import Header from './components/Header';
import LossCurve from './components/LossCurve';
import DiversityChart from './components/DiversityChart';
import PopulationView from './components/PopulationView';
import EnsembleLeaderboard from './components/EnsembleLeaderboard';
import HyperparamEvolution from './components/HyperparamEvolution';

const API_BASE = '/api';

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export default function App() {
  const { lastMessage, isConnected, connectionState } = useWebSocket();

  const [status, setStatus] = useState(null);
  const [generations, setGenerations] = useState([]);
  const [members, setMembers] = useState([]);
  const [ensembles, setEnsembles] = useState([]);
  const [hyperparams, setHyperparams] = useState(null);
  const [scatter, setScatter] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const currentGenRef = useRef(null);

  // Fetch all data
  const fetchAll = useCallback(async () => {
    try {
      const [statusData, genData, hpData] = await Promise.all([
        fetchJson('/status'),
        fetchJson('/generations'),
        fetchJson('/hyperparams'),
      ]);

      setStatus(statusData);
      setGenerations(genData);
      setHyperparams(hpData);
      setError(null);

      const gen = statusData.current_generation;
      if (gen != null && gen > 0) {
        currentGenRef.current = gen;
        const [memberData, ensembleData, scatterData] = await Promise.all([
          fetchJson(`/members?generation=${gen}`),
          fetchJson(`/ensembles?generation=${gen}&top_k=10`),
          fetchJson(`/population_scatter?generation=${gen}`),
        ]);
        setMembers(memberData);
        setEnsembles(ensembleData);
        setScatter(scatterData);
      }

      setLoading(false);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // React to WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    const { type, data } = lastMessage;

    if (type === 'status_update') {
      setStatus(data);
    }

    if (type === 'generation_update') {
      // Re-fetch everything when a new generation completes
      fetchAll();
    }
  }, [lastMessage, fetchAll]);

  // Poll fallback: refetch every 30s if WS disconnected
  useEffect(() => {
    if (isConnected) return;
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, [isConnected, fetchAll]);

  return (
    <>
      <div className={`connection-bar ${connectionState}`}>
        <span className="dot" />
        {connectionState === 'connected' && 'Connected'}
        {connectionState === 'connecting' && 'Reconnecting...'}
        {connectionState === 'disconnected' && 'Disconnected -- retrying'}
      </div>

      <div className="dashboard">
        {error && (
          <div className="error-banner">
            Failed to load data: {error}
          </div>
        )}

        <div className="dashboard-header">
          <Header status={status} isConnected={isConnected} loading={loading} />
        </div>

        <div className="dashboard-charts-row">
          <LossCurve generations={generations} loading={loading} />
          <DiversityChart generations={generations} loading={loading} />
        </div>

        <div className="dashboard-middle-row">
          <PopulationView
            scatter={scatter}
            members={members}
            generation={currentGenRef.current}
            loading={loading}
          />
          <EnsembleLeaderboard
            ensembles={ensembles}
            generation={currentGenRef.current}
            loading={loading}
          />
        </div>

        <div className="dashboard-bottom-row">
          <HyperparamEvolution hyperparams={hyperparams} loading={loading} />
        </div>
      </div>
    </>
  );
}
