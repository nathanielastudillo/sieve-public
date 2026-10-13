import { useEffect, useRef, useState, useCallback } from 'react';

const INITIAL_RETRY_MS = 1000;
const MAX_RETRY_MS = 30000;
const BACKOFF_FACTOR = 1.5;

export function useWebSocket() {
  const [lastMessage, setLastMessage] = useState(null);
  const [connectionState, setConnectionState] = useState('disconnected'); // 'connected' | 'connecting' | 'disconnected'
  const wsRef = useRef(null);
  const retryTimeoutRef = useRef(null);
  const retryDelayRef = useRef(INITIAL_RETRY_MS);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    setConnectionState('connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionState('connected');
        retryDelayRef.current = INITIAL_RETRY_MS;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch {
          // ignore non-JSON messages
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionState('disconnected');
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        // onclose will fire after onerror, which handles reconnection
        ws.close();
      };
    } catch {
      setConnectionState('disconnected');
      scheduleReconnect();
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;
    if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);

    const delay = retryDelayRef.current;
    retryDelayRef.current = Math.min(delay * BACKOFF_FACTOR, MAX_RETRY_MS);

    retryTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, delay);
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return {
    lastMessage,
    isConnected: connectionState === 'connected',
    connectionState,
  };
}
