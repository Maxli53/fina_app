import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketOptions {
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}

interface WebSocketState {
  socket: WebSocket | null;
  isConnected: boolean;
  lastMessage: any;
  error: Error | null;
}

export const useWebSocket = (
  url: string,
  options: WebSocketOptions = {}
) => {
  const {
    reconnect = true,
    reconnectInterval = 3000,
    reconnectAttempts = 5
  } = options;

  const [state, setState] = useState<WebSocketState>({
    socket: null,
    isConnected: false,
    lastMessage: null,
    error: null
  });

  const reconnectCount = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setState(prev => ({
          ...prev,
          socket: ws,
          isConnected: true,
          error: null
        }));
        reconnectCount.current = 0;
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setState(prev => ({
          ...prev,
          isConnected: false
        }));

        if (reconnect && reconnectCount.current < reconnectAttempts) {
          reconnectTimeout.current = setTimeout(() => {
            console.log(`Reconnecting... Attempt ${reconnectCount.current + 1}`);
            reconnectCount.current++;
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setState(prev => ({
          ...prev,
          error: new Error('WebSocket connection error')
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setState(prev => ({
            ...prev,
            lastMessage: data
          }));
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      setState(prev => ({
        ...prev,
        socket: ws
      }));

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setState(prev => ({
        ...prev,
        error: error as Error
      }));
    }
  }, [url, reconnect, reconnectInterval, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
    }
    
    if (state.socket) {
      state.socket.close();
    }
  }, [state.socket]);

  const sendMessage = useCallback((message: any) => {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      const data = typeof message === 'string' ? message : JSON.stringify(message);
      state.socket.send(data);
    } else {
      console.error('WebSocket is not connected');
    }
  }, [state.socket]);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    socket: state.socket,
    isConnected: state.isConnected,
    lastMessage: state.lastMessage,
    error: state.error,
    sendMessage,
    disconnect,
    reconnect: connect
  };
};