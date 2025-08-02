import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { useAuth } from './AuthContext';
import toast from 'react-hot-toast';

interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  timestamp: string;
}

interface PortfolioUpdate {
  positions: any[];
  totalValue: number;
  dailyPnl: number;
  timestamp: string;
}

interface OrderUpdate {
  orderId: string;
  status: string;
  filledQuantity?: number;
  avgFillPrice?: number;
  timestamp: string;
}

interface WebSocketContextType {
  isConnected: boolean;
  marketData: Record<string, MarketData>;
  portfolio: PortfolioUpdate | null;
  orderUpdates: OrderUpdate[];
  subscribeToSymbol: (symbol: string) => void;
  unsubscribeFromSymbol: (symbol: string) => void;
  subscribedSymbols: Set<string>;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const [isConnected, setIsConnected] = useState(false);
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [portfolio, setPortfolio] = useState<PortfolioUpdate | null>(null);
  const [orderUpdates, setOrderUpdates] = useState<OrderUpdate[]>([]);
  const [subscribedSymbols, setSubscribedSymbols] = useState<Set<string>>(new Set());
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    if (!isAuthenticated || ws.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const token = localStorage.getItem('access_token');
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/api/trading/ws/market-data';
      
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        reconnectAttempts.current = 0;
        
        // Send authentication
        if (token) {
          ws.current?.send(JSON.stringify({
            type: 'auth',
            token: token
          }));
        }

        // Resubscribe to symbols
        subscribedSymbols.forEach(symbol => {
          ws.current?.send(JSON.stringify({
            action: 'subscribe',
            symbols: [symbol]
          }));
        });
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'market_data':
            case 'quote':
              setMarketData(prev => ({
                ...prev,
                [data.symbol]: {
                  symbol: data.symbol,
                  bid: data.data.bid,
                  ask: data.data.ask,
                  last: data.data.last,
                  volume: data.data.volume || 0,
                  timestamp: data.data.timestamp
                }
              }));
              break;
              
            case 'portfolio':
              setPortfolio(data.data);
              break;
              
            case 'order':
              setOrderUpdates(prev => [...prev, data.data].slice(-50)); // Keep last 50 updates
              break;
              
            case 'error':
              console.error('WebSocket error:', data.message);
              break;
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect
        if (isAuthenticated && reconnectAttempts.current < 5) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeout.current = setTimeout(connect, delay);
        }
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setIsConnected(false);
    }
  }, [isAuthenticated, subscribedSymbols]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const subscribeToSymbol = useCallback((symbol: string) => {
    setSubscribedSymbols(prev => new Set(prev).add(symbol));
    
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'subscribe',
        symbols: [symbol]
      }));
    }
  }, []);

  const unsubscribeFromSymbol = useCallback((symbol: string) => {
    setSubscribedSymbols(prev => {
      const newSet = new Set(prev);
      newSet.delete(symbol);
      return newSet;
    });
    
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'unsubscribe',
        symbols: [symbol]
      }));
    }
    
    // Remove market data for unsubscribed symbol
    setMarketData(prev => {
      const newData = { ...prev };
      delete newData[symbol];
      return newData;
    });
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [isAuthenticated, connect, disconnect]);

  const value = {
    isConnected,
    marketData,
    portfolio,
    orderUpdates,
    subscribeToSymbol,
    unsubscribeFromSymbol,
    subscribedSymbols,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};