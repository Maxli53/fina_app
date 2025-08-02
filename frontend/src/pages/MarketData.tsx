import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Search,
  Filter,
  Download,
  RefreshCw,
  Activity,
  AlertCircle,
  ChevronUp,
  ChevronDown,
  Circle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import { useWebSocket } from '../contexts/WebSocketContext';
import { dataService } from '../services/api';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MarketQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  marketCap?: number;
  bid?: number;
  ask?: number;
  bidSize?: number;
  askSize?: number;
  lastUpdate: string;
}

interface WatchlistItem {
  symbol: string;
  name: string;
  sector?: string;
}

const MarketData: React.FC = () => {
  const { marketData, isConnected, subscribeToSymbol, unsubscribeFromSymbol } = useWebSocket();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([
    { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', sector: 'Technology' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer Cyclical' },
    { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Consumer Cyclical' },
    { symbol: 'JPM', name: 'JPMorgan Chase', sector: 'Financial' },
    { symbol: 'BAC', name: 'Bank of America', sector: 'Financial' },
    { symbol: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare' },
  ]);

  // Subscribe to watchlist symbols
  useEffect(() => {
    watchlist.forEach(item => {
      subscribeToSymbol(item.symbol);
    });

    return () => {
      watchlist.forEach(item => {
        unsubscribeFromSymbol(item.symbol);
      });
    };
  }, [watchlist, subscribeToSymbol, unsubscribeFromSymbol]);

  // Search symbols
  const { data: searchResults } = useQuery({
    queryKey: ['symbol-search', searchQuery],
    queryFn: () => dataService.searchSymbols(searchQuery),
    enabled: searchQuery.length > 0,
  });

  // Get market status
  const { data: marketStatus } = useQuery({
    queryKey: ['market-status'],
    queryFn: () => dataService.getMarketStatus(),
    refetchInterval: 60000, // Update every minute
  });

  // Get detailed data for selected symbol
  const { data: symbolDetails } = useQuery({
    queryKey: ['symbol-details', selectedSymbol],
    queryFn: async () => {
      if (!selectedSymbol) return null;
      const endDate = format(new Date(), 'yyyy-MM-dd');
      const startDate = format(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), 'yyyy-MM-dd');
      return dataService.getHistoricalData(selectedSymbol, startDate, endDate, '1d');
    },
    enabled: !!selectedSymbol,
  });

  const addToWatchlist = (symbol: string, name: string) => {
    if (!watchlist.find(item => item.symbol === symbol)) {
      const newItem = { symbol, name };
      setWatchlist([...watchlist, newItem]);
      subscribeToSymbol(symbol);
    }
  };

  const removeFromWatchlist = (symbol: string) => {
    setWatchlist(watchlist.filter(item => item.symbol !== symbol));
    unsubscribeFromSymbol(symbol);
  };

  // Mock real-time quotes with WebSocket data
  const quotes: MarketQuote[] = watchlist.map(item => {
    const wsData = marketData[item.symbol];
    return {
      symbol: item.symbol,
      name: item.name,
      price: wsData?.price || Math.random() * 200 + 50,
      change: wsData?.change || (Math.random() - 0.5) * 10,
      changePercent: wsData?.changePercent || (Math.random() - 0.5) * 5,
      volume: wsData?.volume || Math.floor(Math.random() * 10000000),
      high: wsData?.high || (wsData?.price || 100) * 1.02,
      low: wsData?.low || (wsData?.price || 100) * 0.98,
      open: wsData?.open || (wsData?.price || 100) * 0.99,
      previousClose: wsData?.previousClose || (wsData?.price || 100) * 0.995,
      bid: wsData?.bid || (wsData?.price || 100) * 0.999,
      ask: wsData?.ask || (wsData?.price || 100) * 1.001,
      bidSize: wsData?.bidSize || Math.floor(Math.random() * 1000),
      askSize: wsData?.askSize || Math.floor(Math.random() * 1000),
      lastUpdate: format(new Date(), 'HH:mm:ss'),
    };
  });

  // Mock intraday chart data
  const intradayData = Array.from({ length: 390 }, (_, i) => {
    const basePrice = 150;
    const time = new Date();
    time.setHours(9, 30, 0, 0);
    time.setMinutes(time.getMinutes() + i);
    
    return {
      time: format(time, 'HH:mm'),
      price: basePrice + (Math.random() - 0.5) * 10 + Math.sin(i / 50) * 5,
      volume: Math.floor(Math.random() * 100000),
    };
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Market Data</h1>
          <p className="text-gray-600">Real-time quotes and market analysis</p>
        </div>
        <div className="flex items-center space-x-4">
          {/* Market Status */}
          <div className="flex items-center space-x-2">
            <Circle className={`w-3 h-3 ${marketStatus?.is_open ? 'text-green-500' : 'text-red-500'} fill-current`} />
            <span className="text-sm text-gray-600">
              Market {marketStatus?.is_open ? 'Open' : 'Closed'}
            </span>
          </div>
          
          {/* Connection Status */}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            <Activity className="w-4 h-4" />
            <span>{isConnected ? 'Live Data' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Watchlist */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">Watchlist</h3>
              <div className="flex items-center space-x-2">
                <button className="p-2 hover:bg-gray-100 rounded-lg">
                  <Filter className="w-4 h-4 text-gray-600" />
                </button>
                <button className="p-2 hover:bg-gray-100 rounded-lg">
                  <Download className="w-4 h-4 text-gray-600" />
                </button>
              </div>
            </div>
            
            {/* Search Bar */}
            <div className="px-6 py-3 border-b border-gray-200">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search symbols..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              {/* Search Results Dropdown */}
              {searchResults && searchResults.length > 0 && (
                <div className="absolute z-10 mt-2 w-full bg-white border border-gray-200 rounded-lg shadow-lg">
                  {searchResults.map((result) => (
                    <button
                      key={result.symbol}
                      onClick={() => {
                        addToWatchlist(result.symbol, result.name);
                        setSearchQuery('');
                      }}
                      className="w-full px-4 py-2 text-left hover:bg-gray-50 flex justify-between items-center"
                    >
                      <div>
                        <span className="font-medium">{result.symbol}</span>
                        <span className="text-gray-500 ml-2 text-sm">{result.name}</span>
                      </div>
                      <span className="text-xs text-gray-400">{result.exchange}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Quotes Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Change
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Volume
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Bid/Ask
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Day Range
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {quotes.map((quote) => (
                    <motion.tr
                      key={quote.symbol}
                      className="hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedSymbol(quote.symbol)}
                      animate={{ backgroundColor: marketData[quote.symbol]?.updated ? '#FEF3C7' : '#FFFFFF' }}
                      transition={{ duration: 0.5 }}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">{quote.symbol}</div>
                          <div className="text-xs text-gray-500">{quote.name}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          ${quote.price.toFixed(2)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className={`flex items-center ${quote.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {quote.change >= 0 ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          <span className="text-sm font-medium">
                            ${Math.abs(quote.change).toFixed(2)} ({quote.changePercent.toFixed(2)}%)
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {(quote.volume / 1000000).toFixed(2)}M
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm">
                          <span className="text-gray-900">${quote.bid?.toFixed(2)}</span>
                          <span className="text-gray-500 mx-1">/</span>
                          <span className="text-gray-900">${quote.ask?.toFixed(2)}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          {quote.bidSize} x {quote.askSize}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          ${quote.low.toFixed(2)} - ${quote.high.toFixed(2)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromWatchlist(quote.symbol);
                          }}
                          className="text-red-600 hover:text-red-900"
                        >
                          Remove
                        </button>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Symbol Details */}
        <div className="space-y-6">
          {selectedSymbol ? (
            <>
              {/* Quote Details */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white rounded-lg shadow p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-4">{selectedSymbol} Details</h3>
                {(() => {
                  const quote = quotes.find(q => q.symbol === selectedSymbol);
                  if (!quote) return null;
                  
                  return (
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Open</span>
                        <span className="font-medium">${quote.open.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Previous Close</span>
                        <span className="font-medium">${quote.previousClose.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Day High</span>
                        <span className="font-medium">${quote.high.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Day Low</span>
                        <span className="font-medium">${quote.low.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Volume</span>
                        <span className="font-medium">{quote.volume.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Last Update</span>
                        <span className="font-medium">{quote.lastUpdate}</span>
                      </div>
                    </div>
                  );
                })()}
              </motion.div>

              {/* Intraday Chart */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-white rounded-lg shadow p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Intraday Chart</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={intradayData.filter((_, i) => i % 5 === 0)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#9CA3AF"
                        tick={{ fontSize: 12 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis 
                        stroke="#9CA3AF" 
                        tick={{ fontSize: 12 }}
                        domain={['dataMin - 1', 'dataMax + 1']}
                      />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#3B82F6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Volume Chart */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white rounded-lg shadow p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Volume</h3>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={intradayData.filter((_, i) => i % 30 === 0)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#9CA3AF"
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis 
                        stroke="#9CA3AF" 
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip />
                      <Bar dataKey="volume" fill="#10B981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            </>
          ) : (
            <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
              <BarChart3 className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p>Select a symbol to view details</p>
            </div>
          )}
        </div>
      </div>

      {/* Market Overview */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">S&P 500</p>
            <p className="text-xl font-bold">4,512.23</p>
            <p className="text-sm text-green-600">+0.82%</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">Dow Jones</p>
            <p className="text-xl font-bold">35,123.45</p>
            <p className="text-sm text-green-600">+0.65%</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">Nasdaq</p>
            <p className="text-xl font-bold">14,234.56</p>
            <p className="text-sm text-red-600">-0.23%</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">VIX</p>
            <p className="text-xl font-bold">16.45</p>
            <p className="text-sm text-red-600">+2.3%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketData;