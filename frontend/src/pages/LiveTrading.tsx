import React, { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  ShoppingCart,
  Package,
  AlertCircle,
  Check,
  X,
  Loader2,
  BarChart3,
  Clock,
  Zap,
  Shield
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import { useWebSocket } from '../contexts/WebSocketContext';
import api from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  filledQuantity: number;
  avgFillPrice?: number;
  createdAt: string;
  updatedAt: string;
}

interface Position {
  symbol: string;
  quantity: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  pnlPercent: number;
}

const LiveTrading: React.FC = () => {
  const { marketData, isConnected } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState('100');
  const [limitPrice, setLimitPrice] = useState('');
  const [showOrderConfirm, setShowOrderConfirm] = useState(false);

  // Fetch active session
  const { data: tradingSession } = useQuery({
    queryKey: ['trading-session'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/trading/session');
        return response.data;
      } catch {
        return null;
      }
    },
    refetchInterval: 10000,
  });

  // Fetch positions
  const { data: positions = [] } = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/trading/positions');
        return response.data;
      } catch {
        return [];
      }
    },
    refetchInterval: 5000,
  });

  // Fetch open orders
  const { data: openOrders = [] } = useQuery({
    queryKey: ['open-orders'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/trading/orders/open');
        return response.data;
      } catch {
        return [];
      }
    },
    refetchInterval: 2000,
  });

  // Start trading session
  const startSession = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/trading/session/start', {
        broker: 'ibkr_cp',
        data_provider: 'iqfeed',
      });
      return response.data;
    },
  });

  // Place order mutation
  const placeOrder = useMutation({
    mutationFn: async (orderData: any) => {
      const response = await api.post('/api/trading/orders', orderData);
      return response.data;
    },
    onSuccess: () => {
      setShowOrderConfirm(false);
      setQuantity('100');
      setLimitPrice('');
    },
  });

  // Cancel order mutation
  const cancelOrder = useMutation({
    mutationFn: async (orderId: string) => {
      const response = await api.post(`/api/trading/orders/${orderId}/cancel`);
      return response.data;
    },
  });

  const handlePlaceOrder = () => {
    const orderData = {
      symbol: selectedSymbol,
      side: orderSide,
      order_type: orderType,
      quantity: parseInt(quantity),
      price: orderType === 'limit' ? parseFloat(limitPrice) : undefined,
    };
    
    placeOrder.mutate(orderData);
  };

  // Current market data for selected symbol
  const currentPrice = marketData[selectedSymbol]?.price || 0;
  const priceChange = marketData[selectedSymbol]?.change || 0;
  const priceChangePercent = marketData[selectedSymbol]?.changePercent || 0;

  // Mock chart data
  const priceHistory = Array.from({ length: 50 }, (_, i) => ({
    time: format(new Date(Date.now() - (49 - i) * 60000), 'HH:mm'),
    price: currentPrice + (Math.random() - 0.5) * 2,
  }));

  return (
    <div className="p-6 space-y-6">
      {/* Header with Session Status */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Trading</h1>
          <p className="text-gray-600">Execute trades in real-time</p>
        </div>
        <div className="flex items-center space-x-4">
          {tradingSession ? (
            <div className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg">
              <Activity className="w-4 h-4" />
              <span>Session Active</span>
            </div>
          ) : (
            <button
              onClick={() => startSession.mutate()}
              disabled={startSession.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center"
            >
              {startSession.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Start Trading Session
            </button>
          )}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trading Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Market Overview */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900">{selectedSymbol}</h3>
                <div className="flex items-center space-x-4 mt-2">
                  <span className="text-2xl font-bold">
                    ${currentPrice.toFixed(2)}
                  </span>
                  <div className={`flex items-center ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {priceChange >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    <span className="ml-1">${Math.abs(priceChange).toFixed(2)}</span>
                    <span className="ml-1">({priceChangePercent.toFixed(2)}%)</span>
                  </div>
                </div>
              </div>
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="AAPL">AAPL</option>
                <option value="MSFT">MSFT</option>
                <option value="GOOGL">GOOGL</option>
                <option value="AMZN">AMZN</option>
                <option value="TSLA">TSLA</option>
              </select>
            </div>

            {/* Price Chart */}
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={['dataMin - 1', 'dataMax + 1']} />
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
          </div>

          {/* Order Entry */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Place Order</h3>
            
            {/* Buy/Sell Toggle */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <button
                onClick={() => setOrderSide('buy')}
                className={`py-3 rounded-lg font-medium transition-colors ${
                  orderSide === 'buy'
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Buy
              </button>
              <button
                onClick={() => setOrderSide('sell')}
                className={`py-3 rounded-lg font-medium transition-colors ${
                  orderSide === 'sell'
                    ? 'bg-red-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Sell
              </button>
            </div>

            {/* Order Type */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Order Type
              </label>
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as 'market' | 'limit')}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="market">Market</option>
                <option value="limit">Limit</option>
              </select>
            </div>

            {/* Quantity */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Quantity
              </label>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Limit Price (if limit order) */}
            {orderType === 'limit' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Limit Price
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={limitPrice}
                  onChange={(e) => setLimitPrice(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}

            {/* Order Summary */}
            <div className="p-4 bg-gray-50 rounded-lg mb-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Estimated Total</span>
                <span className="font-medium">
                  ${((orderType === 'limit' ? parseFloat(limitPrice || '0') : currentPrice) * parseInt(quantity || '0')).toFixed(2)}
                </span>
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={() => setShowOrderConfirm(true)}
              disabled={!tradingSession || !quantity || (orderType === 'limit' && !limitPrice)}
              className={`w-full py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                orderSide === 'buy'
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              }`}
            >
              {orderSide === 'buy' ? 'Buy' : 'Sell'} {selectedSymbol}
            </button>
          </div>

          {/* Open Orders */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Open Orders</h3>
            </div>
            <div className="p-6">
              {openOrders.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No open orders</p>
              ) : (
                <div className="space-y-3">
                  {openOrders.map((order: Order) => (
                    <div key={order.id} className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <span className={`px-2 py-1 text-xs font-medium rounded ${
                            order.side === 'buy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {order.side.toUpperCase()}
                          </span>
                          <span className="font-medium">{order.symbol}</span>
                          <span className="text-gray-500">{order.quantity} shares</span>
                        </div>
                        <div className="mt-1 text-sm text-gray-500">
                          {order.orderType} order • ${order.price?.toFixed(2) || 'Market'}
                        </div>
                      </div>
                      <button
                        onClick={() => cancelOrder.mutate(order.id)}
                        className="text-red-600 hover:text-red-700"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Sidebar - Positions & Account */}
        <div className="space-y-6">
          {/* Account Summary */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Summary</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Buying Power</span>
                <span className="font-medium">$50,000.00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total Equity</span>
                <span className="font-medium">$125,430.50</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Today's P&L</span>
                <span className="font-medium text-green-600">+$2,340.75</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Open P&L</span>
                <span className="font-medium text-green-600">+$1,500.00</span>
              </div>
            </div>
          </div>

          {/* Current Positions */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Positions</h3>
            </div>
            <div className="p-6">
              {positions.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No open positions</p>
              ) : (
                <div className="space-y-3">
                  {positions.map((position: Position) => (
                    <div key={position.symbol} className="p-3 hover:bg-gray-50 rounded-lg">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">{position.symbol}</p>
                          <p className="text-sm text-gray-500">{position.quantity} shares</p>
                        </div>
                        <div className="text-right">
                          <p className="font-medium">${position.marketValue.toLocaleString()}</p>
                          <p className={`text-sm ${position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {position.unrealizedPnl >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Risk Controls */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Shield className="w-5 h-5 mr-2 text-blue-600" />
              Risk Controls
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Max Position Size</span>
                <span className="text-sm font-medium">$25,000</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Daily Loss Limit</span>
                <span className="text-sm font-medium">$5,000</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Circuit Breaker</span>
                <span className="text-sm font-medium text-green-600">Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Order Confirmation Modal */}
      <AnimatePresence>
        {showOrderConfirm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowOrderConfirm(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white rounded-lg p-6 max-w-md w-full m-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Confirm Order</h3>
              <div className="space-y-3 mb-6">
                <div className="flex justify-between">
                  <span className="text-gray-600">Symbol</span>
                  <span className="font-medium">{selectedSymbol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Side</span>
                  <span className={`font-medium ${orderSide === 'buy' ? 'text-green-600' : 'text-red-600'}`}>
                    {orderSide.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Quantity</span>
                  <span className="font-medium">{quantity} shares</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Order Type</span>
                  <span className="font-medium">{orderType.toUpperCase()}</span>
                </div>
                {orderType === 'limit' && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Limit Price</span>
                    <span className="font-medium">${parseFloat(limitPrice).toFixed(2)}</span>
                  </div>
                )}
                <div className="pt-3 border-t">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Estimated Total</span>
                    <span className="font-semibold text-lg">
                      ${((orderType === 'limit' ? parseFloat(limitPrice || '0') : currentPrice) * parseInt(quantity || '0')).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setShowOrderConfirm(false)}
                  className="flex-1 px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handlePlaceOrder}
                  disabled={placeOrder.isPending}
                  className={`flex-1 px-4 py-2 text-white rounded-lg disabled:opacity-50 flex items-center justify-center ${
                    orderSide === 'buy' ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'
                  }`}
                >
                  {placeOrder.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'Confirm Order'
                  )}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LiveTrading;