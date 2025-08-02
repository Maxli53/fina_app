import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  AlertCircle
} from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { format } from 'date-fns';
import { useWebSocket } from '../contexts/WebSocketContext';
import api from '../services/api';

interface PortfolioStats {
  totalValue: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  totalPnl: number;
  totalPnlPercent: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
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

interface RecentTrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  pnl?: number;
}

const Dashboard: React.FC = () => {
  const { marketData, portfolio } = useWebSocket();
  const [timeRange, setTimeRange] = useState('1D');

  // Fetch portfolio stats
  const { data: portfolioStats, isLoading: statsLoading } = useQuery({
    queryKey: ['portfolio-stats'],
    queryFn: async () => {
      // Mock data for now
      return {
        totalValue: 125430.50,
        dailyPnl: 2340.75,
        dailyPnlPercent: 1.89,
        totalPnl: 25430.50,
        totalPnlPercent: 25.43,
        winRate: 68.5,
        sharpeRatio: 1.85,
        maxDrawdown: -8.3
      } as PortfolioStats;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch positions
  const { data: positions, isLoading: positionsLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/trading/positions');
        return response.data;
      } catch (error) {
        // Return mock data if API fails
        return [
          {
            symbol: 'AAPL',
            quantity: 100,
            avgCost: 145.50,
            currentPrice: 150.25,
            marketValue: 15025,
            unrealizedPnl: 475,
            pnlPercent: 3.26
          },
          {
            symbol: 'MSFT',
            quantity: 50,
            avgCost: 320.00,
            currentPrice: 315.50,
            marketValue: 15775,
            unrealizedPnl: -225,
            pnlPercent: -1.41
          },
          {
            symbol: 'GOOGL',
            quantity: 25,
            avgCost: 2800.00,
            currentPrice: 2850.00,
            marketValue: 71250,
            unrealizedPnl: 1250,
            pnlPercent: 1.79
          }
        ] as Position[];
      }
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Mock performance chart data
  const performanceData = Array.from({ length: 30 }, (_, i) => ({
    date: format(new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000), 'MM/dd'),
    value: 100000 + Math.random() * 30000 - 10000 + i * 500,
  }));

  // Mock allocation data
  const allocationData = positions?.map(pos => ({
    name: pos.symbol,
    value: pos.marketValue,
  })) || [];

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  // Calculate some metrics
  const totalPositions = positions?.length || 0;
  const profitablePositions = positions?.filter(p => p.unrealizedPnl > 0).length || 0;
  const totalMarketValue = positions?.reduce((sum, p) => sum + p.marketValue, 0) || 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Portfolio Dashboard</h1>
          <p className="text-gray-600">Real-time portfolio analytics and performance</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="1D">1 Day</option>
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
            <option value="3M">3 Months</option>
            <option value="1Y">1 Year</option>
          </select>
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Clock className="w-4 h-4" />
            <span>Last updated: {format(new Date(), 'HH:mm:ss')}</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Value</p>
              <p className="text-2xl font-bold text-gray-900">
                ${portfolioStats?.totalValue.toLocaleString() || '0'}
              </p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <DollarSign className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Daily P&L</p>
              <p className={`text-2xl font-bold ${portfolioStats?.dailyPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${Math.abs(portfolioStats?.dailyPnl || 0).toLocaleString()}
              </p>
              <p className={`text-sm ${portfolioStats?.dailyPnl >= 0 ? 'text-green-600' : 'text-red-600'} flex items-center`}>
                {portfolioStats?.dailyPnl >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                {Math.abs(portfolioStats?.dailyPnlPercent || 0).toFixed(2)}%
              </p>
            </div>
            <div className={`p-3 rounded-lg ${portfolioStats?.dailyPnl >= 0 ? 'bg-green-100' : 'bg-red-100'}`}>
              {portfolioStats?.dailyPnl >= 0 ? 
                <TrendingUp className="w-6 h-6 text-green-600" /> : 
                <TrendingDown className="w-6 h-6 text-red-600" />
              }
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Win Rate</p>
              <p className="text-2xl font-bold text-gray-900">{portfolioStats?.winRate || 0}%</p>
              <p className="text-sm text-gray-500">
                {profitablePositions}/{totalPositions} positions
              </p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <BarChart3 className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-gray-900">{portfolioStats?.sharpeRatio || 0}</p>
              <p className="text-sm text-gray-500">Risk-adjusted return</p>
            </div>
            <div className="p-3 bg-amber-100 rounded-lg">
              <Activity className="w-6 h-6 text-amber-600" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 bg-white rounded-lg shadow p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
                itemStyle={{ color: '#F3F4F6' }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#3B82F6"
                fill="url(#colorValue)"
                strokeWidth={2}
              />
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Allocation Chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Allocation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={allocationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {allocationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Positions Table */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-lg shadow"
      >
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Open Positions</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Quantity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Cost
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Market Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  P&L %
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {positions?.map((position) => (
                <tr key={position.symbol} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {position.symbol}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {position.quantity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${position.avgCost.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${position.currentPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${position.marketValue.toLocaleString()}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                    position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${position.unrealizedPnl.toLocaleString()}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                    position.pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {position.pnlPercent > 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Risk Alerts */}
      {portfolioStats?.maxDrawdown < -5 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-amber-50 border border-amber-200 rounded-lg p-4"
        >
          <div className="flex">
            <AlertCircle className="h-5 w-5 text-amber-400 flex-shrink-0" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-amber-800">Risk Alert</h3>
              <p className="mt-1 text-sm text-amber-700">
                Maximum drawdown has reached {portfolioStats.maxDrawdown.toFixed(1)}%. Consider reviewing your risk management strategy.
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Dashboard;