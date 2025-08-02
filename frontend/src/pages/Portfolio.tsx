import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Briefcase,
  TrendingUp,
  TrendingDown,
  DollarSign,
  PieChart,
  Calendar,
  Download,
  Filter,
  Eye,
  EyeOff,
  AlertCircle,
  Info,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';
import { motion } from 'framer-motion';
import { format, subDays } from 'date-fns';
import api from '../services/api';
import { LineChart, Line, AreaChart, Area, PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

interface PortfolioMetrics {
  totalValue: number;
  cashBalance: number;
  investedAmount: number;
  totalPnl: number;
  totalPnlPercent: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  unrealizedPnl: number;
  realizedPnl: number;
  dividends: number;
  fees: number;
}

interface Position {
  symbol: string;
  name: string;
  quantity: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  pnlPercent: number;
  allocation: number;
  sector: string;
}

interface Transaction {
  id: string;
  date: string;
  symbol: string;
  type: 'buy' | 'sell' | 'dividend' | 'fee';
  quantity: number;
  price: number;
  total: number;
  pnl?: number;
}

const Portfolio: React.FC = () => {
  const [timeRange, setTimeRange] = useState('1M');
  const [showValues, setShowValues] = useState(true);
  const [activeView, setActiveView] = useState<'overview' | 'positions' | 'transactions' | 'analysis'>('overview');

  // Fetch portfolio data
  const { data: portfolioData } = useQuery({
    queryKey: ['portfolio-data'],
    queryFn: async () => {
      // Mock data
      return {
        metrics: {
          totalValue: 156789.50,
          cashBalance: 31358.00,
          investedAmount: 125431.50,
          totalPnl: 31358.00,
          totalPnlPercent: 25.0,
          dailyPnl: 1245.75,
          dailyPnlPercent: 0.8,
          unrealizedPnl: 8567.50,
          realizedPnl: 22790.50,
          dividends: 3456.00,
          fees: -1234.00,
        } as PortfolioMetrics,
        positions: [
          {
            symbol: 'AAPL',
            name: 'Apple Inc.',
            quantity: 150,
            avgCost: 145.50,
            currentPrice: 155.25,
            marketValue: 23287.50,
            unrealizedPnl: 1462.50,
            pnlPercent: 6.7,
            allocation: 14.8,
            sector: 'Technology',
          },
          {
            symbol: 'MSFT',
            name: 'Microsoft Corp.',
            quantity: 75,
            avgCost: 325.00,
            currentPrice: 340.50,
            marketValue: 25537.50,
            unrealizedPnl: 1162.50,
            pnlPercent: 4.8,
            allocation: 16.3,
            sector: 'Technology',
          },
          {
            symbol: 'JPM',
            name: 'JPMorgan Chase',
            quantity: 100,
            avgCost: 140.00,
            currentPrice: 152.75,
            marketValue: 15275.00,
            unrealizedPnl: 1275.00,
            pnlPercent: 9.1,
            allocation: 9.7,
            sector: 'Financial',
          },
          {
            symbol: 'AMZN',
            name: 'Amazon.com Inc.',
            quantity: 20,
            avgCost: 3200.00,
            currentPrice: 3350.00,
            marketValue: 67000.00,
            unrealizedPnl: 3000.00,
            pnlPercent: 4.7,
            allocation: 42.7,
            sector: 'Consumer Cyclical',
          },
          {
            symbol: 'JNJ',
            name: 'Johnson & Johnson',
            quantity: 50,
            avgCost: 165.00,
            currentPrice: 158.50,
            marketValue: 7925.00,
            unrealizedPnl: -325.00,
            pnlPercent: -3.9,
            allocation: 5.1,
            sector: 'Healthcare',
          },
        ] as Position[],
        transactions: [
          {
            id: '1',
            date: '2024-01-25T10:30:00Z',
            symbol: 'AAPL',
            type: 'buy',
            quantity: 50,
            price: 155.00,
            total: 7750.00,
          },
          {
            id: '2',
            date: '2024-01-24T14:15:00Z',
            symbol: 'MSFT',
            type: 'sell',
            quantity: 25,
            price: 340.00,
            total: 8500.00,
            pnl: 375.00,
          },
          {
            id: '3',
            date: '2024-01-23T09:45:00Z',
            symbol: 'AAPL',
            type: 'dividend',
            quantity: 150,
            price: 0.96,
            total: 144.00,
          },
        ] as Transaction[],
      };
    },
  });

  // Generate performance chart data
  const performanceData = Array.from({ length: 30 }, (_, i) => {
    const date = subDays(new Date(), 29 - i);
    const baseValue = 150000;
    const randomChange = (Math.random() - 0.5) * 5000;
    const trendValue = i * 200;
    
    return {
      date: format(date, 'MMM dd'),
      value: baseValue + randomChange + trendValue,
      benchmark: baseValue + i * 150,
    };
  });

  // Sector allocation data
  const sectorData = portfolioData?.positions.reduce((acc: any[], position) => {
    const existing = acc.find(item => item.name === position.sector);
    if (existing) {
      existing.value += position.marketValue;
    } else {
      acc.push({ name: position.sector, value: position.marketValue });
    }
    return acc;
  }, []) || [];

  // Risk metrics for radar chart
  const riskMetrics = [
    { metric: 'Volatility', value: 65 },
    { metric: 'Sharpe Ratio', value: 85 },
    { metric: 'Max Drawdown', value: 75 },
    { metric: 'Beta', value: 70 },
    { metric: 'Alpha', value: 80 },
    { metric: 'Concentration', value: 45 },
  ];

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Portfolio</h1>
          <p className="text-gray-600">Comprehensive portfolio analysis and management</p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowValues(!showValues)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            {showValues ? <Eye className="w-5 h-5 text-gray-600" /> : <EyeOff className="w-5 h-5 text-gray-600" />}
          </button>
          <button className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center">
            <Download className="w-4 h-4 mr-2" />
            Export
          </button>
        </div>
      </div>

      {/* Portfolio Summary */}
      {portfolioData && (
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
                  {showValues ? `$${portfolioData.metrics.totalValue.toLocaleString()}` : '••••••'}
                </p>
                <p className={`text-sm mt-1 flex items-center ${
                  portfolioData.metrics.dailyPnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {portfolioData.metrics.dailyPnl >= 0 ? (
                    <ArrowUpRight className="w-4 h-4 mr-1" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4 mr-1" />
                  )}
                  {portfolioData.metrics.dailyPnlPercent >= 0 ? '+' : ''}{portfolioData.metrics.dailyPnlPercent}% Today
                </p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <Briefcase className="w-6 h-6 text-blue-600" />
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
                <p className="text-sm text-gray-600">Total P&L</p>
                <p className={`text-2xl font-bold ${
                  portfolioData.metrics.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {showValues ? `$${Math.abs(portfolioData.metrics.totalPnl).toLocaleString()}` : '••••••'}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {portfolioData.metrics.totalPnlPercent >= 0 ? '+' : ''}{portfolioData.metrics.totalPnlPercent}% All Time
                </p>
              </div>
              <div className={`p-3 rounded-lg ${
                portfolioData.metrics.totalPnl >= 0 ? 'bg-green-100' : 'bg-red-100'
              }`}>
                {portfolioData.metrics.totalPnl >= 0 ? (
                  <TrendingUp className="w-6 h-6 text-green-600" />
                ) : (
                  <TrendingDown className="w-6 h-6 text-red-600" />
                )}
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
                <p className="text-sm text-gray-600">Cash Balance</p>
                <p className="text-2xl font-bold text-gray-900">
                  {showValues ? `$${portfolioData.metrics.cashBalance.toLocaleString()}` : '••••••'}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {((portfolioData.metrics.cashBalance / portfolioData.metrics.totalValue) * 100).toFixed(1)}% of portfolio
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <DollarSign className="w-6 h-6 text-green-600" />
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
                <p className="text-sm text-gray-600">Positions</p>
                <p className="text-2xl font-bold text-gray-900">
                  {portfolioData.positions.length}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {portfolioData.positions.filter(p => p.unrealizedPnl > 0).length} profitable
                </p>
              </div>
              <div className="p-3 bg-purple-100 rounded-lg">
                <PieChart className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            {['overview', 'positions', 'transactions', 'analysis'].map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view as any)}
                className={`px-6 py-3 text-sm font-medium border-b-2 capitalize ${
                  activeView === view
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {view}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Overview Tab */}
          {activeView === 'overview' && (
            <div className="space-y-6">
              {/* Performance Chart */}
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">Performance</h3>
                  <select
                    value={timeRange}
                    onChange={(e) => setTimeRange(e.target.value)}
                    className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
                  >
                    <option value="1W">1 Week</option>
                    <option value="1M">1 Month</option>
                    <option value="3M">3 Months</option>
                    <option value="1Y">1 Year</option>
                    <option value="ALL">All Time</option>
                  </select>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="date" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                        labelStyle={{ color: '#F3F4F6' }}
                      />
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="#3B82F6"
                        fill="url(#colorValue)"
                        strokeWidth={2}
                        name="Portfolio"
                      />
                      <Area
                        type="monotone"
                        dataKey="benchmark"
                        stroke="#10B981"
                        fill="url(#colorBenchmark)"
                        strokeWidth={2}
                        name="S&P 500"
                      />
                      <defs>
                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Allocation Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Sector Allocation */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Sector Allocation</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsPieChart>
                        <Pie
                          data={sectorData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {sectorData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </RechartsPieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Risk Profile */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Profile</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={riskMetrics}>
                        <PolarGrid stroke="#e5e7eb" />
                        <PolarAngleAxis dataKey="metric" stroke="#9CA3AF" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9CA3AF" />
                        <Radar
                          name="Portfolio"
                          dataKey="value"
                          stroke="#3B82F6"
                          fill="#3B82F6"
                          fillOpacity={0.6}
                        />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Positions Tab */}
          {activeView === 'positions' && portfolioData && (
            <div>
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
                        Allocation
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {portfolioData.positions.map((position) => (
                      <tr key={position.symbol} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div>
                            <div className="text-sm font-medium text-gray-900">{position.symbol}</div>
                            <div className="text-xs text-gray-500">{position.name}</div>
                          </div>
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
                          {showValues ? `$${position.marketValue.toLocaleString()}` : '••••••'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`text-sm font-medium ${
                            position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {showValues ? `$${position.unrealizedPnl.toLocaleString()}` : '••••••'}
                          </div>
                          <div className={`text-xs ${
                            position.pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{ width: `${position.allocation}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-900">{position.allocation.toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Transactions Tab */}
          {activeView === 'transactions' && portfolioData && (
            <div>
              <div className="mb-4 flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200">
                    All
                  </button>
                  <button className="px-3 py-1 text-gray-600 hover:bg-gray-100 rounded-lg">
                    Buys
                  </button>
                  <button className="px-3 py-1 text-gray-600 hover:bg-gray-100 rounded-lg">
                    Sells
                  </button>
                  <button className="px-3 py-1 text-gray-600 hover:bg-gray-100 rounded-lg">
                    Dividends
                  </button>
                </div>
                <button className="flex items-center text-gray-600 hover:text-gray-900">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </button>
              </div>
              
              <div className="space-y-2">
                {portfolioData.transactions.map((transaction) => (
                  <div key={transaction.id} className="flex justify-between items-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100">
                    <div className="flex items-center space-x-4">
                      <div className={`p-2 rounded-lg ${
                        transaction.type === 'buy' ? 'bg-green-100' :
                        transaction.type === 'sell' ? 'bg-red-100' :
                        transaction.type === 'dividend' ? 'bg-blue-100' :
                        'bg-gray-100'
                      }`}>
                        {transaction.type === 'buy' ? <TrendingUp className="w-5 h-5 text-green-600" /> :
                         transaction.type === 'sell' ? <TrendingDown className="w-5 h-5 text-red-600" /> :
                         transaction.type === 'dividend' ? <DollarSign className="w-5 h-5 text-blue-600" /> :
                         <AlertCircle className="w-5 h-5 text-gray-600" />}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">
                          {transaction.type.charAt(0).toUpperCase() + transaction.type.slice(1)} {transaction.symbol}
                        </p>
                        <p className="text-sm text-gray-500">
                          {format(new Date(transaction.date), 'MMM dd, yyyy HH:mm')}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">
                        ${transaction.total.toLocaleString()}
                      </p>
                      {transaction.type !== 'dividend' && (
                        <p className="text-sm text-gray-500">
                          {transaction.quantity} @ ${transaction.price.toFixed(2)}
                        </p>
                      )}
                      {transaction.pnl && (
                        <p className={`text-sm font-medium ${transaction.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          P&L: ${transaction.pnl.toFixed(2)}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analysis Tab */}
          {activeView === 'analysis' && (
            <div className="text-center py-12">
              <Info className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Advanced Analytics</h3>
              <p className="text-gray-600">Detailed portfolio analytics coming soon</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Portfolio;