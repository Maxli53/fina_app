import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  Target,
  Plus,
  Edit,
  Trash2,
  Play,
  Pause,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  CheckCircle,
  Code,
  Settings,
  BarChart3,
  Clock,
  DollarSign,
  Activity
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import api from '../services/api';

interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'mean_reversion' | 'momentum' | 'arbitrage' | 'ml_based' | 'custom';
  status: 'active' | 'paused' | 'stopped';
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    totalTrades: number;
  };
  parameters: Record<string, any>;
  symbols: string[];
  createdAt: string;
  lastModified: string;
  lastSignal?: {
    symbol: string;
    action: 'buy' | 'sell';
    timestamp: string;
    price: number;
  };
}

const Strategies: React.FC = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'builder' | 'library'>('overview');

  // Fetch strategies
  const { data: strategies = [], refetch } = useQuery({
    queryKey: ['strategies'],
    queryFn: async () => {
      // Mock data for now
      return [
        {
          id: '1',
          name: 'Mean Reversion SPY',
          description: 'Mean reversion strategy using Bollinger Bands on SPY',
          type: 'mean_reversion',
          status: 'active',
          performance: {
            totalReturn: 15.3,
            sharpeRatio: 1.82,
            maxDrawdown: -8.5,
            winRate: 68.5,
            avgWin: 1.2,
            avgLoss: -0.8,
            totalTrades: 145,
          },
          parameters: {
            lookbackPeriod: 20,
            stdDevMultiplier: 2,
            stopLoss: 0.02,
            takeProfit: 0.03,
          },
          symbols: ['SPY'],
          createdAt: '2024-01-15T10:00:00Z',
          lastModified: '2024-01-20T15:30:00Z',
          lastSignal: {
            symbol: 'SPY',
            action: 'buy',
            timestamp: '2024-01-25T14:30:00Z',
            price: 485.50,
          },
        },
        {
          id: '2',
          name: 'Momentum Tech Stocks',
          description: 'Momentum strategy for high-volume tech stocks',
          type: 'momentum',
          status: 'paused',
          performance: {
            totalReturn: 22.5,
            sharpeRatio: 2.15,
            maxDrawdown: -12.3,
            winRate: 62.3,
            avgWin: 2.1,
            avgLoss: -1.3,
            totalTrades: 89,
          },
          parameters: {
            momentumPeriod: 30,
            volumeThreshold: 1000000,
            rsiThreshold: 70,
            holdingPeriod: 5,
          },
          symbols: ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
          createdAt: '2024-01-10T09:00:00Z',
          lastModified: '2024-01-18T11:20:00Z',
        },
        {
          id: '3',
          name: 'ML Pattern Recognition',
          description: 'Machine learning based pattern recognition using Random Forest',
          type: 'ml_based',
          status: 'active',
          performance: {
            totalReturn: 18.7,
            sharpeRatio: 1.95,
            maxDrawdown: -9.8,
            winRate: 71.2,
            avgWin: 1.5,
            avgLoss: -0.9,
            totalTrades: 203,
          },
          parameters: {
            model: 'random_forest',
            features: ['rsi', 'macd', 'volume_ratio', 'price_change'],
            trainingWindow: 250,
            predictionHorizon: 5,
          },
          symbols: ['QQQ', 'IWM'],
          createdAt: '2023-12-20T08:00:00Z',
          lastModified: '2024-01-22T16:45:00Z',
        },
      ] as Strategy[];
    },
  });

  // Toggle strategy status
  const toggleStrategy = useMutation({
    mutationFn: async ({ id, status }: { id: string; status: string }) => {
      // API call would go here
      return { id, status };
    },
    onSuccess: () => {
      refetch();
    },
  });

  const strategyTypeConfig = {
    mean_reversion: { color: 'blue', icon: Activity },
    momentum: { color: 'green', icon: TrendingUp },
    arbitrage: { color: 'purple', icon: BarChart3 },
    ml_based: { color: 'amber', icon: Code },
    custom: { color: 'gray', icon: Settings },
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trading Strategies</h1>
          <p className="text-gray-600">Design, test, and deploy automated trading strategies</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Strategy
        </button>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-6 py-3 text-sm font-medium border-b-2 ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('builder')}
              className={`px-6 py-3 text-sm font-medium border-b-2 ${
                activeTab === 'builder'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Strategy Builder
            </button>
            <button
              onClick={() => setActiveTab('library')}
              className={`px-6 py-3 text-sm font-medium border-b-2 ${
                activeTab === 'library'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Template Library
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="p-6">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Active Strategies</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {strategies.filter(s => s.status === 'active').length}
                      </p>
                    </div>
                    <Activity className="w-8 h-8 text-blue-500" />
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Avg Return</p>
                      <p className="text-2xl font-bold text-green-600">
                        {(strategies.reduce((acc, s) => acc + s.performance.totalReturn, 0) / strategies.length).toFixed(1)}%
                      </p>
                    </div>
                    <TrendingUp className="w-8 h-8 text-green-500" />
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Total Trades</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {strategies.reduce((acc, s) => acc + s.performance.totalTrades, 0)}
                      </p>
                    </div>
                    <BarChart3 className="w-8 h-8 text-purple-500" />
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Avg Win Rate</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(strategies.reduce((acc, s) => acc + s.performance.winRate, 0) / strategies.length).toFixed(1)}%
                      </p>
                    </div>
                    <Target className="w-8 h-8 text-amber-500" />
                  </div>
                </div>
              </div>

              {/* Strategy List */}
              <div className="space-y-4">
                {strategies.map((strategy) => {
                  const config = strategyTypeConfig[strategy.type];
                  const Icon = config.icon;
                  
                  return (
                    <motion.div
                      key={strategy.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow"
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <div className={`p-2 bg-${config.color}-100 rounded-lg`}>
                              <Icon className={`w-5 h-5 text-${config.color}-600`} />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-900">{strategy.name}</h3>
                            <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                              strategy.status === 'active'
                                ? 'bg-green-100 text-green-700'
                                : strategy.status === 'paused'
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-gray-100 text-gray-700'
                            }`}>
                              {strategy.status}
                            </span>
                          </div>
                          <p className="text-gray-600 mb-4">{strategy.description}</p>
                          
                          {/* Performance Metrics */}
                          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                            <div>
                              <p className="text-xs text-gray-500">Return</p>
                              <p className={`text-sm font-medium ${
                                strategy.performance.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {strategy.performance.totalReturn >= 0 ? '+' : ''}{strategy.performance.totalReturn}%
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Sharpe</p>
                              <p className="text-sm font-medium">{strategy.performance.sharpeRatio}</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Max DD</p>
                              <p className="text-sm font-medium text-red-600">
                                {strategy.performance.maxDrawdown}%
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Win Rate</p>
                              <p className="text-sm font-medium">{strategy.performance.winRate}%</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Trades</p>
                              <p className="text-sm font-medium">{strategy.performance.totalTrades}</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Symbols</p>
                              <p className="text-sm font-medium">{strategy.symbols.join(', ')}</p>
                            </div>
                          </div>

                          {/* Last Signal */}
                          {strategy.lastSignal && (
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <Clock className="w-4 h-4" />
                              <span>Last signal:</span>
                              <span className={`font-medium ${
                                strategy.lastSignal.action === 'buy' ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {strategy.lastSignal.action.toUpperCase()} {strategy.lastSignal.symbol}
                              </span>
                              <span>@ ${strategy.lastSignal.price}</span>
                              <span className="text-gray-400">
                                ({format(new Date(strategy.lastSignal.timestamp), 'MMM dd, HH:mm')})
                              </span>
                            </div>
                          )}
                        </div>

                        {/* Actions */}
                        <div className="flex items-center space-x-2 ml-4">
                          <button
                            onClick={() => toggleStrategy.mutate({
                              id: strategy.id,
                              status: strategy.status === 'active' ? 'paused' : 'active'
                            })}
                            className={`p-2 rounded-lg transition-colors ${
                              strategy.status === 'active'
                                ? 'bg-yellow-100 text-yellow-600 hover:bg-yellow-200'
                                : 'bg-green-100 text-green-600 hover:bg-green-200'
                            }`}
                          >
                            {strategy.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                          </button>
                          <button
                            onClick={() => setSelectedStrategy(strategy)}
                            className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors"
                          >
                            <Edit className="w-4 h-4" />
                          </button>
                          <button className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors">
                            <BarChart3 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'builder' && (
            <div className="text-center py-12">
              <Target className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Strategy Builder</h3>
              <p className="text-gray-600 mb-4">Visual strategy builder coming soon</p>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                Use Code Editor
              </button>
            </div>
          )}

          {activeTab === 'library' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Template Cards */}
              {[
                {
                  name: 'Moving Average Crossover',
                  description: 'Classic trend-following strategy using MA crossovers',
                  type: 'momentum',
                  difficulty: 'Beginner',
                },
                {
                  name: 'RSI Mean Reversion',
                  description: 'Trade oversold/overbought conditions using RSI',
                  type: 'mean_reversion',
                  difficulty: 'Beginner',
                },
                {
                  name: 'Pairs Trading',
                  description: 'Statistical arbitrage between correlated pairs',
                  type: 'arbitrage',
                  difficulty: 'Advanced',
                },
                {
                  name: 'LSTM Price Prediction',
                  description: 'Deep learning model for price forecasting',
                  type: 'ml_based',
                  difficulty: 'Expert',
                },
                {
                  name: 'Options Gamma Scalping',
                  description: 'Dynamic hedging strategy for options',
                  type: 'custom',
                  difficulty: 'Expert',
                },
                {
                  name: 'Volume Profile Trading',
                  description: 'Trade based on volume distribution levels',
                  type: 'custom',
                  difficulty: 'Intermediate',
                },
              ].map((template, idx) => (
                <div key={idx} className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow cursor-pointer">
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">{template.name}</h3>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      template.difficulty === 'Beginner'
                        ? 'bg-green-100 text-green-700'
                        : template.difficulty === 'Intermediate'
                        ? 'bg-yellow-100 text-yellow-700'
                        : template.difficulty === 'Advanced'
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-red-100 text-red-700'
                    }`}>
                      {template.difficulty}
                    </span>
                  </div>
                  <p className="text-gray-600 mb-4">{template.description}</p>
                  <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Use Template
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Strategies;