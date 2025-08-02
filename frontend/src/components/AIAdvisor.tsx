import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, 
  MessageSquare, 
  TrendingUp, 
  Shield, 
  BarChart3,
  FileText,
  Send,
  Loader2,
  ChevronDown,
  AlertTriangle,
  CheckCircle,
  Info,
  Zap,
  Network,
  GitBranch,
  Layers,
  Cpu,
  HelpCircle,
  ChevronRight,
  Download
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  context?: any;
}

interface Recommendation {
  text: string;
  priority: 'high' | 'medium' | 'low';
  category: string;
}

interface AdvisoryResponse {
  recommendations: Recommendation[];
  insights: string[];
  warnings: string[];
  confidence_level: number;
  optimal_configuration?: any;
}

// Context type for AI advisor
interface AnalysisContext {
  activeAnalysis: 'idtxl' | 'ml' | 'neural' | 'integrated';
  symbols: string[];
  dateRange: { start: string; end: string };
  // IDTxl context
  idtxl?: {
    analysisType: string;
    estimator: string;
    maxLag: number;
    significanceLevel: number;
    permutations: number;
    kNeighbors?: number;
    noiseLevel?: number;
    alphabetSize?: number;
  };
  // ML context
  ml?: {
    modelType: string;
    target: string;
    predictionHorizon: number;
    validation: string;
    testSize: number;
    hyperparameters: any;
    features: any;
  };
  // Neural Network context
  neural?: {
    architecture: string;
    epochs: number;
    batchSize: number;
    optimizer: string;
    learningRate: number;
    dropoutRate: number;
    layers?: number[];
    bidirectional?: boolean;
    attentionHeads?: number;
    encoderLayers?: number;
  };
  // Integrated context
  integrated?: {
    components: {
      idtxl: { enabled: boolean; weight: number };
      ml: { enabled: boolean; weight: number };
      nn: { enabled: boolean; weight: number };
    };
    ensembleMethod: string;
    signalThreshold: number;
  };
  // Performance metrics
  historicalPerformance?: {
    sharpeRatio?: number;
    maxDrawdown?: number;
    winRate?: number;
  };
}

// Analysis results interface
interface AnalysisResults {
  type: 'idtxl' | 'ml' | 'neural' | 'integrated';
  taskId: string;
  timestamp: string;
  // IDTxl results
  idtxlResults?: {
    transferEntropy?: Record<string, any>;
    mutualInformation?: Record<string, any>;
    causalNetwork?: Record<string, any>;
    significantConnections: Array<{
      source: string;
      target: string;
      strength: number;
      pValue: number;
    }>;
    processingTime: number;
  };
  // ML results
  mlResults?: {
    modelType: string;
    target: string;
    finalMetrics: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1Score?: number;
      rmse?: number;
      mae?: number;
      sharpeRatio?: number;
    };
    featureImportance: Record<string, number>;
    validationResults: any;
  };
  // Neural Network results
  neuralResults?: {
    architecture: string;
    finalMetrics: {
      trainLoss: number;
      valLoss: number;
      trainAccuracy?: number;
      valAccuracy?: number;
    };
    trainingHistory: {
      loss: number[];
      valLoss: number[];
      accuracy?: number[];
      valAccuracy?: number[];
    };
  };
  // Integrated results
  integratedResults?: {
    ensemblePerformance: Record<string, number>;
    componentContributions: Record<string, number>;
    signals: Array<{
      timestamp: string;
      symbol: string;
      signal: 'buy' | 'sell' | 'hold';
      confidence: number;
    }>;
  };
}

export const AIAdvisor: React.FC<{ 
  analysisContext?: AnalysisContext;
  analysisResults?: AnalysisResults;
}> = ({ analysisContext, analysisResults }) => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<'chat' | 'analysis' | 'results' | 'strategy' | 'risk'>('analysis');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [advisoryResponse, setAdvisoryResponse] = useState<AdvisoryResponse | null>(null);
  const [showContextPanel, setShowContextPanel] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // When analysis context changes, automatically generate recommendations
    if (analysisContext && activeTab === 'analysis') {
      generateContextualRecommendations();
    }
  }, [analysisContext]);

  useEffect(() => {
    // When analysis results are available, switch to results tab
    if (analysisResults && activeTab !== 'results') {
      setActiveTab('results');
      interpretAnalysisResults();
    }
  }, [analysisResults]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const interpretAnalysisResults = async () => {
    if (!analysisResults) return;
    
    setIsLoading(true);
    try {
      const response = await api.post('/api/ai-advisor/analysis/interpret', {
        results: analysisResults,
        context: {
          symbols: analysisContext?.symbols || [],
          timeframe: analysisContext?.dateRange || {},
          analysis_type: analysisResults.type,
          objectives: ['identify_patterns', 'generate_signals', 'optimize_strategy'],
          risk_tolerance: 'moderate',
          capital: 100000
        },
        role: 'quantitative_analyst'
      });

      const interpretation = response.data.interpretation;
      const nextSteps: Recommendation[] = [];
      const insights: string[] = interpretation.insights || [];
      const warnings: string[] = interpretation.warnings || [];

      // Generate specific next-step recommendations based on results
      if (analysisResults.type === 'idtxl' && analysisResults.idtxlResults) {
        const significantConnections = analysisResults.idtxlResults.significantConnections || [];
        
        if (significantConnections.length > 0) {
          // Found causal relationships
          nextSteps.push({
            text: `Detected ${significantConnections.length} significant causal relationships. Next: Use these connections to build a predictive ML model focusing on ${significantConnections[0]?.target || 'identified targets'}.`,
            priority: 'high',
            category: 'next_analysis'
          });
          
          if (significantConnections.some(c => c.strength > 0.7)) {
            nextSteps.push({
              text: 'Strong causal links detected (>0.7). Consider implementing a pairs trading strategy based on these relationships.',
              priority: 'high',
              category: 'strategy_development'
            });
          }
        } else {
          warnings.push('No significant causal relationships found. Consider increasing max lag or using different estimator.');
          nextSteps.push({
            text: 'Try ML classification to identify non-linear patterns that IDTxl might have missed.',
            priority: 'medium',
            category: 'alternative_analysis'
          });
        }
      }
      
      else if (analysisResults.type === 'ml' && analysisResults.mlResults) {
        const metrics = analysisResults.mlResults.finalMetrics;
        const accuracy = metrics.accuracy || metrics.rmse;
        
        if (metrics.accuracy && metrics.accuracy > 0.65) {
          nextSteps.push({
            text: `Model shows ${(metrics.accuracy * 100).toFixed(1)}% accuracy. Next: Implement backtesting with realistic transaction costs and slippage.`,
            priority: 'high',
            category: 'backtesting'
          });
          
          nextSteps.push({
            text: 'Deploy neural network ensemble to potentially improve prediction accuracy by 5-10%.',
            priority: 'medium',
            category: 'model_enhancement'
          });
        }
        
        // Feature importance insights
        const topFeatures = Object.entries(analysisResults.mlResults.featureImportance || {})
          .sort(([,a], [,b]) => b - a)
          .slice(0, 3);
        
        if (topFeatures.length > 0) {
          insights.push(`Top predictive features: ${topFeatures.map(([f]) => f).join(', ')}. Focus strategy development on these indicators.`);
        }
      }
      
      else if (analysisResults.type === 'neural' && analysisResults.neuralResults) {
        const finalLoss = analysisResults.neuralResults.finalMetrics.valLoss;
        const trainLoss = analysisResults.neuralResults.finalMetrics.trainLoss;
        
        if (Math.abs(trainLoss - finalLoss) > 0.1) {
          warnings.push('Significant train/validation loss gap indicates overfitting. Consider increasing dropout or reducing model complexity.');
          nextSteps.push({
            text: 'Apply regularization techniques: increase dropout to 0.3-0.4, add L2 regularization, or use early stopping.',
            priority: 'high',
            category: 'model_improvement'
          });
        }
        
        if (finalLoss < 0.05) {
          nextSteps.push({
            text: 'Excellent model performance. Next: Create ensemble with different architectures (LSTM + Transformer) for production deployment.',
            priority: 'high',
            category: 'production_ready'
          });
        }
      }
      
      else if (analysisResults.type === 'integrated' && analysisResults.integratedResults) {
        const signals = analysisResults.integratedResults.signals || [];
        const recentSignals = signals.filter(s => s.confidence > 0.7);
        
        if (recentSignals.length > 0) {
          nextSteps.push({
            text: `${recentSignals.length} high-confidence signals generated. Next: Backtest with position sizing based on signal confidence levels.`,
            priority: 'high',
            category: 'signal_validation'
          });
          
          nextSteps.push({
            text: 'Implement risk management: 2% per trade, correlation-based position limits, and dynamic stop losses.',
            priority: 'high',
            category: 'risk_management'
          });
        }
      }

      // General next steps based on performance
      if (analysisContext?.historicalPerformance) {
        const sharpe = analysisContext.historicalPerformance.sharpeRatio || 0;
        if (sharpe > 1.5) {
          nextSteps.push({
            text: 'Strong historical performance. Consider scaling up with proper risk controls and live paper trading.',
            priority: 'medium',
            category: 'deployment'
          });
        }
      }

      setAdvisoryResponse({
        recommendations: [...interpretation.recommendations || [], ...nextSteps],
        insights,
        warnings,
        confidence_level: interpretation.confidence_level || 0.8,
        optimal_configuration: interpretation.optimal_configuration
      });
    } catch (error) {
      console.error('Error interpreting results:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateContextualRecommendations = async () => {
    if (!analysisContext) return;
    
    setIsLoading(true);
    try {
      const response = await api.post('/api/ai-advisor/contextual-analysis', {
        context: analysisContext,
        requestType: 'configuration_optimization'
      });

      const recommendations: Recommendation[] = [];
      const insights: string[] = [];
      const warnings: string[] = [];

      // Generate specific recommendations based on context
      if (analysisContext.activeAnalysis === 'idtxl') {
        const idtxl = analysisContext.idtxl!;
        
        if (idtxl.estimator === 'kraskov') {
          recommendations.push({
            text: `For ${analysisContext.symbols.join(', ')} with Kraskov estimator, k=${idtxl.kNeighbors} neighbors is ${idtxl.kNeighbors! < 5 ? 'low' : 'appropriate'} for financial time series. Consider k=5-7 for optimal bias-variance trade-off.`,
            priority: idtxl.kNeighbors! < 5 ? 'high' : 'medium',
            category: 'parameter_optimization'
          });
          
          if (idtxl.noiseLevel! < 1e-8) {
            warnings.push('Very low noise level may cause numerical instability in Kraskov estimator');
          }
        }
        
        if (idtxl.maxLag > 10) {
          insights.push('High lag values (>10) may capture long-term dependencies but increase computational complexity exponentially');
        }
        
        if (idtxl.permutations < 500 && idtxl.significanceLevel < 0.01) {
          recommendations.push({
            text: 'For significance level < 0.01, increase permutations to at least 500 for reliable p-values',
            priority: 'high',
            category: 'statistical_validity'
          });
        }
      }
      
      else if (analysisContext.activeAnalysis === 'ml') {
        const ml = analysisContext.ml!;
        
        if (ml.modelType === 'xgboost') {
          recommendations.push({
            text: `XGBoost with learning rate ${ml.hyperparameters?.learning_rate || 0.1} and ${ml.hyperparameters?.n_boost_rounds || 100} rounds. For financial data, try learning_rate=0.05 with 200-300 rounds for better generalization.`,
            priority: 'high',
            category: 'hyperparameter_tuning'
          });
        }
        
        if (ml.validation === 'time_series_cv' && ml.testSize < 0.2) {
          warnings.push('Test size < 20% may not provide reliable out-of-sample performance estimates for time series');
        }
        
        if (ml.target === 'volatility' && !ml.features?.technical_indicators?.includes('atr')) {
          recommendations.push({
            text: 'For volatility prediction, include ATR (Average True Range) in technical indicators',
            priority: 'medium',
            category: 'feature_engineering'
          });
        }
      }
      
      else if (analysisContext.activeAnalysis === 'neural') {
        const nn = analysisContext.neural!;
        
        if (nn.architecture === 'lstm' || nn.architecture === 'gru') {
          if (!nn.bidirectional && analysisContext.symbols.length > 1) {
            recommendations.push({
              text: 'For multi-asset analysis, bidirectional LSTM/GRU can capture both forward and backward temporal dependencies',
              priority: 'medium',
              category: 'architecture'
            });
          }
          
          if (nn.layers && nn.layers[0] > 256) {
            warnings.push('Large LSTM/GRU layers (>256 units) may lead to overfitting on financial data');
          }
        }
        
        if (nn.architecture === 'transformer') {
          insights.push(`Transformer with ${nn.attentionHeads} heads and ${nn.encoderLayers} layers is computationally intensive. Ensure GPU acceleration is enabled.`);
          
          if (nn.batchSize < 32) {
            recommendations.push({
              text: 'Transformers benefit from larger batch sizes (64-128) for stable training',
              priority: 'medium',
              category: 'training'
            });
          }
        }
        
        if (nn.dropoutRate > 0.5) {
          warnings.push('High dropout rate (>0.5) may prevent the network from learning complex patterns');
        }
      }
      
      else if (analysisContext.activeAnalysis === 'integrated') {
        const integrated = analysisContext.integrated!;
        const totalWeight = (integrated.components.idtxl.weight + 
                           integrated.components.ml.weight + 
                           integrated.components.nn.weight);
        
        if (Math.abs(totalWeight - 1.0) > 0.01) {
          warnings.push(`Component weights sum to ${totalWeight.toFixed(2)}, not 1.0. This may affect signal generation.`);
        }
        
        if (integrated.ensembleMethod === 'stacking' && integrated.signalThreshold < 0.7) {
          recommendations.push({
            text: 'Stacking ensemble with low signal threshold (<0.7) may generate too many false signals',
            priority: 'high',
            category: 'signal_generation'
          });
        }
      }

      // General insights based on symbols and date range
      const dateRange = analysisContext.dateRange;
      const daysDiff = Math.floor((new Date(dateRange.end).getTime() - new Date(dateRange.start).getTime()) / (1000 * 60 * 60 * 24));
      
      if (daysDiff < 365) {
        insights.push('Less than 1 year of data may not capture full market cycles and seasonal patterns');
      }
      
      if (analysisContext.symbols.includes('BTC') || analysisContext.symbols.includes('ETH')) {
        insights.push('Cryptocurrency markets show 24/7 trading patterns - ensure your analysis accounts for weekend effects');
      }

      setAdvisoryResponse({
        recommendations,
        insights,
        warnings,
        confidence_level: 0.85,
        optimal_configuration: response.data.optimal_configuration
      });
    } catch (error) {
      console.error('Error generating recommendations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
      context: analysisContext
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await api.post('/api/ai-advisor/chat', {
        message: inputMessage,
        context: {
          current_tab: activeTab,
          analysis_context: analysisContext,
          conversation_history: messages.slice(-10) // Last 10 messages for context
        }
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderContextPanel = () => {
    if (!analysisContext) return null;

    return (
      <div className="bg-gray-50 border-b p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-semibold text-sm">Current Analysis Context</h4>
          <button
            onClick={() => setShowContextPanel(!showContextPanel)}
            className="text-gray-500 hover:text-gray-700"
          >
            <ChevronDown className={`w-4 h-4 transition-transform ${showContextPanel ? '' : '-rotate-90'}`} />
          </button>
        </div>
        
        <AnimatePresence>
          {showContextPanel && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="space-y-2 text-sm"
            >
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">Analysis Type:</span>
                <span className="font-medium capitalize">{analysisContext.activeAnalysis}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">Symbols:</span>
                <div className="flex flex-wrap gap-1">
                  {analysisContext.symbols.map(symbol => (
                    <span key={symbol} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                      {symbol}
                    </span>
                  ))}
                </div>
              </div>
              
              {analysisContext.activeAnalysis === 'idtxl' && analysisContext.idtxl && (
                <div className="pl-4 border-l-2 border-gray-300">
                  <p>Estimator: {analysisContext.idtxl.estimator}</p>
                  <p>Max Lag: {analysisContext.idtxl.maxLag}</p>
                  <p>Significance: {analysisContext.idtxl.significanceLevel}</p>
                </div>
              )}
              
              {analysisContext.activeAnalysis === 'ml' && analysisContext.ml && (
                <div className="pl-4 border-l-2 border-gray-300">
                  <p>Model: {analysisContext.ml.modelType}</p>
                  <p>Target: {analysisContext.ml.target}</p>
                  <p>Validation: {analysisContext.ml.validation}</p>
                </div>
              )}
              
              {analysisContext.activeAnalysis === 'neural' && analysisContext.neural && (
                <div className="pl-4 border-l-2 border-gray-300">
                  <p>Architecture: {analysisContext.neural.architecture.toUpperCase()}</p>
                  <p>Epochs: {analysisContext.neural.epochs}</p>
                  <p>Learning Rate: {analysisContext.neural.learningRate}</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  const renderChat = () => (
    <div className="flex flex-col h-full">
      {renderContextPanel()}
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-semibold mb-2">PhD-Level Financial AI Advisor</h3>
            <p className="text-sm">I have full context of your analysis configuration. Ask me anything about:</p>
            <div className="mt-4 text-left max-w-md mx-auto">
              <ul className="space-y-1 text-sm">
                <li>• Optimal parameter settings for your specific analysis</li>
                <li>• Advanced quantitative finance techniques</li>
                <li>• Risk management and portfolio optimization</li>
                <li>• Market microstructure and execution strategies</li>
                <li>• Statistical validation and backtesting methodologies</li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map(message => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                <p className="text-xs mt-1 opacity-70">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 px-4 py-2 rounded-lg">
              <Loader2 className="w-5 h-5 animate-spin" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="border-t p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about your analysis configuration..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );

  const renderResultsInterpreter = () => (
    <div className="flex flex-col h-full">
      {renderContextPanel()}
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div>
          <h3 className="text-lg font-semibold mb-4">PhD-Level Results Interpretation</h3>
          
          {analysisResults ? (
            <div className="space-y-4">
              {/* Results Summary */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold mb-2 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-gray-600" />
                  Analysis Results Summary
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Analysis Type:</span>
                    <span className="ml-2 font-medium capitalize">{analysisResults.type}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Completed:</span>
                    <span className="ml-2 font-medium">
                      {new Date(analysisResults.timestamp).toLocaleString()}
                    </span>
                  </div>
                  
                  {/* Type-specific metrics */}
                  {analysisResults.type === 'idtxl' && analysisResults.idtxlResults && (
                    <>
                      <div>
                        <span className="text-gray-600">Causal Links:</span>
                        <span className="ml-2 font-medium">
                          {analysisResults.idtxlResults.significantConnections.length}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Processing Time:</span>
                        <span className="ml-2 font-medium">
                          {analysisResults.idtxlResults.processingTime.toFixed(2)}s
                        </span>
                      </div>
                    </>
                  )}
                  
                  {analysisResults.type === 'ml' && analysisResults.mlResults && (
                    <>
                      <div>
                        <span className="text-gray-600">Model:</span>
                        <span className="ml-2 font-medium">
                          {analysisResults.mlResults.modelType}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Accuracy:</span>
                        <span className="ml-2 font-medium">
                          {(analysisResults.mlResults.finalMetrics.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    </>
                  )}
                  
                  {analysisResults.type === 'neural' && analysisResults.neuralResults && (
                    <>
                      <div>
                        <span className="text-gray-600">Architecture:</span>
                        <span className="ml-2 font-medium">
                          {analysisResults.neuralResults.architecture.toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Val Loss:</span>
                        <span className="ml-2 font-medium">
                          {analysisResults.neuralResults.finalMetrics.valLoss.toFixed(4)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Generate Interpretation Button */}
              <div className="mb-6">
                <button
                  onClick={interpretAnalysisResults}
                  disabled={isLoading}
                  className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing Results...
                    </>
                  ) : (
                    <>
                      <Brain className="w-5 h-5 mr-2" />
                      Generate Expert Interpretation
                    </>
                  )}
                </button>
              </div>

              {/* AI Interpretation */}
              {advisoryResponse && (
                <div className="space-y-4">
                  {/* Next Steps Recommendations */}
                  {advisoryResponse.recommendations.filter(r => r.category === 'next_analysis' || 
                                                                    r.category === 'backtesting' || 
                                                                    r.category === 'strategy_development' ||
                                                                    r.category === 'deployment').length > 0 && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <h4 className="font-semibold mb-3 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
                        Recommended Next Steps
                      </h4>
                      <div className="space-y-3">
                        {advisoryResponse.recommendations
                          .filter(r => ['next_analysis', 'backtesting', 'strategy_development', 'deployment'].includes(r.category))
                          .map((rec, index) => (
                            <div key={index} className="flex items-start">
                              <span className={`inline-block w-6 h-6 rounded-full text-white text-xs flex items-center justify-center mt-0.5 mr-3 flex-shrink-0 ${
                                rec.priority === 'high' ? 'bg-red-500' :
                                rec.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                              }`}>
                                {index + 1}
                              </span>
                              <div className="flex-1">
                                <p className="text-sm">{rec.text}</p>
                                <div className="mt-1 flex items-center space-x-2">
                                  <span className="text-xs text-gray-500">{rec.category.replace('_', ' ')}</span>
                                  <button className="text-xs text-blue-600 hover:text-blue-700 underline">
                                    Learn more
                                  </button>
                                </div>
                              </div>
                            </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Key Insights */}
                  {advisoryResponse.insights.length > 0 && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h4 className="font-semibold mb-3 flex items-center">
                        <Info className="w-5 h-5 mr-2 text-green-600" />
                        Key Insights & Patterns
                      </h4>
                      <ul className="space-y-2">
                        {advisoryResponse.insights.map((insight, index) => (
                          <li key={index} className="text-sm flex items-start">
                            <Zap className="w-4 h-4 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Warnings & Considerations */}
                  {advisoryResponse.warnings.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-semibold mb-3 flex items-center">
                        <AlertTriangle className="w-5 h-5 mr-2 text-yellow-600" />
                        Important Considerations
                      </h4>
                      <ul className="space-y-2">
                        {advisoryResponse.warnings.map((warning, index) => (
                          <li key={index} className="text-sm flex items-start">
                            <span className="text-yellow-600 mr-2">⚠</span>
                            <span>{warning}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="grid grid-cols-2 gap-4 mt-6">
                    <button className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 flex items-center justify-center">
                      <Download className="w-4 h-4 mr-2" />
                      Export Report
                    </button>
                    <button 
                      onClick={() => setActiveTab('strategy')}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center"
                    >
                      Build Strategy
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <FileText className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-semibold mb-2">No Analysis Results Yet</h3>
              <p className="text-sm">Run an analysis to see PhD-level interpretation and next-step recommendations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderAnalysisAdvisor = () => (
    <div className="flex flex-col h-full">
      {renderContextPanel()}
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div>
          <h3 className="text-lg font-semibold mb-4">AI Analysis Configuration Assistant</h3>
          
          <div className="mb-6">
            <button
              onClick={generateContextualRecommendations}
              disabled={isLoading || !analysisContext}
              className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing Configuration...
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5 mr-2" />
                  Generate AI Recommendations
                </>
              )}
            </button>
          </div>
        </div>

        {advisoryResponse && (
          <div className="space-y-4">
            {/* Recommendations */}
            {advisoryResponse.recommendations.length > 0 && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold mb-3 flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2 text-blue-600" />
                  Configuration Recommendations
                </h4>
                <div className="space-y-3">
                  {advisoryResponse.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start">
                      <span className={`inline-block w-2 h-2 rounded-full mt-1.5 mr-3 flex-shrink-0 ${
                        rec.priority === 'high' ? 'bg-red-500' :
                        rec.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                      }`} />
                      <div className="flex-1">
                        <p className="text-sm">{rec.text}</p>
                        <span className="text-xs text-gray-500 mt-1">{rec.category.replace('_', ' ')}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Insights */}
            {advisoryResponse.insights.length > 0 && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-semibold mb-3 flex items-center">
                  <Info className="w-5 h-5 mr-2 text-green-600" />
                  Expert Insights
                </h4>
                <ul className="space-y-2">
                  {advisoryResponse.insights.map((insight, index) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="text-green-600 mr-2">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Warnings */}
            {advisoryResponse.warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-semibold mb-3 flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2 text-yellow-600" />
                  Important Warnings
                </h4>
                <ul className="space-y-2">
                  {advisoryResponse.warnings.map((warning, index) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="text-yellow-600 mr-2">⚠</span>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Confidence Level */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">AI Confidence Level</span>
                <div className="flex items-center">
                  <div className="w-32 bg-gray-200 rounded-full h-2 mr-3">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${advisoryResponse.confidence_level * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold">
                    {(advisoryResponse.confidence_level * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Optimal Configuration Suggestion */}
            {advisoryResponse.optimal_configuration && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold mb-3 flex items-center">
                  <Zap className="w-5 h-5 mr-2 text-purple-600" />
                  Optimal Configuration Detected
                </h4>
                <p className="text-sm text-gray-700 mb-3">
                  Based on your objectives and historical performance patterns, here's an optimized configuration:
                </p>
                <button className="text-sm text-purple-600 hover:text-purple-700 underline">
                  Apply Optimal Settings
                </button>
              </div>
            )}
          </div>
        )}

        {/* Quick Tips */}
        <div className="mt-8 p-4 bg-gray-100 rounded-lg">
          <h4 className="font-semibold mb-2 flex items-center">
            <HelpCircle className="w-4 h-4 mr-2" />
            Quick Tips
          </h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• The AI advisor considers your specific symbols, timeframes, and analysis objectives</li>
            <li>• Recommendations are based on academic research and industry best practices</li>
            <li>• Always validate recommendations with backtesting before live deployment</li>
            <li>• Click on any recommendation to get more detailed explanation</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderStrategyBuilder = () => (
    <div className="flex flex-col h-full">
      {renderContextPanel()}
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <h3 className="text-lg font-semibold mb-4">AI-Powered Strategy Builder</h3>
        
        {analysisResults ? (
          <div className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
                Strategy Development Based on Analysis Results
              </h4>
              <p className="text-sm text-blue-800">
                Based on your {analysisResults.type} analysis results, I'll help you build a comprehensive 
                trading strategy with entry/exit rules, position sizing, and risk management.
              </p>
            </div>

            {/* Strategy Components */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Signal Generation */}
              <div className="bg-white border rounded-lg p-4">
                <h5 className="font-semibold mb-2 flex items-center">
                  <Zap className="w-4 h-4 mr-2 text-yellow-500" />
                  Signal Generation
                </h5>
                <p className="text-sm text-gray-600">
                  {analysisResults.type === 'idtxl' && 'Using causal relationships for predictive signals'}
                  {analysisResults.type === 'ml' && 'Using ML predictions with feature importance weights'}
                  {analysisResults.type === 'neural' && 'Using neural network probability outputs'}
                  {analysisResults.type === 'integrated' && 'Using ensemble consensus signals'}
                </p>
              </div>

              {/* Risk Management */}
              <div className="bg-white border rounded-lg p-4">
                <h5 className="font-semibold mb-2 flex items-center">
                  <Shield className="w-4 h-4 mr-2 text-red-500" />
                  Risk Management
                </h5>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Position sizing: 2% risk per trade</li>
                  <li>• Stop loss: Dynamic based on ATR</li>
                  <li>• Max exposure: 20% of portfolio</li>
                </ul>
              </div>
            </div>

            {/* Generate Strategy Button */}
            <button 
              onClick={async () => {
                setIsLoading(true);
                try {
                  const response = await api.post('/api/ai-advisor/strategy/recommend', {
                    analysis_results: analysisResults,
                    market_data: { volatility: 'moderate', trend: 'bullish' },
                    risk_profile: { 
                      max_drawdown: 0.15,
                      target_sharpe: 1.5,
                      capital: 100000
                    }
                  });
                  setAdvisoryResponse(response.data.strategy);
                } catch (error) {
                  console.error('Error generating strategy:', error);
                } finally {
                  setIsLoading(false);
                }
              }}
              disabled={isLoading}
              className="w-full py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Building Strategy...
                </>
              ) : (
                <>
                  <TrendingUp className="w-5 h-5 mr-2" />
                  Generate Complete Trading Strategy
                </>
              )}
            </button>

            {/* Strategy Output */}
            {advisoryResponse && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                <h4 className="font-semibold mb-4">Generated Trading Strategy</h4>
                <div className="space-y-4">
                  {advisoryResponse.recommendations.map((rec, idx) => (
                    <div key={idx} className="border-l-4 border-blue-500 pl-4">
                      <p className="text-sm">{rec.text}</p>
                    </div>
                  ))}
                </div>
                
                <div className="mt-6 flex space-x-4">
                  <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                    Backtest Strategy
                  </button>
                  <button className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
                    Save Strategy
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <TrendingUp className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-semibold mb-2">No Analysis Results</h3>
            <p className="text-sm">Complete an analysis first to build data-driven trading strategies</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderRiskConsultant = () => (
    <div className="p-6">
      <h3 className="text-lg font-semibold mb-4">Risk Management Consultant</h3>
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-sm text-red-800">
          Advanced risk analysis including VaR calculations, stress testing, and portfolio optimization
          based on your current configuration.
        </p>
      </div>
    </div>
  );

  return (
    <div className="flex h-full bg-white rounded-lg shadow-lg">
      {/* Sidebar */}
      <div className="w-64 border-r bg-gray-50">
        <div className="p-4">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Brain className="w-6 h-6 mr-2" />
            AI Advisor
          </h2>
          
          <nav className="space-y-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full text-left px-4 py-2 rounded-lg flex items-center transition-colors ${
                activeTab === 'chat' ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'
              }`}
            >
              <MessageSquare className="w-5 h-5 mr-3" />
              Chat Assistant
            </button>
            
            <button
              onClick={() => setActiveTab('analysis')}
              className={`w-full text-left px-4 py-2 rounded-lg flex items-center transition-colors ${
                activeTab === 'analysis' ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'
              }`}
            >
              <BarChart3 className="w-5 h-5 mr-3" />
              Analysis Config
            </button>
            
            <button
              onClick={() => setActiveTab('results')}
              className={`w-full text-left px-4 py-2 rounded-lg flex items-center transition-colors ${
                activeTab === 'results' ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'
              }`}
              disabled={!analysisResults}
            >
              <FileText className="w-5 h-5 mr-3" />
              Results Interpreter
              {analysisResults && (
                <span className="ml-auto text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">New</span>
              )}
            </button>
            
            <button
              onClick={() => setActiveTab('strategy')}
              className={`w-full text-left px-4 py-2 rounded-lg flex items-center transition-colors ${
                activeTab === 'strategy' ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'
              }`}
            >
              <TrendingUp className="w-5 h-5 mr-3" />
              Strategy Builder
            </button>
            
            <button
              onClick={() => setActiveTab('risk')}
              className={`w-full text-left px-4 py-2 rounded-lg flex items-center transition-colors ${
                activeTab === 'risk' ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'
              }`}
            >
              <Shield className="w-5 h-5 mr-3" />
              Risk Consultant
            </button>
          </nav>
        </div>

        <div className="p-4 border-t">
          <div className="text-sm text-gray-600">
            <p className="font-semibold">AI Model</p>
            <p>Claude Opus 4</p>
            <p className="text-xs mt-1">PhD-level financial expertise</p>
            <p className="text-xs text-green-600 mt-1">
              {analysisContext ? '✓ Context-aware mode' : '○ No context'}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {activeTab === 'chat' && renderChat()}
        {activeTab === 'analysis' && renderAnalysisAdvisor()}
        {activeTab === 'results' && renderResultsInterpreter()}
        {activeTab === 'strategy' && renderStrategyBuilder()}
        {activeTab === 'risk' && renderRiskConsultant()}
      </div>
    </div>
  );
};

export default AIAdvisor;