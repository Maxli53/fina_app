import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQuery } from '@tanstack/react-query';
import { 
  Brain, 
  Settings, 
  Play, 
  Download, 
  Info,
  Loader2,
  CheckCircle,
  AlertCircle,
  Network,
  BarChart,
  Cpu,
  Zap,
  HelpCircle,
  TrendingUp,
  GitBranch,
  Layers,
  ToggleLeft,
  ToggleRight,
  Search
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import api, { analysisService, dataService } from '../services/api';
import { format } from 'date-fns';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import toast from 'react-hot-toast';

// Parameter optimization types
interface ParameterOptimization {
  enabled: boolean;
  type: 'range' | 'list';
  // For numeric ranges
  min?: number;
  max?: number;
  step?: number;
  // For categorical/list selections
  values?: any[];
}

// Track which parameters have optimization enabled
type OptimizationSettings = Record<string, ParameterOptimization>;

// Optimization configuration
interface OptimizationConfig {
  method: 'grid_search' | 'random_search' | 'bayesian' | 'genetic';
  objective: string; // Dynamic based on analysis type
  maxIterations?: number;
  crossValidationFolds?: number;
}

// Analysis-specific optimization objectives focused on next-bar prediction
const OPTIMIZATION_OBJECTIVES = {
  idtxl: [
    { 
      value: 'max_predictive_te', 
      label: 'Maximize Predictive Transfer Entropy',
      description: 'Find connections where source movements predict target\'s next bar'
    },
    { 
      value: 'max_lead_time', 
      label: 'Maximize Lead-Time Advantage',
      description: 'Find earliest reliable predictors for next-bar movements'
    },
    { 
      value: 'max_network_consensus', 
      label: 'Maximize Network Consensus',
      description: 'Find parameters where multiple sources agree on next-bar direction'
    },
    { 
      value: 'min_false_signals', 
      label: 'Minimize False Signals',
      description: 'Reduce spurious connections that don\'t predict next bar'
    }
  ],
  ml: [
    { 
      value: 'max_directional_accuracy', 
      label: 'Maximize Next-Bar Directional Accuracy',
      description: 'Highest % of correct up/down predictions for next bar'
    },
    { 
      value: 'max_profit_factor', 
      label: 'Maximize Profit Factor',
      description: 'Optimize for profitable predictions (correct × magnitude)'
    },
    { 
      value: 'min_false_positives', 
      label: 'Minimize False Positives',
      description: 'Reduce costly wrong "up" predictions'
    },
    { 
      value: 'max_consistency', 
      label: 'Maximize Cross-Period Consistency',
      description: 'Find parameters that work across different market regimes'
    }
  ],
  neural: [
    { 
      value: 'max_next_bar_accuracy', 
      label: 'Maximize Next-Bar Prediction Accuracy',
      description: 'Best up/down classification for next bar'
    },
    { 
      value: 'max_confidence_calibration', 
      label: 'Maximize Confidence Calibration',
      description: 'High confidence on correct predictions, low on wrong ones'
    },
    { 
      value: 'min_volatility_sensitivity', 
      label: 'Minimize Volatility Sensitivity',
      description: 'Stable predictions across different volatility regimes'
    },
    { 
      value: 'max_early_stopping', 
      label: 'Optimize Early Prediction',
      description: 'Best accuracy with minimal training time'
    }
  ],
  integrated: [
    { 
      value: 'max_ensemble_agreement', 
      label: 'Maximize Ensemble Agreement',
      description: 'All methods agree on next-bar direction'
    },
    { 
      value: 'max_signal_quality', 
      label: 'Maximize Signal Quality Score',
      description: 'Clear, high-confidence next-bar predictions'
    },
    { 
      value: 'optimize_weights', 
      label: 'Optimize Component Weights',
      description: 'Find best combination for next-bar accuracy'
    }
  ]
};

// Parameter Input Component with Optimization
interface OptimizableInputProps {
  name: string;
  label: string;
  type: 'number' | 'select';
  register: any;
  value?: any;
  min?: number;
  max?: number;
  step?: number | string;
  options?: { value: string; label: string }[];
  tooltip?: string;
  optimization: ParameterOptimization | undefined;
  onOptimizationChange: (enabled: boolean, config?: Partial<ParameterOptimization>) => void;
}

const OptimizableInput: React.FC<OptimizableInputProps> = ({
  name,
  label,
  type,
  register,
  value,
  min,
  max,
  step,
  options,
  tooltip,
  optimization,
  onOptimizationChange
}) => {
  const isOptimized = optimization?.enabled || false;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          {label}
          {tooltip && (
            <span className="ml-1 text-gray-400 cursor-help" title={tooltip}>
              ⓘ
            </span>
          )}
        </label>
        <button
          type="button"
          onClick={() => onOptimizationChange(!isOptimized, {
            type: type === 'number' ? 'range' : 'list',
            min: min,
            max: max,
            step: typeof step === 'number' ? step : 1
          })}
          className="text-sm flex items-center space-x-1 text-purple-600 hover:text-purple-700"
        >
          {isOptimized ? (
            <><ToggleRight className="w-4 h-4" /> <span>Optimize</span></>
          ) : (
            <><ToggleLeft className="w-4 h-4" /> <span>Fixed</span></>
          )}
        </button>
      </div>
      
      {!isOptimized ? (
        // Single value input
        type === 'number' ? (
          <input
            type="number"
            {...register(name, { valueAsNumber: true })}
            min={min}
            max={max}
            step={step}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        ) : (
          <select
            {...register(name)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {options?.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        )
      ) : (
        // Optimization range/list inputs
        <div className="space-y-2">
          {type === 'number' ? (
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="text-xs text-gray-500">Min</label>
                <input
                  type="number"
                  value={optimization?.min || min}
                  onChange={(e) => onOptimizationChange(true, { ...optimization, min: parseFloat(e.target.value) })}
                  className="w-full px-3 py-1.5 border border-purple-300 rounded-lg bg-purple-50 focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Max</label>
                <input
                  type="number"
                  value={optimization?.max || max}
                  onChange={(e) => onOptimizationChange(true, { ...optimization, max: parseFloat(e.target.value) })}
                  className="w-full px-3 py-1.5 border border-purple-300 rounded-lg bg-purple-50 focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Step</label>
                <input
                  type="number"
                  value={optimization?.step || (typeof step === 'number' ? step : 1)}
                  onChange={(e) => onOptimizationChange(true, { ...optimization, step: parseFloat(e.target.value) })}
                  className="w-full px-3 py-1.5 border border-purple-300 rounded-lg bg-purple-50 focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
          ) : (
            <div>
              <label className="text-xs text-gray-500">Select options to test</label>
              <div className="space-y-1 mt-1">
                {options?.map(opt => (
                  <label key={opt.value} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={optimization?.values?.includes(opt.value) || false}
                      onChange={(e) => {
                        const values = optimization?.values || [];
                        const newValues = e.target.checked 
                          ? [...values, opt.value]
                          : values.filter(v => v !== opt.value);
                        onOptimizationChange(true, { ...optimization, values: newValues });
                      }}
                      className="rounded border-purple-300 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="text-sm">{opt.label}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Comprehensive Analysis Configuration Schemas
const baseSchema = z.object({
  symbols: z.array(z.string()).min(1, 'Select at least 1 symbol for analysis'),
  startDate: z.string().min(1, 'Start date is required'),
  endDate: z.string().min(1, 'End date is required'),
  useGpu: z.boolean().optional(),
});

// IDTxl Schema
const idtxlSchema = baseSchema.extend({
  analysisType: z.enum(['transfer_entropy', 'mutual_information', 'both']),
  maxLag: z.number().min(1).max(20),
  estimator: z.enum(['gaussian', 'symbolic', 'kraskov']),
  significanceLevel: z.number().min(0.001).max(0.1),
  permutations: z.number().min(50).max(1000),
  // Estimator-specific settings
  kNeighbors: z.number().min(1).max(10).optional(), // For Kraskov
  noiseLevel: z.number().min(1e-10).max(1e-6).optional(), // For Kraskov
  alphabetSize: z.number().min(2).max(5).optional(), // For Symbolic
});

// ML Schema
const mlSchema = baseSchema.extend({
  modelType: z.enum(['random_forest', 'xgboost', 'svm', 'logistic_regression']),
  target: z.enum(['direction', 'returns', 'volatility']),
  predictionHorizon: z.number().min(1).max(20),
  validation: z.enum(['time_series_cv', 'walk_forward', 'purged_cv', 'train_test_split']),
  testSize: z.number().min(0.1).max(0.5),
  // Model-specific hyperparameters
  // Random Forest
  nEstimators: z.number().min(10).max(1000).optional(),
  maxDepth: z.number().min(1).max(50).optional(),
  minSamplesSplit: z.number().min(2).max(20).optional(),
  minSamplesLeaf: z.number().min(1).max(10).optional(),
  // XGBoost
  learningRate: z.number().min(0.001).max(0.3).optional(),
  nBoostRounds: z.number().min(10).max(1000).optional(),
  maxDepthXgb: z.number().min(1).max(15).optional(),
  subsample: z.number().min(0.5).max(1.0).optional(),
  colsampleBytree: z.number().min(0.5).max(1.0).optional(),
  // SVM
  kernel: z.enum(['linear', 'rbf', 'poly', 'sigmoid']).optional(),
  C: z.number().min(0.01).max(100).optional(),
  gamma: z.enum(['scale', 'auto']).or(z.number().min(0.0001).max(1)).optional(),
  // Feature engineering
  technicalIndicators: z.array(z.string()).optional(),
  laggedFeatures: z.number().min(1).max(20).optional(),
  rollingWindows: z.array(z.number()).optional(),
});

// Neural Network Schema
const nnSchema = baseSchema.extend({
  architecture: z.enum(['lstm', 'gru', 'cnn', 'transformer']),
  // Common settings
  epochs: z.number().min(1).max(1000),
  batchSize: z.number().min(1).max(512),
  optimizer: z.enum(['adam', 'sgd', 'rmsprop']),
  learningRate: z.number().min(0.0001).max(0.1),
  dropoutRate: z.number().min(0).max(0.5),
  earlyStoppingPatience: z.number().min(5).max(50),
  batchNormalization: z.boolean(),
  // Architecture-specific
  // LSTM/GRU
  layers: z.array(z.number()).min(1).optional(),
  bidirectional: z.boolean().optional(),
  returnSequences: z.boolean().optional(),
  // CNN
  filters: z.array(z.number()).optional(),
  kernelSize: z.number().min(1).max(10).optional(),
  poolingSize: z.number().min(1).max(5).optional(),
  // Transformer
  attentionHeads: z.number().min(4).max(16).optional(),
  encoderLayers: z.number().min(2).max(8).optional(),
  dModel: z.number().min(64).max(512).optional(),
  feedforwardDim: z.number().min(128).max(2048).optional(),
  // Dense layers (for all architectures)
  denseLayers: z.array(z.number()).optional(),
});

// Integrated Analysis Schema
const integratedSchema = baseSchema.extend({
  analysisComponents: z.object({
    idtxl: z.object({
      enabled: z.boolean(),
      weight: z.number().min(0).max(1),
      config: idtxlSchema.partial(),
    }),
    ml: z.object({
      enabled: z.boolean(),
      weight: z.number().min(0).max(1),
      config: mlSchema.partial(),
    }),
    nn: z.object({
      enabled: z.boolean(),
      weight: z.number().min(0).max(1),
      config: nnSchema.partial(),
    }),
  }),
  ensembleMethod: z.enum(['weighted_average', 'voting', 'stacking', 'blending']),
  signalThreshold: z.number().min(0).max(1),
});

type IDTxlFormData = z.infer<typeof idtxlSchema>;
type MLFormData = z.infer<typeof mlSchema>;
type NNFormData = z.infer<typeof nnSchema>;
type IntegratedFormData = z.infer<typeof integratedSchema>;

// Parameter range types for optimization
interface ParameterRange {
  enabled: boolean;
  min: number;
  max: number;
  step?: number;
}

interface OptimizationConfig {
  enabled: boolean;
  method: 'grid_search' | 'random_search' | 'bayesian' | 'genetic';
  objective: 'sharpe_ratio' | 'returns' | 'accuracy' | 'min_drawdown';
  parameterRanges: Record<string, ParameterRange>;
  maxIterations?: number;
  crossValidationFolds?: number;
}

const Analysis: React.FC = () => {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'idtxl' | 'ml' | 'neural' | 'integrated'>('idtxl');
  const [showAIAdvisor, setShowAIAdvisor] = useState(false);
  
  // Optimization settings per tab
  const [idtxlOptimization, setIdtxlOptimization] = useState<OptimizationSettings>({});
  const [mlOptimization, setMlOptimization] = useState<OptimizationSettings>({});
  const [nnOptimization, setNnOptimization] = useState<OptimizationSettings>({});
  
  // Global optimization config
  const [optimizationConfig, setOptimizationConfig] = useState<OptimizationConfig>({
    method: 'bayesian',
    objective: 'max_predictive_te', // Default for IDTxl
    maxIterations: 100,
    crossValidationFolds: 5
  });
  
  // Update default objective when tab changes
  useEffect(() => {
    const defaultObjectives = {
      idtxl: 'max_predictive_te',
      ml: 'max_directional_accuracy',
      neural: 'max_next_bar_accuracy',
      integrated: 'max_ensemble_agreement'
    };
    setOptimizationConfig(prev => ({
      ...prev,
      objective: defaultObjectives[activeTab]
    }));
  }, [activeTab]);
  
  // Check if any optimization is enabled
  const hasOptimization = () => {
    const currentOptimization = activeTab === 'idtxl' ? idtxlOptimization : 
                               activeTab === 'ml' ? mlOptimization : 
                               activeTab === 'neural' ? nnOptimization : {};
    return Object.values(currentOptimization).some(opt => opt.enabled);
  };

  // Separate forms for each analysis type
  const idtxlForm = useForm<IDTxlFormData>({
    resolver: zodResolver(idtxlSchema),
    defaultValues: {
      symbols: [],
      analysisType: 'both',
      maxLag: 5,
      estimator: 'kraskov',
      significanceLevel: 0.05,
      permutations: 200,
      kNeighbors: 3,
      noiseLevel: 1e-8,
      alphabetSize: 2,
      useGpu: false,
    },
  });

  const mlForm = useForm<MLFormData>({
    resolver: zodResolver(mlSchema),
    defaultValues: {
      symbols: [],
      modelType: 'random_forest',
      target: 'direction',
      predictionHorizon: 1,
      validation: 'time_series_cv',
      testSize: 0.2,
      // Random Forest defaults
      nEstimators: 100,
      maxDepth: 10,
      minSamplesSplit: 5,
      minSamplesLeaf: 2,
      // XGBoost defaults
      learningRate: 0.1,
      nBoostRounds: 100,
      maxDepthXgb: 6,
      subsample: 0.8,
      colsampleBytree: 0.8,
      // SVM defaults
      kernel: 'rbf',
      C: 1.0,
      gamma: 'scale',
      // Feature engineering
      technicalIndicators: ['rsi', 'macd', 'bb'],
      laggedFeatures: 5,
      rollingWindows: [5, 20, 50],
      useGpu: false,
    },
  });

  const nnForm = useForm<NNFormData>({
    resolver: zodResolver(nnSchema),
    defaultValues: {
      symbols: [],
      architecture: 'lstm',
      epochs: 100,
      batchSize: 32,
      optimizer: 'adam',
      learningRate: 0.001,
      dropoutRate: 0.2,
      earlyStoppingPatience: 10,
      batchNormalization: false,
      // LSTM defaults
      layers: [128, 64],
      bidirectional: false,
      returnSequences: false,
      // CNN defaults
      filters: [64, 128],
      kernelSize: 3,
      poolingSize: 2,
      // Transformer defaults
      attentionHeads: 8,
      encoderLayers: 4,
      dModel: 128,
      feedforwardDim: 512,
      // Dense layers
      denseLayers: [64, 32],
      useGpu: true,
    },
  });

  const integratedForm = useForm<IntegratedFormData>({
    resolver: zodResolver(integratedSchema),
    defaultValues: {
      symbols: [],
      analysisComponents: {
        idtxl: { enabled: true, weight: 0.33, config: {} },
        ml: { enabled: true, weight: 0.33, config: {} },
        nn: { enabled: true, weight: 0.34, config: {} },
      },
      ensembleMethod: 'weighted_average',
      signalThreshold: 0.6,
      useGpu: true,
    },
  });

  // Watch form values for dynamic UI updates
  const watchedEstimator = idtxlForm.watch('estimator');
  const watchedModelType = mlForm.watch('modelType');
  const watchedArchitecture = nnForm.watch('architecture');
  const watchedKernel = mlForm.watch('kernel');

  // Search symbols
  const { data: searchResults } = useQuery({
    queryKey: ['symbol-search', searchQuery],
    queryFn: () => dataService.searchSymbols(searchQuery),
    enabled: searchQuery.length > 0,
  });

  // Get AI recommendations based on current configuration
  const getAIRecommendations = useMutation({
    mutationFn: async (config: any) => {
      const context = {
        analysisType: activeTab,
        configuration: config,
        symbols: selectedSymbols,
        historicalPerformance: {}, // Could fetch actual performance data
      };
      
      return api.post('/api/ai-advisor/analysis-recommendation', context);
    },
    onSuccess: (data) => {
      toast.success('AI recommendations received');
    },
  });

  // Run analysis mutations
  const runIDTxlAnalysis = useMutation({
    mutationFn: async (data: IDTxlFormData) => {
      const timeSeriesData: any = {};
      for (const symbol of data.symbols) {
        const historicalData = await dataService.getHistoricalData(
          symbol,
          data.startDate,
          data.endDate,
          '1d'
        );
        timeSeriesData[symbol] = historicalData;
      }

      const config = {
        analysis_type: data.analysisType,
        max_lag: data.maxLag,
        estimator: data.estimator,
        significance_level: data.significanceLevel,
        permutations: data.permutations,
        variables: data.symbols,
        k_neighbors: data.kNeighbors,
        noise_level: data.noiseLevel,
        alphabet_size: data.alphabetSize,
      };

      return analysisService.runIDTxlAnalysis({ ...config, time_series_data: timeSeriesData });
    },
  });

  const runMLAnalysis = useMutation({
    mutationFn: async (data: MLFormData) => {
      const config = {
        model_type: data.modelType,
        target: data.target,
        prediction_horizon: data.predictionHorizon,
        validation: data.validation,
        test_size: data.testSize,
        hyperparameters: {
          // Include only relevant hyperparameters based on model type
          ...(data.modelType === 'random_forest' && {
            n_estimators: data.nEstimators,
            max_depth: data.maxDepth,
            min_samples_split: data.minSamplesSplit,
            min_samples_leaf: data.minSamplesLeaf,
          }),
          ...(data.modelType === 'xgboost' && {
            learning_rate: data.learningRate,
            n_boost_rounds: data.nBoostRounds,
            max_depth: data.maxDepthXgb,
            subsample: data.subsample,
            colsample_bytree: data.colsampleBytree,
          }),
          ...(data.modelType === 'svm' && {
            kernel: data.kernel,
            C: data.C,
            gamma: data.gamma,
          }),
        },
        features: {
          technical_indicators: data.technicalIndicators,
          lagged_features: data.laggedFeatures,
          rolling_windows: data.rollingWindows,
        },
      };

      return analysisService.runMLAnalysis(config);
    },
  });

  const runNNAnalysis = useMutation({
    mutationFn: async (data: NNFormData) => {
      const config = {
        architecture: data.architecture,
        epochs: data.epochs,
        batch_size: data.batchSize,
        optimizer: data.optimizer,
        learning_rate: data.learningRate,
        dropout_rate: data.dropoutRate,
        early_stopping_patience: data.earlyStoppingPatience,
        batch_normalization: data.batchNormalization,
        // Architecture-specific settings
        ...((['lstm', 'gru'].includes(data.architecture)) && {
          layers: data.layers,
          bidirectional: data.bidirectional,
          return_sequences: data.returnSequences,
        }),
        ...(data.architecture === 'cnn' && {
          filters: data.filters,
          kernel_size: data.kernelSize,
          pooling_size: data.poolingSize,
        }),
        ...(data.architecture === 'transformer' && {
          attention_heads: data.attentionHeads,
          encoder_layers: data.encoderLayers,
          d_model: data.dModel,
          feedforward_dim: data.feedforwardDim,
        }),
        dense_layers: data.denseLayers,
      };

      return analysisService.runNNAnalysis(config);
    },
  });

  const runIntegratedAnalysis = useMutation({
    mutationFn: async (data: IntegratedFormData) => {
      return analysisService.runIntegratedAnalysis(data);
    },
  });

  // Form submission handlers
  const onIDTxlSubmit = async (data: IDTxlFormData) => {
    data.symbols = selectedSymbols;
    
    // Check if optimization is enabled
    const hasOptimizationEnabled = Object.values(idtxlOptimization).some(opt => opt.enabled);
    
    if (hasOptimizationEnabled) {
      // Submit optimization task
      const optimizationPayload = {
        ...data,
        optimization: {
          enabled: true,
          method: optimizationConfig.method,
          objective: optimizationConfig.objective,
          maxIterations: optimizationConfig.maxIterations,
          crossValidationFolds: optimizationConfig.crossValidationFolds,
          parameterRanges: idtxlOptimization
        }
      };
      await runIDTxlAnalysis.mutateAsync(optimizationPayload);
    } else {
      // Regular single analysis
      await runIDTxlAnalysis.mutateAsync(data);
    }
  };

  const onMLSubmit = async (data: MLFormData) => {
    data.symbols = selectedSymbols;
    
    const hasOptimizationEnabled = Object.values(mlOptimization).some(opt => opt.enabled);
    
    if (hasOptimizationEnabled) {
      const optimizationPayload = {
        ...data,
        optimization: {
          enabled: true,
          method: optimizationConfig.method,
          objective: optimizationConfig.objective,
          maxIterations: optimizationConfig.maxIterations,
          crossValidationFolds: optimizationConfig.crossValidationFolds,
          parameterRanges: mlOptimization
        }
      };
      await runMLAnalysis.mutateAsync(optimizationPayload);
    } else {
      await runMLAnalysis.mutateAsync(data);
    }
  };

  const onNNSubmit = async (data: NNFormData) => {
    data.symbols = selectedSymbols;
    
    const hasOptimizationEnabled = Object.values(nnOptimization).some(opt => opt.enabled);
    
    if (hasOptimizationEnabled) {
      const optimizationPayload = {
        ...data,
        optimization: {
          enabled: true,
          method: optimizationConfig.method,
          objective: optimizationConfig.objective,
          maxIterations: optimizationConfig.maxIterations,
          crossValidationFolds: optimizationConfig.crossValidationFolds,
          parameterRanges: nnOptimization
        }
      };
      await runNNAnalysis.mutateAsync(optimizationPayload);
    } else {
      await runNNAnalysis.mutateAsync(data);
    }
  };

  const onIntegratedSubmit = async (data: IntegratedFormData) => {
    data.symbols = selectedSymbols;
    await runIntegratedAnalysis.mutateAsync(data);
  };

  // Symbol management
  const addSymbol = (symbol: string) => {
    if (!selectedSymbols.includes(symbol) && selectedSymbols.length < 10) {
      setSelectedSymbols([...selectedSymbols, symbol]);
    }
  };

  const removeSymbol = (symbol: string) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
  };

  // Update all forms when symbols change
  useEffect(() => {
    idtxlForm.setValue('symbols', selectedSymbols);
    mlForm.setValue('symbols', selectedSymbols);
    nnForm.setValue('symbols', selectedSymbols);
    integratedForm.setValue('symbols', selectedSymbols);
  }, [selectedSymbols]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Expert Analysis Configuration</h1>
          <p className="text-gray-600">
            {optimizationMode ? 'Configure parameter ranges for optimization' : 'Configure advanced analysis with full parameter control'}
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setOptimizationMode(!optimizationMode)}
            className={`px-4 py-2 rounded-lg ${
              optimizationMode 
                ? 'bg-purple-600 text-white' 
                : 'text-gray-700 bg-white border border-gray-300 hover:bg-gray-50'
            }`}
          >
            <Zap className="w-4 h-4 inline mr-2" />
            {optimizationMode ? 'Optimization Mode' : 'Single Run Mode'}
          </button>
          <button
            onClick={() => setShowAIAdvisor(!showAIAdvisor)}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <Brain className="w-4 h-4 inline mr-2" />
            AI Advisor
          </button>
          <button className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50">
            <Download className="w-4 h-4 inline mr-2" />
            Export Config
          </button>
        </div>
      </div>

      {/* AI Advisor Panel */}
      <AnimatePresence>
        {showAIAdvisor && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-blue-50 border border-blue-200 rounded-lg p-4"
          >
            <div className="flex items-start space-x-3">
              <Brain className="w-5 h-5 text-blue-600 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-blue-900">AI Configuration Assistant</h3>
                <p className="text-sm text-blue-700 mt-1">
                  Based on your selected {activeTab} analysis with {selectedSymbols.join(', ')} symbols,
                  I recommend adjusting your configuration for optimal results. 
                  {activeTab === 'idtxl' && watchedEstimator === 'kraskov' && 
                    ' For Kraskov estimator with financial data, consider using k=5 neighbors for better bias-variance trade-off.'}
                  {activeTab === 'ml' && watchedModelType === 'xgboost' && 
                    ' XGBoost typically performs well with learning_rate=0.05 and max_depth=6 for financial time series.'}
                  {activeTab === 'neural' && watchedArchitecture === 'lstm' && 
                    ' For LSTM on financial data, bidirectional architecture with 2 layers of [128, 64] units often captures temporal patterns effectively.'}
                </p>
                <button
                  onClick={() => {
                    const currentConfig = activeTab === 'idtxl' ? idtxlForm.getValues() :
                                        activeTab === 'ml' ? mlForm.getValues() :
                                        activeTab === 'neural' ? nnForm.getValues() :
                                        integratedForm.getValues();
                    getAIRecommendations.mutate(currentConfig);
                  }}
                  className="mt-2 text-sm text-blue-600 hover:text-blue-700 underline"
                >
                  Get detailed recommendations
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Optimization Summary Panel */}
      <AnimatePresence>
        {hasOptimization() && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-purple-50 border border-purple-200 rounded-lg p-4 mt-4"
          >
            <div className="flex items-start space-x-3">
              <Zap className="w-5 h-5 text-purple-600 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-purple-900">Parameter Optimization Enabled</h3>
                <div className="mt-2 text-sm text-purple-700">
                  <p>
                    Optimizing {Object.entries(
                      activeTab === 'idtxl' ? idtxlOptimization : 
                      activeTab === 'ml' ? mlOptimization : 
                      activeTab === 'neural' ? nnOptimization : {}
                    ).filter(([_, opt]) => opt.enabled).length} parameters
                  </p>
                  
                  {/* Optimization Objective Selector */}
                  <div className="mt-3 space-y-3">
                    <div>
                      <label className="block text-sm font-medium text-purple-900 mb-1">
                        Optimization Objective <span className="text-red-500">*</span>
                      </label>
                      <select
                        value={optimizationConfig.objective}
                        onChange={(e) => setOptimizationConfig(prev => ({
                          ...prev,
                          objective: e.target.value
                        }))}
                        className="w-full px-3 py-2 border border-purple-300 rounded-lg bg-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      >
                        <option value="">Select optimization objective...</option>
                        {OPTIMIZATION_OBJECTIVES[activeTab]?.map(obj => (
                          <option key={obj.value} value={obj.value}>
                            {obj.label}
                          </option>
                        ))}
                      </select>
                      {optimizationConfig.objective && (
                        <p className="mt-1 text-xs text-purple-600">
                          {OPTIMIZATION_OBJECTIVES[activeTab]?.find(obj => obj.value === optimizationConfig.objective)?.description}
                        </p>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-4 text-sm">
                      <span>Method: <strong className="capitalize">{optimizationConfig.method.replace('_', ' ')}</strong></span>
                      <span>Max iterations: <strong>{optimizationConfig.maxIterations}</strong></span>
                      <button
                        onClick={() => {
                          // TODO: Show advanced optimization config modal
                        }}
                        className="text-purple-600 hover:text-purple-700 underline text-xs"
                      >
                        Advanced settings
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analysis Type Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button
              onClick={() => setActiveTab('idtxl')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'idtxl'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Brain className="w-4 h-4 inline mr-2" />
              IDTxl Analysis
            </button>
            <button
              onClick={() => setActiveTab('ml')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'ml'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <BarChart className="w-4 h-4 inline mr-2" />
              Machine Learning
            </button>
            <button
              onClick={() => setActiveTab('neural')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'neural'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Network className="w-4 h-4 inline mr-2" />
              Neural Networks
            </button>
            <button
              onClick={() => setActiveTab('integrated')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'integrated'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Layers className="w-4 h-4 inline mr-2" />
              Integrated Analysis
            </button>
          </nav>
        </div>

        <div className="p-6">
          {/* Common Symbol Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Symbols for Analysis
            </label>
            <div className="space-y-3">
              <div className="flex space-x-2">
                <input
                  type="text"
                  placeholder="Search symbols..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              {/* Search Results */}
              {searchResults && searchResults.length > 0 && (
                <div className="border border-gray-200 rounded-lg max-h-40 overflow-y-auto">
                  {searchResults.map((result) => (
                    <button
                      key={result.symbol}
                      type="button"
                      onClick={() => addSymbol(result.symbol)}
                      className="w-full px-4 py-2 text-left hover:bg-gray-50 flex justify-between items-center"
                    >
                      <div>
                        <span className="font-medium">{result.symbol}</span>
                        <span className="text-gray-500 ml-2">{result.name}</span>
                      </div>
                      <span className="text-xs text-gray-400">{result.exchange}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* Selected Symbols */}
              <div className="flex flex-wrap gap-2">
                {selectedSymbols.map((symbol) => (
                  <span
                    key={symbol}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm flex items-center"
                  >
                    {symbol}
                    <button
                      type="button"
                      onClick={() => removeSymbol(symbol)}
                      className="ml-2 text-blue-500 hover:text-blue-700"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* IDTxl Configuration */}
          {activeTab === 'idtxl' && (
            <form onSubmit={idtxlForm.handleSubmit(onIDTxlSubmit)} className="space-y-6">
              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    {...idtxlForm.register('startDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {idtxlForm.formState.errors.startDate && (
                    <p className="mt-1 text-sm text-red-600">{idtxlForm.formState.errors.startDate.message}</p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    {...idtxlForm.register('endDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {idtxlForm.formState.errors.endDate && (
                    <p className="mt-1 text-sm text-red-600">{idtxlForm.formState.errors.endDate.message}</p>
                  )}
                </div>
              </div>

              {/* Basic Settings */}
              <div className="grid grid-cols-2 gap-4">
                <OptimizableInput
                  name="analysisType"
                  label="Analysis Type"
                  type="select"
                  register={idtxlForm.register}
                  options={[
                    { value: 'transfer_entropy', label: 'Transfer Entropy' },
                    { value: 'mutual_information', label: 'Mutual Information' },
                    { value: 'both', label: 'Both TE & MI' }
                  ]}
                  tooltip="Transfer Entropy captures directional causality, Mutual Information captures correlation"
                  optimization={idtxlOptimization.analysisType}
                  onOptimizationChange={(enabled, config) => {
                    setIdtxlOptimization(prev => ({
                      ...prev,
                      analysisType: enabled ? { enabled, type: 'list', ...config } : undefined
                    }));
                  }}
                />
                <OptimizableInput
                  name="estimator"
                  label="Estimator"
                  type="select"
                  register={idtxlForm.register}
                  options={[
                    { value: 'gaussian', label: 'Gaussian' },
                    { value: 'symbolic', label: 'Symbolic' },
                    { value: 'kraskov', label: 'Kraskov' }
                  ]}
                  tooltip="Kraskov: Non-parametric, best for continuous data. Gaussian: Assumes normal distribution, faster. Symbolic: Discretizes data, good for categorical patterns."
                  optimization={idtxlOptimization.estimator}
                  onOptimizationChange={(enabled, config) => {
                    setIdtxlOptimization(prev => ({
                      ...prev,
                      estimator: enabled ? { enabled, type: 'list', ...config } : undefined
                    }));
                  }}
                />
              </div>

              {/* Advanced Settings */}
              <div className="grid grid-cols-3 gap-4">
                <OptimizableInput
                  name="maxLag"
                  label="Max Lag"
                  type="number"
                  register={idtxlForm.register}
                  min={1}
                  max={20}
                  step={1}
                  tooltip="Maximum time lag to consider for information transfer. Higher values capture longer-term dependencies but increase computational complexity."
                  optimization={idtxlOptimization.maxLag}
                  onOptimizationChange={(enabled, config) => {
                    setIdtxlOptimization(prev => ({
                      ...prev,
                      maxLag: enabled ? { enabled, ...config } : undefined
                    }));
                  }}
                />
                <OptimizableInput
                  name="significanceLevel"
                  label="Significance Level"
                  type="number"
                  register={idtxlForm.register}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                  tooltip="P-value threshold for statistical significance. Lower values (0.01) are more conservative, higher values (0.05) detect more connections."
                  optimization={idtxlOptimization.significanceLevel}
                  onOptimizationChange={(enabled, config) => {
                    setIdtxlOptimization(prev => ({
                      ...prev,
                      significanceLevel: enabled ? { enabled, ...config } : undefined
                    }));
                  }}
                />
                <OptimizableInput
                  name="permutations"
                  label="Permutations"
                  type="number"
                  register={idtxlForm.register}
                  min={50}
                  max={1000}
                  step={50}
                  tooltip="Number of surrogate data permutations for significance testing. More permutations = more reliable p-values but longer computation."
                  optimization={idtxlOptimization.permutations}
                  onOptimizationChange={(enabled, config) => {
                    setIdtxlOptimization(prev => ({
                      ...prev,
                      permutations: enabled ? { enabled, ...config } : undefined
                    }));
                  }}
                />
              </div>

              {/* Estimator-specific settings */}
              <AnimatePresence mode="wait">
                {watchedEstimator === 'kraskov' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">Kraskov Estimator Settings</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          K Neighbors
                          <HelpCircle className="w-4 h-4 inline ml-1 text-gray-400" />
                        </label>
                        <input
                          type="number"
                          {...idtxlForm.register('kNeighbors', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Noise Level
                          <HelpCircle className="w-4 h-4 inline ml-1 text-gray-400" />
                        </label>
                        <input
                          type="number"
                          step="1e-10"
                          {...idtxlForm.register('noiseLevel', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}

                {watchedEstimator === 'symbolic' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">Symbolic Estimator Settings</h4>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Alphabet Size
                        <HelpCircle className="w-4 h-4 inline ml-1 text-gray-400" />
                      </label>
                      <input
                        type="number"
                        {...idtxlForm.register('alphabetSize', { valueAsNumber: true })}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* GPU Acceleration */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Zap className="w-5 h-5 text-amber-500" />
                  <div>
                    <p className="font-medium text-gray-900">GPU Acceleration</p>
                    <p className="text-sm text-gray-500">Use local GPU for faster processing</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...idtxlForm.register('useGpu')}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              {/* Submit Section */}
              <div className="space-y-3">
                {/* Validation Message */}
                {Object.values(idtxlOptimization).some(opt => opt.enabled) && !optimizationConfig.objective && (
                  <div className="flex items-center space-x-2 text-sm text-amber-600 bg-amber-50 p-3 rounded-lg">
                    <AlertCircle className="w-4 h-4" />
                    <span>Please select an optimization objective above before running the optimization</span>
                  </div>
                )}
                
                <div className="flex justify-end space-x-3">
                  <button
                    type="button"
                    className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={
                      runIDTxlAnalysis.isPending || 
                    (Object.values(idtxlOptimization).some(opt => opt.enabled) && !optimizationConfig.objective)
                  }
                  title={
                    Object.values(idtxlOptimization).some(opt => opt.enabled) && !optimizationConfig.objective
                      ? 'Please select an optimization objective'
                      : ''
                  }
                  className={`px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center ${
                    Object.values(idtxlOptimization).some(opt => opt.enabled)
                      ? 'bg-purple-600 hover:bg-purple-700 text-white'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {runIDTxlAnalysis.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running {Object.values(idtxlOptimization).some(opt => opt.enabled) ? 'Optimization' : 'Analysis'}...
                    </>
                  ) : (
                    <>
                      {Object.values(idtxlOptimization).some(opt => opt.enabled) ? (
                        <>
                          <Zap className="w-4 h-4 mr-2" />
                          Run IDTxl Optimization
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          Run IDTxl Analysis
                        </>
                      )}
                    </>
                  )}
                </button>
                </div>
              </div>
            </form>
          )}

          {/* Machine Learning Configuration */}
          {activeTab === 'ml' && (
            <form onSubmit={mlForm.handleSubmit(onMLSubmit)} className="space-y-6">
              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    {...mlForm.register('startDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    {...mlForm.register('endDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Model Configuration */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Model Type
                    <HelpCircle className="w-4 h-4 inline ml-1 text-gray-400" />
                  </label>
                  <select
                    {...mlForm.register('modelType')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="random_forest">Random Forest</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="svm">Support Vector Machine</option>
                    <option value="logistic_regression">Logistic Regression</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prediction Target
                  </label>
                  <select
                    {...mlForm.register('target')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="direction">Price Direction</option>
                    <option value="returns">Returns</option>
                    <option value="volatility">Volatility</option>
                  </select>
                </div>
              </div>

              {/* Training Configuration */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prediction Horizon
                  </label>
                  <input
                    type="number"
                    {...mlForm.register('predictionHorizon', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Validation Strategy
                  </label>
                  <select
                    {...mlForm.register('validation')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="time_series_cv">Time Series CV</option>
                    <option value="walk_forward">Walk Forward</option>
                    <option value="purged_cv">Purged CV</option>
                    <option value="train_test_split">Train/Test Split</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Test Size
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    {...mlForm.register('testSize', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Model-specific Hyperparameters */}
              <AnimatePresence mode="wait">
                {watchedModelType === 'random_forest' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">Random Forest Hyperparameters</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Number of Estimators
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('nEstimators', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Max Depth
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('maxDepth', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Min Samples Split
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('minSamplesSplit', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Min Samples Leaf
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('minSamplesLeaf', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}

                {watchedModelType === 'xgboost' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">XGBoost Hyperparameters</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Learning Rate
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          {...mlForm.register('learningRate', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          N Boost Rounds
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('nBoostRounds', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Max Depth
                        </label>
                        <input
                          type="number"
                          {...mlForm.register('maxDepthXgb', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Subsample
                        </label>
                        <input
                          type="number"
                          step="0.05"
                          {...mlForm.register('subsample', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Column Sample by Tree
                        </label>
                        <input
                          type="number"
                          step="0.05"
                          {...mlForm.register('colsampleBytree', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}

                {watchedModelType === 'svm' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">SVM Hyperparameters</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Kernel
                        </label>
                        <select
                          {...mlForm.register('kernel')}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                          <option value="linear">Linear</option>
                          <option value="rbf">RBF</option>
                          <option value="poly">Polynomial</option>
                          <option value="sigmoid">Sigmoid</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          C (Regularization)
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          {...mlForm.register('C', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      {watchedKernel !== 'linear' && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Gamma
                          </label>
                          <select
                            {...mlForm.register('gamma')}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="scale">Scale</option>
                            <option value="auto">Auto</option>
                            <option value="0.001">0.001</option>
                            <option value="0.01">0.01</option>
                            <option value="0.1">0.1</option>
                          </select>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Feature Engineering */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Feature Engineering</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Technical Indicators
                    </label>
                    <div className="space-y-2">
                      {['RSI', 'MACD', 'Bollinger Bands', 'EMA', 'SMA', 'ATR'].map((indicator) => (
                        <label key={indicator} className="flex items-center">
                          <input
                            type="checkbox"
                            value={indicator.toLowerCase().replace(' ', '_')}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="ml-2 text-sm text-gray-700">{indicator}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Lagged Features
                    </label>
                    <input
                      type="number"
                      {...mlForm.register('laggedFeatures', { valueAsNumber: true })}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">
                      Rolling Windows
                    </label>
                    <input
                      type="text"
                      placeholder="5, 20, 50"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              {/* GPU Acceleration */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Zap className="w-5 h-5 text-amber-500" />
                  <div>
                    <p className="font-medium text-gray-900">GPU Acceleration</p>
                    <p className="text-sm text-gray-500">Use GPU for XGBoost (if available)</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...mlForm.register('useGpu')}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              {/* Submit Button */}
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={runMLAnalysis.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  {runMLAnalysis.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Training Model...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run ML Analysis
                    </>
                  )}
                </button>
              </div>
            </form>
          )}

          {/* Neural Network Configuration */}
          {activeTab === 'neural' && (
            <form onSubmit={nnForm.handleSubmit(onNNSubmit)} className="space-y-6">
              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    {...nnForm.register('startDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    {...nnForm.register('endDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Architecture Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Neural Network Architecture
                </label>
                <div className="grid grid-cols-4 gap-4">
                  {[
                    { value: 'lstm', label: 'LSTM', icon: GitBranch },
                    { value: 'gru', label: 'GRU', icon: Network },
                    { value: 'cnn', label: 'CNN', icon: Layers },
                    { value: 'transformer', label: 'Transformer', icon: Cpu },
                  ].map(({ value, label, icon: Icon }) => (
                    <label
                      key={value}
                      className={`relative flex flex-col items-center p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                        watchedArchitecture === value
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="radio"
                        value={value}
                        {...nnForm.register('architecture')}
                        className="sr-only"
                      />
                      <Icon className={`w-8 h-8 mb-2 ${
                        watchedArchitecture === value ? 'text-blue-600' : 'text-gray-400'
                      }`} />
                      <span className={`text-sm font-medium ${
                        watchedArchitecture === value ? 'text-blue-900' : 'text-gray-700'
                      }`}>
                        {label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Common Training Parameters */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Epochs
                  </label>
                  <input
                    type="number"
                    {...nnForm.register('epochs', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    {...nnForm.register('batchSize', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Optimizer
                  </label>
                  <select
                    {...nnForm.register('optimizer')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="adam">Adam</option>
                    <option value="sgd">SGD</option>
                    <option value="rmsprop">RMSprop</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    {...nnForm.register('learningRate', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Dropout Rate
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    {...nnForm.register('dropoutRate', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Early Stopping Patience
                  </label>
                  <input
                    type="number"
                    {...nnForm.register('earlyStoppingPatience', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Architecture-specific Settings */}
              <AnimatePresence mode="wait">
                {(watchedArchitecture === 'lstm' || watchedArchitecture === 'gru') && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">{watchedArchitecture.toUpperCase()} Settings</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Layer Units (comma-separated)
                        </label>
                        <input
                          type="text"
                          placeholder="128, 64"
                          defaultValue="128, 64"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div className="space-y-3">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            {...nnForm.register('bidirectional')}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="ml-2 text-sm text-gray-700">Bidirectional</span>
                        </label>
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            {...nnForm.register('returnSequences')}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="ml-2 text-sm text-gray-700">Return Sequences</span>
                        </label>
                      </div>
                    </div>
                  </motion.div>
                )}

                {watchedArchitecture === 'cnn' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">CNN Settings</h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Filters (comma-separated)
                        </label>
                        <input
                          type="text"
                          placeholder="64, 128"
                          defaultValue="64, 128"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Kernel Size
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('kernelSize', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Pooling Size
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('poolingSize', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}

                {watchedArchitecture === 'transformer' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <h4 className="font-medium text-gray-900">Transformer Settings</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Attention Heads
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('attentionHeads', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Encoder Layers
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('encoderLayers', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Model Dimension
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('dModel', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Feedforward Dimension
                        </label>
                        <input
                          type="number"
                          {...nnForm.register('feedforwardDim', { valueAsNumber: true })}
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Dense Layers (for all architectures) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Dense Layers (comma-separated)
                </label>
                <input
                  type="text"
                  placeholder="64, 32"
                  defaultValue="64, 32"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Batch Normalization */}
              <div className="flex items-center">
                <input
                  type="checkbox"
                  {...nnForm.register('batchNormalization')}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <label className="ml-2 text-sm text-gray-700">
                  Enable Batch Normalization
                </label>
              </div>

              {/* GPU Acceleration */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Zap className="w-5 h-5 text-amber-500" />
                  <div>
                    <p className="font-medium text-gray-900">GPU Acceleration</p>
                    <p className="text-sm text-gray-500">Highly recommended for neural networks</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...nnForm.register('useGpu')}
                    defaultChecked
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              {/* Submit Button */}
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={runNNAnalysis.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  {runNNAnalysis.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Training Network...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Neural Network
                    </>
                  )}
                </button>
              </div>
            </form>
          )}

          {/* Integrated Analysis Configuration */}
          {activeTab === 'integrated' && (
            <form onSubmit={integratedForm.handleSubmit(onIntegratedSubmit)} className="space-y-6">
              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    {...integratedForm.register('startDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    {...integratedForm.register('endDate')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Analysis Components */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Select Analysis Components</h4>
                
                {/* IDTxl Component */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        {...integratedForm.register('analysisComponents.idtxl.enabled')}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 font-medium">IDTxl Analysis</span>
                    </label>
                    <div className="flex items-center space-x-2">
                      <label className="text-sm text-gray-600">Weight:</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        {...integratedForm.register('analysisComponents.idtxl.weight', { valueAsNumber: true })}
                        className="w-20 px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Information-theoretic analysis for causal relationships
                  </p>
                </div>

                {/* ML Component */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        {...integratedForm.register('analysisComponents.ml.enabled')}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 font-medium">Machine Learning</span>
                    </label>
                    <div className="flex items-center space-x-2">
                      <label className="text-sm text-gray-600">Weight:</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        {...integratedForm.register('analysisComponents.ml.weight', { valueAsNumber: true })}
                        className="w-20 px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Traditional ML models for pattern recognition
                  </p>
                </div>

                {/* NN Component */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        {...integratedForm.register('analysisComponents.nn.enabled')}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 font-medium">Neural Networks</span>
                    </label>
                    <div className="flex items-center space-x-2">
                      <label className="text-sm text-gray-600">Weight:</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        {...integratedForm.register('analysisComponents.nn.weight', { valueAsNumber: true })}
                        className="w-20 px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Deep learning for complex temporal patterns
                  </p>
                </div>
              </div>

              {/* Ensemble Method */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Ensemble Method
                  </label>
                  <select
                    {...integratedForm.register('ensembleMethod')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="weighted_average">Weighted Average</option>
                    <option value="voting">Majority Voting</option>
                    <option value="stacking">Stacking</option>
                    <option value="blending">Blending</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Signal Threshold
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    {...integratedForm.register('signalThreshold', { valueAsNumber: true })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Weight Normalization Info */}
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <div className="flex items-start space-x-2">
                  <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
                  <p className="text-sm text-amber-800">
                    Weights should sum to 1.0 for optimal ensemble performance. Current sum: {
                      (integratedForm.watch('analysisComponents.idtxl.weight') || 0) +
                      (integratedForm.watch('analysisComponents.ml.weight') || 0) +
                      (integratedForm.watch('analysisComponents.nn.weight') || 0)
                    }
                  </p>
                </div>
              </div>

              {/* GPU Acceleration */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Zap className="w-5 h-5 text-amber-500" />
                  <div>
                    <p className="font-medium text-gray-900">GPU Acceleration</p>
                    <p className="text-sm text-gray-500">Recommended for integrated analysis</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...integratedForm.register('useGpu')}
                    defaultChecked
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              {/* Submit Button */}
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={runIntegratedAnalysis.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  {runIntegratedAnalysis.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running Integrated Analysis...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Integrated Analysis
                    </>
                  )}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>

      {/* Results Section */}
      {(runIDTxlAnalysis.isSuccess || runMLAnalysis.isSuccess || runNNAnalysis.isSuccess || runIntegratedAnalysis.isSuccess) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
            <button className="text-blue-600 hover:text-blue-700 text-sm">
              View Full Report
            </button>
          </div>
          
          {/* Results visualization would go here */}
          <div className="bg-gray-50 rounded-lg p-8 text-center">
            <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
            <p className="text-gray-600">Analysis completed successfully</p>
            <p className="text-sm text-gray-500 mt-2">
              Processing time: {
                runIDTxlAnalysis.data?.processing_time ||
                runMLAnalysis.data?.processing_time ||
                runNNAnalysis.data?.processing_time ||
                runIntegratedAnalysis.data?.processing_time
              }s
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Analysis;