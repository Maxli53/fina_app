import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

// API Response Types
export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  services: {
    database: string;
    cache: string;
    analysis_engine: string;
  };
}

export interface SymbolSearchResult {
  symbol: string;
  name: string;
  exchange: string;
  type: string;
  currency: string;
}

export interface OHLCVData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjusted_close?: number;
}

export interface TimeSeriesData {
  symbol: string;
  interval: string;
  data: OHLCVData[];
  metadata: {
    currency: string;
    exchange: string;
    timezone: string;
    [key: string]: any;
  };
}

export interface IDTxlAnalysisResult {
  transfer_entropy?: {
    connections: Array<{
      source: string;
      target: string;
      te_value: number;
      lag: number;
    }>;
  };
  mutual_information?: {
    matrix: number[][];
    variables: string[];
    significant_pairs: Array<{
      var1: string;
      var2: string;
      mi_value: number;
    }>;
  };
  significant_connections: Array<{
    type: string;
    source: string;
    target: string;
    strength: number;
    lag: number;
  }>;
  processing_time: number;
}

// API Services
// API Services
export const healthService = {
  getHealth: (): Promise<HealthResponse> =>
    api.get('/api/health').then(res => res.data),
  
  getDetailedHealth: (): Promise<any> =>
    api.get('/api/health/detailed').then(res => res.data),
};

export const dataService = {
  searchSymbols: (query: string, limit: number = 10): Promise<SymbolSearchResult[]> =>
    api.get(`/api/data/search?query=${query}&limit=${limit}`).then(res => res.data),
  
  getHistoricalData: (
    symbol: string, 
    startDate: string, 
    endDate: string, 
    interval: string = '1d'
  ): Promise<TimeSeriesData> =>
    api.get(`/api/data/historical/${symbol}?start_date=${startDate}&end_date=${endDate}&interval=${interval}`)
       .then(res => res.data),
  
  getDataQuality: (
    symbol: string, 
    startDate: string, 
    endDate: string
  ): Promise<any> =>
    api.get(`/api/data/quality/${symbol}?start_date=${startDate}&end_date=${endDate}`)
       .then(res => res.data),
  
  getMarketStatus: (): Promise<any> =>
    api.get('/api/data/market-status').then(res => res.data),
};

export const analysisService = {
  runIDTxlAnalysis: (config: {
    analysis_type: string;
    max_lag: number;
    estimator: string;
    significance_level: number;
    permutations: number;
    variables: string[];
    time_series_data: { [symbol: string]: TimeSeriesData };
  }): Promise<IDTxlAnalysisResult> =>
    api.post('/api/analysis/idtxl', config).then(res => res.data),
  
  getAnalysisHistory: (): Promise<any[]> =>
    api.get('/api/analysis/history').then(res => res.data),
};