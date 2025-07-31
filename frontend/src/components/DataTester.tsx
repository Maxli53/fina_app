import React, { useState } from 'react';
import { Search, BarChart3, Calendar, Download, Play, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { dataService, analysisService, TimeSeriesData, IDTxlAnalysisResult } from '../services/api';

const DataTester: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [interval, setInterval] = useState('1d');
  
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | null>(null);
  const [analysisResult, setAnalysisResult] = useState<IDTxlAnalysisResult | null>(null);
  
  const [loadingData, setLoadingData] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoadingData(true);
      setError(null);
      setTimeSeriesData(null);
      setAnalysisResult(null);
      
      const data = await dataService.getHistoricalData(symbol, startDate, endDate, interval);
      setTimeSeriesData(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch data');
    } finally {
      setLoadingData(false);
    }
  };

  const runAnalysis = async () => {
    if (!timeSeriesData) {
      setError('Please fetch data first');
      return;
    }

    try {
      setLoadingAnalysis(true);
      setError(null);
      setAnalysisResult(null);
      
      // Create multiple symbols for analysis (use the same data for demo)
      const symbols = [symbol, `${symbol}_LAG1`, `${symbol}_LAG2`];
      const multiSeriesData: { [key: string]: TimeSeriesData } = {};
      
      // Create lagged versions for demonstration
      symbols.forEach((sym, index) => {
        const laggedData = { ...timeSeriesData };
        if (index > 0) {
          // Create artificial lag by shifting data
          laggedData.data = timeSeriesData.data.slice(index).map((item, i) => ({
            ...item,
            // Add some noise to make it interesting
            close: item.close * (1 + (Math.random() - 0.5) * 0.02)
          }));
        }
        multiSeriesData[sym] = laggedData;
      });

      const config = {
        analysis_type: 'both',
        max_lag: 3,
        estimator: 'gaussian',
        significance_level: 0.05,
        permutations: 100,
        variables: symbols,
        time_series_data: multiSeriesData
      };

      const result = await analysisService.runIDTxlAnalysis(config);
      setAnalysisResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to run analysis');
    } finally {
      setLoadingAnalysis(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center mb-6">
        <BarChart3 className="w-5 h-5 mr-2" />
        <h3 className="text-lg font-semibold text-gray-900">Data & Analysis Tester</h3>
      </div>

      {/* Configuration Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
          <div className="relative">
            <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="AAPL"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
          <div className="relative">
            <Calendar className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
          <div className="relative">
            <Calendar className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Interval</label>
          <select
            value={interval}
            onChange={(e) => setInterval(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="1d">Daily (1d)</option>
            <option value="1h">Hourly (1h)</option>
            <option value="5m">5 Minutes</option>
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4 mb-6">
        <button
          onClick={fetchData}
          disabled={loadingData}
          className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loadingData ? (
            <Clock className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Download className="w-4 h-4 mr-2" />
          )}
          {loadingData ? 'Fetching...' : 'Fetch Data'}
        </button>

        <button
          onClick={runAnalysis}
          disabled={!timeSeriesData || loadingAnalysis}
          className="flex items-center px-4 py-2 bg-success-600 text-white rounded-md hover:bg-success-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loadingAnalysis ? (
            <Clock className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Play className="w-4 h-4 mr-2" />
          )}
          {loadingAnalysis ? 'Analyzing...' : 'Run IDTxl Analysis'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-error-50 border border-error-200 rounded-md">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 text-error-600 mr-2" />
            <span className="text-error-800 font-medium">Error</span>
          </div>
          <p className="text-error-700 mt-1">{error}</p>
        </div>
      )}

      {/* Data Results */}
      {timeSeriesData && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center mb-3">
            <CheckCircle className="w-5 h-5 text-success-600 mr-2" />
            <h4 className="font-medium text-gray-900">Data Retrieved Successfully</h4>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Symbol:</span>
              <span className="ml-2 font-medium">{timeSeriesData.symbol}</span>
            </div>
            <div>
              <span className="text-gray-600">Data Points:</span>
              <span className="ml-2 font-medium">{timeSeriesData.data.length}</span>
            </div>
            <div>
              <span className="text-gray-600">Exchange:</span>
              <span className="ml-2 font-medium">{timeSeriesData.metadata.exchange || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-600">Currency:</span>
              <span className="ml-2 font-medium">{timeSeriesData.metadata.currency}</span>
            </div>
          </div>
          
          {timeSeriesData.data.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <span className="text-gray-600 text-sm">Latest Price:</span>
              <span className="ml-2 font-semibold text-lg">
                ${timeSeriesData.data[timeSeriesData.data.length - 1].close.toFixed(2)}
              </span>
              <span className="ml-2 text-gray-500 text-sm">
                on {new Date(timeSeriesData.data[timeSeriesData.data.length - 1].timestamp).toLocaleDateString()}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Analysis Results */}
      {analysisResult && (
        <div className="p-4 bg-success-50 rounded-lg">
          <div className="flex items-center mb-3">
            <CheckCircle className="w-5 h-5 text-success-600 mr-2" />
            <h4 className="font-medium text-gray-900">IDTxl Analysis Complete</h4>
            <span className="ml-auto text-sm text-gray-600">
              Processing time: {analysisResult.processing_time.toFixed(2)}s
            </span>
          </div>
          
          <div className="space-y-4">
            {/* Transfer Entropy Results */}
            {analysisResult.transfer_entropy && (
              <div>
                <h5 className="font-medium text-gray-800 mb-2">Transfer Entropy Connections</h5>
                {analysisResult.transfer_entropy.connections.length > 0 ? (
                  <div className="space-y-2">
                    {analysisResult.transfer_entropy.connections.map((conn, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                        <span className="text-sm">
                          <span className="font-medium">{conn.source}</span> → <span className="font-medium">{conn.target}</span>
                        </span>
                        <div className="text-sm text-gray-600">
                          <span>TE: {conn.te_value.toFixed(4)}</span>
                          <span className="ml-2">Lag: {conn.lag}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-600">No significant transfer entropy connections found.</p>
                )}
              </div>
            )}

            {/* Mutual Information Results */}
            {analysisResult.mutual_information && (
              <div>
                <h5 className="font-medium text-gray-800 mb-2">Mutual Information</h5>
                {analysisResult.mutual_information.significant_pairs.length > 0 ? (
                  <div className="space-y-2">
                    {analysisResult.mutual_information.significant_pairs.map((pair, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                        <span className="text-sm">
                          <span className="font-medium">{pair.var1}</span> ↔ <span className="font-medium">{pair.var2}</span>
                        </span>
                        <span className="text-sm text-gray-600">MI: {pair.mi_value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-600">No significant mutual information pairs found.</p>
                )}
              </div>
            )}

            {/* All Significant Connections */}
            <div>
              <h5 className="font-medium text-gray-800 mb-2">
                All Significant Connections ({analysisResult.significant_connections.length})
              </h5>
              {analysisResult.significant_connections.length > 0 ? (
                <div className="space-y-1">
                  {analysisResult.significant_connections.map((conn, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white rounded border text-sm">
                      <span>
                        <span className="inline-block w-8 h-5 text-xs bg-primary-100 text-primary-700 rounded px-1 mr-2">
                          {conn.type === 'transfer_entropy' ? 'TE' : 'MI'}
                        </span>
                        <span className="font-medium">{conn.source}</span> → <span className="font-medium">{conn.target}</span>
                      </span>
                      <div className="text-gray-600">
                        <span>Strength: {conn.strength.toFixed(4)}</span>
                        {conn.lag > 0 && <span className="ml-2">Lag: {conn.lag}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-600">No significant connections detected.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataTester;