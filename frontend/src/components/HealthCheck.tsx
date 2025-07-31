import React, { useState, useEffect } from 'react';
import { Activity, Server, Database, Zap, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';
import { healthService, HealthResponse } from '../services/api';

interface HealthCheckProps {
  className?: string;
}

const HealthCheck: React.FC<HealthCheckProps> = ({ className = '' }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await healthService.getHealth();
      setHealth(data);
      setLastUpdate(new Date());
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch health status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'connected':
        return 'text-success-600 bg-success-50';
      case 'warning':
        return 'text-warning-600 bg-warning-50';
      case 'error':
      case 'disconnected':
      case 'unhealthy':
        return 'text-error-600 bg-error-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'connected':
        return <CheckCircle className="w-4 h-4" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4" />;
      case 'error':
      case 'disconnected':
      case 'unhealthy':
        return <AlertCircle className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  if (loading && !health) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            System Health
          </h3>
          <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />
        </div>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded mb-3"></div>
          <div className="h-4 bg-gray-200 rounded mb-3"></div>
          <div className="h-4 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            System Health
          </h3>
          <button
            onClick={fetchHealth}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        <div className="text-center py-8">
          <AlertCircle className="w-12 h-12 text-error-500 mx-auto mb-4" />
          <p className="text-error-600 font-medium mb-2">Health Check Failed</p>
          <p className="text-gray-600 text-sm mb-4">{error}</p>
          <button
            onClick={fetchHealth}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          System Health
        </h3>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">
            Updated {lastUpdate.toLocaleTimeString()}
          </span>
          <button
            onClick={fetchHealth}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            title="Refresh"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {health && (
        <div className="space-y-4">
          {/* Overall Status */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <Server className="w-5 h-5 text-gray-600 mr-3" />
              <span className="font-medium text-gray-900">API Status</span>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getStatusColor(health.status)}`}>
              {getStatusIcon(health.status)}
              <span className="ml-1 capitalize">{health.status}</span>
            </span>
          </div>

          {/* Service Status */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-gray-700">Services</h4>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <div className="flex items-center">
                  <Database className="w-4 h-4 text-gray-600 mr-2" />
                  <span className="text-sm text-gray-700">Database</span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center ${getStatusColor(health.services.database)}`}>
                  {getStatusIcon(health.services.database)}
                  <span className="ml-1 capitalize">{health.services.database}</span>
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <div className="flex items-center">
                  <Zap className="w-4 h-4 text-gray-600 mr-2" />
                  <span className="text-sm text-gray-700">Cache</span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center ${getStatusColor(health.services.cache)}`}>
                  {getStatusIcon(health.services.cache)}
                  <span className="ml-1 capitalize">{health.services.cache}</span>
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                <div className="flex items-center">
                  <Activity className="w-4 h-4 text-gray-600 mr-2" />
                  <span className="text-sm text-gray-700">Analysis</span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center ${getStatusColor(health.services.analysis_engine)}`}>
                  {getStatusIcon(health.services.analysis_engine)}
                  <span className="ml-1 capitalize">{health.services.analysis_engine}</span>
                </span>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Version: {health.version}</span>
              <span>Uptime: {new Date(health.timestamp).toLocaleString()}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HealthCheck;