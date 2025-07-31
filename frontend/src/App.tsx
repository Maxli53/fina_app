import React from 'react';
import { TrendingUp, Activity, BarChart3 } from 'lucide-react';
import HealthCheck from './components/HealthCheck';
import DataTester from './components/DataTester';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <div className="p-2 bg-primary-600 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">
                    Financial Time Series Analysis Platform
                  </h1>
                  <p className="text-sm text-gray-600">
                    IDTxl • Machine Learning • Neural Networks
                  </p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-700">Development Environment</p>
                <p className="text-xs text-gray-500">Backend Integration Test</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Welcome Section */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center mb-4">
              <Activity className="w-5 h-5 mr-2 text-primary-600" />
              <h2 className="text-lg font-semibold text-gray-900">Welcome to the Platform</h2>
            </div>
            <p className="text-gray-600 mb-4">
              This is a development interface for testing the Financial Time Series Analysis Platform. 
              The platform integrates advanced information-theoretic analysis using IDTxl with traditional 
              machine learning and neural network approaches.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <BarChart3 className="w-5 h-5 text-blue-600 mr-2" />
                  <h3 className="font-medium text-blue-900">IDTxl Analysis</h3>
                </div>
                <p className="text-sm text-blue-700">
                  Information-theoretic analysis for detecting causal relationships in financial time series.
                </p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <Activity className="w-5 h-5 text-green-600 mr-2" />
                  <h3 className="font-medium text-green-900">Real-time Data</h3>
                </div>
                <p className="text-sm text-green-700">
                  Live market data integration with proper timezone handling and quality assessment.
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <TrendingUp className="w-5 h-5 text-purple-600 mr-2" />
                  <h3 className="font-medium text-purple-900">Advanced Analytics</h3>
                </div>
                <p className="text-sm text-purple-700">
                  Machine learning and neural network frameworks for predictive modeling.
                </p>
              </div>
            </div>
          </div>

          {/* Health Check Component */}
          <HealthCheck />

          {/* Data Tester Component */}
          <DataTester />

          {/* API Documentation Link */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">API Documentation</h3>
            <p className="text-gray-600 mb-4">
              Explore the full API documentation to understand all available endpoints and their parameters.
            </p>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Open API Documentation
            </a>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              Financial Time Series Analysis Platform - Development Environment
            </p>
            <p className="text-xs mt-1">
              Phase 1 (Core Analysis Engine) Complete • Phase 2 (Strategy Framework) In Progress
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;