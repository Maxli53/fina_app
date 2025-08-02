import React from 'react';
import { LineChart } from 'lucide-react';

const Backtesting: React.FC = () => {
  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-16">
          <LineChart className="w-20 h-20 text-gray-300 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Backtesting Engine</h1>
          <p className="text-gray-600 mb-8">
            Test your trading strategies against historical data
          </p>
          <p className="text-sm text-gray-500">
            This feature is coming soon. You'll be able to backtest strategies with comprehensive metrics and visualizations.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Backtesting;