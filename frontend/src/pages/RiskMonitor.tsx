import React from 'react';
import { Shield } from 'lucide-react';

const RiskMonitor: React.FC = () => {
  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-16">
          <Shield className="w-20 h-20 text-gray-300 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Risk Monitor</h1>
          <p className="text-gray-600 mb-8">
            Real-time risk monitoring and management
          </p>
          <p className="text-sm text-gray-500">
            This feature is coming soon. Monitor portfolio risk metrics, exposure limits, and get real-time alerts.
          </p>
        </div>
      </div>
    </div>
  );
};

export default RiskMonitor;