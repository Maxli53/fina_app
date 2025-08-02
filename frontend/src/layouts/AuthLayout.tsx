import React from 'react';
import { Outlet } from 'react-router-dom';
import { TrendingUp } from 'lucide-react';

const AuthLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center p-3 bg-blue-600 rounded-xl mb-4">
            <TrendingUp className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">Financial Analysis Platform</h1>
          <p className="text-gray-400 mt-2">Advanced Quantitative Trading System</p>
        </div>
        <Outlet />
      </div>
    </div>
  );
};

export default AuthLayout;