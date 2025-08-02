import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

// Layouts
import MainLayout from './layouts/MainLayout';
import AuthLayout from './layouts/AuthLayout';

// Pages
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Strategies from './pages/Strategies';
import StrategyBuilder from './pages/StrategyBuilder';
import Backtesting from './pages/Backtesting';
import LiveTrading from './pages/LiveTrading';
import RiskMonitor from './pages/RiskMonitor';
import MarketData from './pages/MarketData';
import Portfolio from './pages/Portfolio';
import Settings from './pages/Settings';
import SystemStatus from './pages/SystemStatus';

// Contexts
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

// Guards
import PrivateRoute from './components/guards/PrivateRoute';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Auth Routes */}
            <Route element={<AuthLayout />}>
              <Route path="/login" element={<Login />} />
            </Route>

            {/* Protected Routes */}
            <Route element={<PrivateRoute><WebSocketProvider><MainLayout /></WebSocketProvider></PrivateRoute>}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/strategies" element={<Strategies />} />
              <Route path="/strategies/new" element={<StrategyBuilder />} />
              <Route path="/strategies/:id/edit" element={<StrategyBuilder />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/trading" element={<LiveTrading />} />
              <Route path="/risk" element={<RiskMonitor />} />
              <Route path="/market-data" element={<MarketData />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/system" element={<SystemStatus />} />
            </Route>
          </Routes>
        </Router>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#fff',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#fff',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;