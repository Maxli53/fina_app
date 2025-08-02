import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  TrendingUp,
  Home,
  Brain,
  Target,
  LineChart,
  Activity,
  Shield,
  BarChart3,
  Briefcase,
  Settings,
  LogOut,
  Menu,
  X,
  Bell,
  Circle,
  Server
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useWebSocket } from '../contexts/WebSocketContext';

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
}

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'Analysis', href: '/analysis', icon: Brain },
  { name: 'Strategies', href: '/strategies', icon: Target },
  { name: 'Backtesting', href: '/backtesting', icon: LineChart },
  { name: 'Live Trading', href: '/trading', icon: Activity },
  { name: 'Risk Monitor', href: '/risk', icon: Shield },
  { name: 'Market Data', href: '/market-data', icon: BarChart3 },
  { name: 'Portfolio', href: '/portfolio', icon: Briefcase },
  { name: 'System Status', href: '/system', icon: Server },
];

const MainLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const { user, logout } = useAuth();
  const { isConnected } = useWebSocket();

  const isActive = (href: string) => location.pathname === href;

  return (
    <div className="h-screen flex overflow-hidden bg-gray-100">
      {/* Mobile sidebar */}
      <div className={`${sidebarOpen ? 'block' : 'hidden'} fixed inset-0 z-40 md:hidden`}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 flex flex-col w-64 bg-gray-900">
          <div className="flex items-center justify-between h-16 px-4 bg-gray-800">
            <div className="flex items-center">
              <TrendingUp className="w-8 h-8 text-blue-500" />
              <span className="ml-2 text-white font-semibold">FinPlatform</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} className="text-gray-400 hover:text-white">
              <X className="w-6 h-6" />
            </button>
          </div>
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`${
                  isActive(item.href)
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                } group flex items-center px-2 py-2 text-sm font-medium rounded-md`}
                onClick={() => setSidebarOpen(false)}
              >
                <item.icon className="mr-3 h-5 w-5" />
                {item.name}
              </Link>
            ))}
          </nav>
          <div className="p-4 border-t border-gray-700">
            <Link
              to="/settings"
              className="text-gray-300 hover:bg-gray-700 hover:text-white group flex items-center px-2 py-2 text-sm font-medium rounded-md"
            >
              <Settings className="mr-3 h-5 w-5" />
              Settings
            </Link>
            <button
              onClick={logout}
              className="w-full text-gray-300 hover:bg-gray-700 hover:text-white group flex items-center px-2 py-2 text-sm font-medium rounded-md mt-1"
            >
              <LogOut className="mr-3 h-5 w-5" />
              Logout
            </button>
          </div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden md:flex md:flex-shrink-0">
        <div className="flex flex-col w-64">
          <div className="flex flex-col h-0 flex-1 bg-gray-900">
            <div className="flex items-center h-16 px-4 bg-gray-800">
              <TrendingUp className="w-8 h-8 text-blue-500" />
              <span className="ml-2 text-white font-semibold text-lg">FinPlatform</span>
            </div>
            <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`${
                    isActive(item.href)
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  } group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors`}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              ))}
            </nav>
            <div className="p-4 border-t border-gray-700">
              <Link
                to="/settings"
                className="text-gray-300 hover:bg-gray-700 hover:text-white group flex items-center px-2 py-2 text-sm font-medium rounded-md"
              >
                <Settings className="mr-3 h-5 w-5" />
                Settings
              </Link>
              <button
                onClick={logout}
                className="w-full text-gray-300 hover:bg-gray-700 hover:text-white group flex items-center px-2 py-2 text-sm font-medium rounded-md mt-1"
              >
                <LogOut className="mr-3 h-5 w-5" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col w-0 flex-1 overflow-hidden">
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="md:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
                >
                  <Menu className="h-6 w-6" />
                </button>
                <h1 className="ml-2 md:ml-0 text-xl font-semibold text-gray-900">
                  {navigation.find(item => isActive(item.href))?.name || 'Financial Analysis Platform'}
                </h1>
              </div>
              
              <div className="flex items-center space-x-4">
                {/* WebSocket Connection Status */}
                <div className="flex items-center space-x-2 text-sm">
                  <Circle 
                    className={`h-2 w-2 ${isConnected ? 'text-green-500' : 'text-red-500'} fill-current`} 
                  />
                  <span className="text-gray-600">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>

                {/* Notifications */}
                <button className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 relative">
                  <Bell className="h-5 w-5" />
                  <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
                </button>

                {/* User Menu */}
                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{user?.username || 'User'}</p>
                    <p className="text-xs text-gray-500">{user?.role || 'Trader'}</p>
                  </div>
                  <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                    <span className="text-gray-600 font-medium">
                      {user?.username?.charAt(0).toUpperCase() || 'U'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;