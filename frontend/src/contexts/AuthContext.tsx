import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import api from '../services/api';

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  const checkAuth = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setIsLoading(false);
        return;
      }

      // For now, we'll simulate authentication
      // In production, this would validate the token with the backend
      const mockUser: User = {
        id: '1',
        username: 'trader',
        email: 'trader@finplatform.com',
        role: 'admin'
      };

      setUser(mockUser);
    } catch (error) {
      console.error('Auth check failed:', error);
      localStorage.removeItem('access_token');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    try {
      // In production, this would be an actual API call
      // const response = await api.post('/auth/login', { username, password });
      // const { access_token, user } = response.data;

      // Mock authentication for development
      if (username === 'demo' && password === 'demo') {
        const mockToken = 'mock-jwt-token';
        const mockUser: User = {
          id: '1',
          username: 'demo',
          email: 'demo@finplatform.com',
          role: 'admin'
        };

        localStorage.setItem('access_token', mockToken);
        api.defaults.headers.common['Authorization'] = `Bearer ${mockToken}`;
        setUser(mockUser);
        
        toast.success('Login successful!');
        navigate('/dashboard');
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (error: any) {
      toast.error(error.message || 'Login failed');
      throw error;
    }
  }, [navigate]);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    delete api.defaults.headers.common['Authorization'];
    setUser(null);
    navigate('/login');
    toast.success('Logged out successfully');
  }, [navigate]);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  const value = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};