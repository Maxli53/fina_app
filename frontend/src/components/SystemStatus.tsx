import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  HardDrive,
  Network,
  Play,
  Pause,
  RefreshCw,
  Server,
  TrendingUp,
  Zap,
  XCircle
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { formatDistanceToNow } from 'date-fns';

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'critical';
  latency_ms: number;
  details?: string;
}

interface SystemHealth {
  overall: HealthStatus;
  database: HealthStatus;
  cache: HealthStatus;
  market_data: HealthStatus;
  trading: HealthStatus;
  analysis_engine: HealthStatus;
}

interface SystemMetrics {
  active_workflows: number;
  websocket_clients: number;
  event_history_size: number;
  uptime: string;
}

interface SystemEvent {
  timestamp: string;
  event_type: string;
  source: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  data?: any;
}

interface RiskMetrics {
  portfolio_value: number;
  var_95: number;
  var_99: number;
  expected_shortfall: number;
  high_volatility: boolean;
}

export const SystemStatus: React.FC = () => {
  const [systemState, setSystemState] = useState<string>('initializing');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [activeWorkflows, setActiveWorkflows] = useState<string[]>([]);

  const { socket, isConnected } = useWebSocket('ws://localhost:8765');

  useEffect(() => {
    if (!socket) return;

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'system_state':
          setSystemState(data.data.state);
          setActiveWorkflows(data.data.active_workflows || []);
          setMetrics(data.data.metrics);
          break;

        case 'health_update':
          setHealth(data.data.status);
          break;

        case 'system_event':
          setEvents(prev => [data.data, ...prev].slice(0, 50)); // Keep last 50 events
          break;

        case 'risk_update':
          setRiskMetrics(data.data);
          break;

        case 'status_update':
          // Handle comprehensive status update
          if (data.data) {
            setSystemState(data.data.state);
            setMetrics(data.data);
            setActiveWorkflows(data.data.active_workflows || []);
          }
          break;
      }
    };

    // Request initial status
    if (isConnected) {
      socket.send(JSON.stringify({
        type: 'get_status'
      }));
    }
  }, [socket, isConnected]);

  const getStateColor = (state: string) => {
    switch (state) {
      case 'running':
        return 'text-green-600';
      case 'ready':
        return 'text-blue-600';
      case 'degraded':
        return 'text-yellow-600';
      case 'error':
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="h-5 w-5 text-yellow-500" />;
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const executeWorkflow = (workflowType: string) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({
        type: 'execute_workflow',
        workflow: workflowType,
        params: {}
      }));
    }
  };

  const getEventIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* System Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Server className="h-6 w-6" />
              System Status
            </span>
            <div className="flex items-center gap-4">
              <Badge className={`${getStateColor(systemState)} bg-opacity-10`}>
                {systemState.toUpperCase()}
              </Badge>
              {isConnected ? (
                <Badge className="bg-green-100 text-green-800">Connected</Badge>
              ) : (
                <Badge className="bg-red-100 text-red-800">Disconnected</Badge>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="text-sm text-gray-500">Uptime</div>
              <div className="text-2xl font-semibold">{metrics?.uptime || '---'}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-500">Active Workflows</div>
              <div className="text-2xl font-semibold">{metrics?.active_workflows || 0}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-500">Connected Clients</div>
              <div className="text-2xl font-semibold">{metrics?.websocket_clients || 0}</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-500">Event History</div>
              <div className="text-2xl font-semibold">{metrics?.event_history_size || 0}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs for different views */}
      <Tabs defaultValue="health" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="health">Health Monitor</TabsTrigger>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="risk">Risk Monitor</TabsTrigger>
          <TabsTrigger value="events">Event Log</TabsTrigger>
        </TabsList>

        {/* Health Monitor Tab */}
        <TabsContent value="health" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {health && Object.entries(health).map(([service, status]) => (
              <Card key={service}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center justify-between">
                    <span className="capitalize">{service.replace('_', ' ')}</span>
                    {getHealthIcon(status.status)}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Status</span>
                      <Badge variant={status.status === 'healthy' ? 'default' : 'destructive'}>
                        {status.status}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Latency</span>
                      <span className="font-medium">{status.latency_ms}ms</span>
                    </div>
                    {status.details && (
                      <div className="text-xs text-gray-600 mt-2">
                        {status.details}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Workflows Tab */}
        <TabsContent value="workflows" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Automated Workflows</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button
                  onClick={() => executeWorkflow('analysis_to_trade')}
                  className="justify-start"
                  variant="outline"
                >
                  <Zap className="mr-2 h-4 w-4" />
                  Analysis to Trade
                </Button>
                <Button
                  onClick={() => executeWorkflow('portfolio_rebalance')}
                  className="justify-start"
                  variant="outline"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Portfolio Rebalance
                </Button>
                <Button
                  onClick={() => executeWorkflow('risk_monitoring')}
                  className="justify-start"
                  variant="outline"
                >
                  <AlertCircle className="mr-2 h-4 w-4" />
                  Risk Monitoring
                </Button>
                <Button
                  onClick={() => executeWorkflow('strategy_optimization')}
                  className="justify-start"
                  variant="outline"
                >
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Strategy Optimization
                </Button>
              </div>

              {activeWorkflows.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-sm font-medium mb-3">Active Workflows</h4>
                  <div className="space-y-2">
                    {activeWorkflows.map((workflow) => (
                      <div key={workflow} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                        <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
                        <span className="text-sm">{workflow}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Risk Monitor Tab */}
        <TabsContent value="risk" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Risk Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              {riskMetrics ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">Portfolio Value</div>
                      <div className="text-2xl font-semibold">
                        ${riskMetrics.portfolio_value.toLocaleString()}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">VaR (95%)</div>
                      <div className="text-2xl font-semibold text-red-600">
                        ${riskMetrics.var_95.toLocaleString()}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">VaR (99%)</div>
                      <div className="text-2xl font-semibold text-red-600">
                        ${riskMetrics.var_99.toLocaleString()}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">Expected Shortfall</div>
                      <div className="text-2xl font-semibold text-red-600">
                        ${riskMetrics.expected_shortfall.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  {riskMetrics.high_volatility && (
                    <Alert className="bg-yellow-50 border-yellow-200">
                      <AlertCircle className="h-4 w-4 text-yellow-600" />
                      <AlertDescription className="text-yellow-800">
                        High volatility detected in portfolio. Consider risk reduction measures.
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No risk metrics available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Event Log Tab */}
        <TabsContent value="events" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Events</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {events.length > 0 ? (
                  events.map((event, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg"
                    >
                      {getEventIcon(event.severity)}
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{event.event_type}</span>
                          <span className="text-xs text-gray-500">
                            {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 mt-1">
                          Source: {event.source}
                        </div>
                        {event.data && (
                          <div className="text-xs text-gray-600 mt-1">
                            {JSON.stringify(event.data, null, 2)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No events recorded
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};