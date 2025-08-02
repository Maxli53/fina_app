"""
System management models
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class SystemState(str, Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HealthStatus(str, Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ServiceHealth(BaseModel):
    """Individual service health status"""
    status: HealthStatus
    latency_ms: float
    details: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)


class WorkflowRequest(BaseModel):
    """Request to execute a workflow"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    timeout_seconds: int = 300


class WorkflowResult(BaseModel):
    """Result of workflow execution"""
    status: str
    results: Dict[str, Any]
    duration: float
    error: Optional[str] = None


class SystemEvent(BaseModel):
    """System event model"""
    timestamp: datetime
    event_type: str
    source: str
    severity: str = "info"
    data: Dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System performance metrics"""
    active_workflows: int
    websocket_clients: int
    event_history_size: int
    uptime: str
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None


class SystemStatusResponse(BaseModel):
    """System status response"""
    state: SystemState
    uptime: str
    active_workflows: List[str]
    metrics: Dict[str, Any]
    connected_clients: int
    event_count: int
    last_events: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Dict[str, Any]]


class RiskMetrics(BaseModel):
    """Risk monitoring metrics"""
    portfolio_value: float
    var_95: float = Field(alias="var_95")
    var_99: float = Field(alias="var_99")
    expected_shortfall: float
    high_volatility: bool = False
    risk_violations: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    """Workflow definition"""
    name: str
    type: str
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: int = 300
    retries: int = 3


class SystemConfiguration(BaseModel):
    """System configuration model"""
    use_gpu: bool = True
    health_check_interval: int = 30
    risk_check_interval: int = 60
    data_pipeline_interval: int = 1
    event_history_limit: int = 1000
    websocket_port: int = 8765
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    emergency_close_positions: bool = False