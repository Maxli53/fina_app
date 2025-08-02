"""
System Health Monitoring Service
Provides comprehensive health checks and monitoring for all platform components
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as aioredis
from dataclasses import dataclass
from enum import Enum
import logging
import psutil
import GPUtil
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ServiceType(Enum):
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    MARKET_DATA = "market_data"
    TRADING = "trading"
    ANALYSIS = "analysis"
    WEBSOCKET = "websocket"


@dataclass
class HealthCheckResult:
    service: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any]
    timestamp: datetime
    error: Optional[str] = None


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        config: Dict[str, Any]
    ):
        self.db = db_session
        self.redis = redis_client
        self.config = config
        self.health_history: List[HealthCheckResult] = []
        self.alert_thresholds = {
            "api_latency_ms": 1000,
            "db_latency_ms": 500,
            "cache_latency_ms": 100,
            "cpu_percent": 80,
            "memory_percent": 85,
            "gpu_memory_percent": 90,
            "disk_percent": 90
        }
        
    async def check_all_systems(self) -> Dict[str, HealthCheckResult]:
        """
        Run comprehensive health checks on all systems
        """
        logger.info("Starting comprehensive system health check")
        
        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_api_health(),
            self.check_database_health(),
            self.check_cache_health(),
            self.check_market_data_health(),
            self.check_trading_health(),
            self.check_analysis_health(),
            self.check_infrastructure_health(),
            return_exceptions=True
        )
        
        # Process results
        results = {}
        for check in checks:
            if isinstance(check, Exception):
                logger.error(f"Health check failed: {check}")
                continue
            if isinstance(check, HealthCheckResult):
                results[check.service] = check
                self.health_history.append(check)
        
        # Trim history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        # Calculate overall system health
        overall_health = self._calculate_overall_health(results)
        results["overall"] = overall_health
        
        return results
    
    async def check_api_health(self) -> HealthCheckResult:
        """Check API service health"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['api_base_url']}/api/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            service="api",
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency_ms,
                            details=data,
                            timestamp=datetime.utcnow()
                        )
                    else:
                        return HealthCheckResult(
                            service="api",
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=latency_ms,
                            details={"status_code": response.status},
                            timestamp=datetime.utcnow(),
                            error=f"HTTP {response.status}"
                        )
                        
        except Exception as e:
            return HealthCheckResult(
                service="api",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_database_health(self) -> HealthCheckResult:
        """Check database health"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Test query
            result = await self.db.execute(text("SELECT 1"))
            await self.db.commit()
            
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get connection pool stats
            pool_stats = {
                "active_connections": self.db.bind.pool.size(),
                "idle_connections": self.db.bind.pool.idle(),
                "max_connections": self.db.bind.pool.maxsize
            }
            
            # Check replication lag if applicable
            replication_lag = await self._check_replication_lag()
            
            status = HealthStatus.HEALTHY
            if latency_ms > self.alert_thresholds["db_latency_ms"]:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                service="database",
                status=status,
                latency_ms=latency_ms,
                details={
                    "pool_stats": pool_stats,
                    "replication_lag_seconds": replication_lag
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="database",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_cache_health(self) -> HealthCheckResult:
        """Check Redis cache health"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Ping Redis
            await self.redis.ping()
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get Redis info
            info = await self.redis.info()
            
            details = {
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
            # Calculate hit rate
            total_ops = details["keyspace_hits"] + details["keyspace_misses"]
            if total_ops > 0:
                details["hit_rate"] = details["keyspace_hits"] / total_ops
            
            status = HealthStatus.HEALTHY
            if latency_ms > self.alert_thresholds["cache_latency_ms"]:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                service="cache",
                status=status,
                latency_ms=latency_ms,
                details=details,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="cache",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_market_data_health(self) -> HealthCheckResult:
        """Check market data service health"""
        try:
            # Check last update time for key symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]
            stale_data = []
            
            for symbol in symbols:
                last_update = await self.redis.get(f"market_data:{symbol}:last_update")
                if last_update:
                    last_update_time = datetime.fromisoformat(last_update.decode())
                    age_seconds = (datetime.utcnow() - last_update_time).total_seconds()
                    
                    if age_seconds > 60:  # Data older than 1 minute
                        stale_data.append({
                            "symbol": symbol,
                            "age_seconds": age_seconds
                        })
            
            status = HealthStatus.HEALTHY
            if len(stale_data) > 0:
                status = HealthStatus.DEGRADED
            if len(stale_data) >= len(symbols):
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                service="market_data",
                status=status,
                latency_ms=0,
                details={
                    "stale_symbols": stale_data,
                    "checked_symbols": symbols
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="market_data",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_trading_health(self) -> HealthCheckResult:
        """Check trading service health"""
        try:
            # Check circuit breakers
            circuit_breakers = await self._check_circuit_breakers()
            
            # Check order processing metrics
            order_metrics = await self._get_order_metrics()
            
            # Check risk limits
            risk_status = await self._check_risk_limits()
            
            # Determine overall trading health
            status = HealthStatus.HEALTHY
            if circuit_breakers["triggered"] > 0:
                status = HealthStatus.DEGRADED
            if not risk_status["within_limits"]:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                service="trading",
                status=status,
                latency_ms=0,
                details={
                    "circuit_breakers": circuit_breakers,
                    "order_metrics": order_metrics,
                    "risk_status": risk_status
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="trading",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_analysis_health(self) -> HealthCheckResult:
        """Check analysis service health"""
        try:
            # Check analysis queue depth
            queue_depth = await self.redis.llen("analysis_queue")
            
            # Check GPU status if available
            gpu_status = self._check_gpu_status()
            
            # Check recent analysis completion rate
            completion_rate = await self._get_analysis_completion_rate()
            
            status = HealthStatus.HEALTHY
            if queue_depth > 100:
                status = HealthStatus.DEGRADED
            if queue_depth > 500:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                service="analysis",
                status=status,
                latency_ms=0,
                details={
                    "queue_depth": queue_depth,
                    "gpu_status": gpu_status,
                    "completion_rate": completion_rate
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="analysis",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def check_infrastructure_health(self) -> HealthCheckResult:
        """Check infrastructure health (CPU, memory, disk)"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "network_sent_mb": net_io.bytes_sent / (1024**2),
                "network_recv_mb": net_io.bytes_recv / (1024**2)
            }
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            if (cpu_percent > self.alert_thresholds["cpu_percent"] or
                memory.percent > self.alert_thresholds["memory_percent"] or
                disk.percent > self.alert_thresholds["disk_percent"]):
                status = HealthStatus.DEGRADED
            
            if (cpu_percent > 95 or memory.percent > 95 or disk.percent > 95):
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                service="infrastructure",
                status=status,
                latency_ms=0,
                details=details,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service="infrastructure",
                status=HealthStatus.CRITICAL,
                latency_ms=-1,
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    def _check_gpu_status(self) -> Dict[str, Any]:
        """Check GPU status if available"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    "available": True,
                    "name": gpu.name,
                    "memory_used_percent": gpu.memoryUtil * 100,
                    "gpu_utilization_percent": gpu.load * 100,
                    "temperature_c": gpu.temperature
                }
        except:
            pass
        
        return {"available": False}
    
    async def _check_replication_lag(self) -> Optional[float]:
        """Check database replication lag"""
        try:
            result = await self.db.execute(
                text("SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag")
            )
            row = result.fetchone()
            return row[0] if row else None
        except:
            return None
    
    async def _check_circuit_breakers(self) -> Dict[str, Any]:
        """Check circuit breaker status"""
        breakers = {
            "trading_halt": await self.redis.get("circuit_breaker:trading_halt"),
            "order_limit": await self.redis.get("circuit_breaker:order_limit"),
            "loss_limit": await self.redis.get("circuit_breaker:loss_limit")
        }
        
        triggered = sum(1 for v in breakers.values() if v == b"1")
        
        return {
            "triggered": triggered,
            "breakers": {k: v == b"1" for k, v in breakers.items()}
        }
    
    async def _get_order_metrics(self) -> Dict[str, Any]:
        """Get order processing metrics"""
        # Get from Redis or calculate
        return {
            "orders_per_minute": await self.redis.get("metrics:orders_per_minute") or 0,
            "fill_rate": await self.redis.get("metrics:fill_rate") or 0,
            "rejection_rate": await self.redis.get("metrics:rejection_rate") or 0
        }
    
    async def _check_risk_limits(self) -> Dict[str, bool]:
        """Check if within risk limits"""
        return {
            "within_limits": True,  # Implement actual risk checks
            "var_limit": True,
            "exposure_limit": True,
            "concentration_limit": True
        }
    
    async def _get_analysis_completion_rate(self) -> float:
        """Get analysis task completion rate"""
        completed = await self.redis.get("metrics:analysis_completed") or 0
        total = await self.redis.get("metrics:analysis_total") or 1
        return float(completed) / float(total) if total else 0
    
    def _calculate_overall_health(self, results: Dict[str, HealthCheckResult]) -> HealthCheckResult:
        """Calculate overall system health from individual checks"""
        
        # Count status types
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.CRITICAL: 0
        }
        
        for result in results.values():
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > len(results) / 2:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return HealthCheckResult(
            service="overall",
            status=overall_status,
            latency_ms=0,
            details={
                "service_count": len(results),
                "status_breakdown": {k.value: v for k, v in status_counts.items()}
            },
            timestamp=datetime.utcnow()
        )
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        current_health = await self.check_all_systems()
        
        # Calculate trends from history
        trends = self._calculate_health_trends()
        
        # Get recommendations
        recommendations = self._generate_recommendations(current_health)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_health": {
                service: {
                    "status": result.status.value,
                    "latency_ms": result.latency_ms,
                    "details": result.details,
                    "error": result.error
                }
                for service, result in current_health.items()
            },
            "trends": trends,
            "recommendations": recommendations
        }
    
    def _calculate_health_trends(self) -> Dict[str, Any]:
        """Calculate health trends from history"""
        if not self.health_history:
            return {}
        
        # Group by service
        service_history = {}
        for check in self.health_history:
            if check.service not in service_history:
                service_history[check.service] = []
            service_history[check.service].append(check)
        
        # Calculate trends
        trends = {}
        for service, history in service_history.items():
            # Get last hour of data
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_history = [h for h in history if h.timestamp > one_hour_ago]
            
            if recent_history:
                healthy_count = sum(1 for h in recent_history if h.status == HealthStatus.HEALTHY)
                uptime_percent = (healthy_count / len(recent_history)) * 100
                avg_latency = sum(h.latency_ms for h in recent_history if h.latency_ms > 0) / len(recent_history)
                
                trends[service] = {
                    "uptime_percent": uptime_percent,
                    "avg_latency_ms": avg_latency,
                    "check_count": len(recent_history)
                }
        
        return trends
    
    def _generate_recommendations(self, health_results: Dict[str, HealthCheckResult]) -> List[str]:
        """Generate recommendations based on health status"""
        recommendations = []
        
        for service, result in health_results.items():
            if result.status == HealthStatus.CRITICAL:
                recommendations.append(f"CRITICAL: {service} requires immediate attention")
            elif result.status == HealthStatus.UNHEALTHY:
                recommendations.append(f"WARNING: {service} is unhealthy and needs investigation")
            elif result.status == HealthStatus.DEGRADED:
                if result.latency_ms > 0:
                    recommendations.append(f"Monitor {service} - high latency detected ({result.latency_ms}ms)")
        
        return recommendations