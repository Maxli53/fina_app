# Performance Optimization Guide

## Overview

This guide covers performance optimization strategies for the Financial Time Series Analysis Platform, including backend optimization, frontend performance, database tuning, and infrastructure scaling.

## Performance Architecture

### System Performance Goals

```yaml
Performance Targets:
  API Response Time:
    - p50: < 100ms
    - p95: < 500ms
    - p99: < 1000ms
  
  Analysis Processing:
    - IDTxl (1000 points): < 15s (GPU)
    - ML Training (10k samples): < 2 min (GPU)
    - NN Training (50k sequences): < 10 min (GPU)
  
  Real-time Updates:
    - Market Data Latency: < 50ms
    - WebSocket Broadcast: < 10ms
    - Order Execution: < 100ms
  
  Throughput:
    - API Requests: 10,000 req/s
    - WebSocket Messages: 100,000 msg/s
    - Analysis Jobs: 1,000 concurrent
```

## Backend Optimization

### Async Performance

```python
# backend/app/services/async_optimization.py
import asyncio
from typing import List, Dict, Any
import aiohttp
from functools import lru_cache
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer

class AsyncOptimizedService:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent operations
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    # Batch processing for efficiency
    async def batch_fetch_data(
        self, 
        symbols: List[str], 
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Fetch data in batches to avoid overwhelming the API"""
        results = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self._fetch_with_semaphore(symbol) 
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            for symbol, result in zip(batch, batch_results):
                results[symbol] = result
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _fetch_with_semaphore(self, symbol: str) -> Dict:
        """Fetch with semaphore to limit concurrent requests"""
        async with self.semaphore:
            return await self._fetch_symbol_data(symbol)
    
    # Caching for frequently accessed data
    @cached(
        ttl=300,  # 5 minutes
        cache=Cache.REDIS,
        serializer=JsonSerializer(),
        key_builder=lambda f, *args, **kwargs: f"{f.__name__}:{args[1]}"  # Use symbol as key
    )
    async def _fetch_symbol_data(self, symbol: str) -> Dict:
        """Fetch and cache symbol data"""
        async with self.session.get(f"/api/quote/{symbol}") as response:
            return await response.json()
    
    # Connection pooling for database
    async def execute_batch_queries(
        self, 
        queries: List[str], 
        pool
    ) -> List[Any]:
        """Execute multiple queries efficiently using connection pool"""
        async with pool.acquire() as connection:
            # Use prepared statements for better performance
            results = []
            
            async with connection.transaction():
                for query in queries:
                    stmt = await connection.prepare(query)
                    result = await stmt.fetch()
                    results.append(result)
            
            return results
```

### GPU Acceleration

```python
# backend/app/services/gpu_acceleration.py
import numpy as np
import cupy as cp  # GPU arrays
import torch
import tensorflow as tf
from typing import Union, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

class GPUAccelerator:
    def __init__(self):
        self.gpu_available = self._check_gpu()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        cuda_available = torch.cuda.is_available()
        tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        return cuda_available or tf_gpu
    
    async def accelerate_computation(
        self, 
        data: np.ndarray, 
        operation: str
    ) -> np.ndarray:
        """Accelerate computation using GPU"""
        if not self.gpu_available:
            return await self._cpu_fallback(data, operation)
        
        # Run GPU computation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._gpu_compute,
            data,
            operation
        )
        
        return result
    
    def _gpu_compute(self, data: np.ndarray, operation: str) -> np.ndarray:
        """Perform GPU computation"""
        # Transfer to GPU
        gpu_data = cp.asarray(data)
        
        if operation == "correlation":
            result = self._gpu_correlation(gpu_data)
        elif operation == "moving_average":
            result = self._gpu_moving_average(gpu_data)
        elif operation == "matrix_multiply":
            result = self._gpu_matrix_multiply(gpu_data)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Transfer back to CPU
        return cp.asnumpy(result)
    
    def _gpu_correlation(self, data: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated correlation calculation"""
        # Normalize data
        data_norm = (data - cp.mean(data, axis=0)) / cp.std(data, axis=0)
        
        # Compute correlation matrix
        correlation = cp.dot(data_norm.T, data_norm) / data_norm.shape[0]
        
        return correlation
    
    def _gpu_moving_average(
        self, 
        data: cp.ndarray, 
        window: int = 20
    ) -> cp.ndarray:
        """GPU-accelerated moving average"""
        kernel = cp.ones(window) / window
        
        # Use convolution for moving average
        result = cp.convolve(data, kernel, mode='valid')
        
        return result
    
    async def batch_gpu_process(
        self, 
        datasets: List[np.ndarray], 
        operation: str
    ) -> List[np.ndarray]:
        """Process multiple datasets on GPU in parallel"""
        if not self.gpu_available:
            return [await self._cpu_fallback(d, operation) for d in datasets]
        
        # Process on multiple GPU streams
        streams = [cp.cuda.Stream() for _ in range(min(len(datasets), 4))]
        results = []
        
        for i, data in enumerate(datasets):
            stream = streams[i % len(streams)]
            
            with stream:
                gpu_data = cp.asarray(data)
                result = self._gpu_compute(gpu_data, operation)
                results.append(result)
        
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        
        return results
```

### Memory Management

```python
# backend/app/services/memory_optimization.py
import gc
import psutil
import asyncio
from typing import Any, Callable
from functools import wraps
import weakref
from memory_profiler import profile

class MemoryOptimizer:
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.large_objects = weakref.WeakValueDictionary()
        
    def memory_efficient(self, func: Callable) -> Callable:
        """Decorator for memory-efficient function execution"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check memory before execution
            await self.check_memory_pressure()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                # Cleanup after execution
                gc.collect()
                
        return wrapper
    
    async def check_memory_pressure(self) -> None:
        """Check and handle memory pressure"""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.max_memory_percent:
            # Trigger aggressive garbage collection
            gc.collect(2)  # Full collection
            
            # Clear caches
            await self.clear_caches()
            
            # If still high, wait
            if psutil.virtual_memory().percent > self.max_memory_percent:
                await asyncio.sleep(1)
    
    async def clear_caches(self) -> None:
        """Clear various caches to free memory"""
        # Clear LRU caches
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear'):
                obj.cache_clear()
    
    def track_large_object(self, name: str, obj: Any) -> None:
        """Track large objects for memory management"""
        self.large_objects[name] = obj
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "large_objects": len(self.large_objects),
            "gc_stats": gc.get_stats()
        }

# Memory-efficient data processing
class DataProcessor:
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.memory_optimizer = MemoryOptimizer()
    
    async def process_large_dataset(
        self, 
        data_generator: AsyncGenerator,
        process_func: Callable
    ) -> AsyncGenerator:
        """Process large dataset in chunks"""
        chunk = []
        
        async for item in data_generator:
            chunk.append(item)
            
            if len(chunk) >= self.chunk_size:
                # Process chunk
                processed = await process_func(chunk)
                
                # Yield results
                for result in processed:
                    yield result
                
                # Clear chunk
                chunk.clear()
                
                # Check memory
                await self.memory_optimizer.check_memory_pressure()
        
        # Process remaining items
        if chunk:
            processed = await process_func(chunk)
            for result in processed:
                yield result
```

## Frontend Optimization

### React Performance

```typescript
// frontend/src/hooks/useOptimizedState.ts
import { useState, useCallback, useRef, useEffect } from 'react';
import { debounce, throttle } from 'lodash';

// Optimized state hook with debouncing
export function useOptimizedState<T>(
  initialValue: T,
  delay: number = 300
) {
  const [value, setValue] = useState<T>(initialValue);
  const [debouncedValue, setDebouncedValue] = useState<T>(initialValue);
  
  const debouncedSetValue = useCallback(
    debounce((newValue: T) => {
      setDebouncedValue(newValue);
    }, delay),
    [delay]
  );
  
  useEffect(() => {
    debouncedSetValue(value);
  }, [value, debouncedSetValue]);
  
  return [value, setValue, debouncedValue] as const;
}

// Virtual scrolling hook for large lists
export function useVirtualScroll<T>(
  items: T[],
  itemHeight: number,
  containerHeight: number,
  overscan: number = 5
) {
  const [scrollTop, setScrollTop] = useState(0);
  
  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / itemHeight) - overscan
  );
  
  const endIndex = Math.min(
    items.length - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  );
  
  const visibleItems = items.slice(startIndex, endIndex + 1);
  
  const totalHeight = items.length * itemHeight;
  const offsetY = startIndex * itemHeight;
  
  const handleScroll = throttle((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, 16); // 60fps
  
  return {
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll,
    startIndex,
    endIndex
  };
}

// Memoization helper for expensive computations
export function useMemoizedComputation<T>(
  computation: () => T,
  deps: React.DependencyList
): T {
  const resultRef = useRef<T>();
  const depsRef = useRef<React.DependencyList>();
  
  if (!depsRef.current || !depsEqual(depsRef.current, deps)) {
    resultRef.current = computation();
    depsRef.current = deps;
  }
  
  return resultRef.current as T;
}

function depsEqual(a: React.DependencyList, b: React.DependencyList): boolean {
  if (a.length !== b.length) return false;
  
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) return false;
  }
  
  return true;
}
```

### Code Splitting

```typescript
// frontend/src/App.tsx
import React, { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import LoadingSpinner from './components/LoadingSpinner';

// Lazy load heavy components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analysis = lazy(() => import('./pages/Analysis'));
const Trading = lazy(() => import('./pages/Trading'));
const SystemStatus = lazy(() => import('./pages/SystemStatus'));

// Preload component helper
export function preloadComponent(
  componentPromise: () => Promise<{ default: React.ComponentType<any> }>
) {
  const Component = lazy(componentPromise);
  componentPromise(); // Start loading immediately
  return Component;
}

// Intersection Observer for lazy loading
export function LazyLoad({ children }: { children: React.ReactNode }) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    
    if (ref.current) {
      observer.observe(ref.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div ref={ref}>
      {isVisible ? children : <div style={{ minHeight: '200px' }} />}
    </div>
  );
}
```

### WebSocket Optimization

```typescript
// frontend/src/services/optimizedWebSocket.ts
interface Message {
  id: string;
  type: string;
  data: any;
  timestamp: number;
}

export class OptimizedWebSocket {
  private ws: WebSocket | null = null;
  private messageQueue: Message[] = [];
  private subscribers = new Map<string, Set<(data: any) => void>>();
  private reconnectAttempts = 0;
  private messageBuffer: Message[] = [];
  private flushInterval: NodeJS.Timer | null = null;
  
  constructor(
    private url: string,
    private options: {
      batchInterval?: number;
      maxBatchSize?: number;
      compression?: boolean;
    } = {}
  ) {
    this.connect();
  }
  
  private connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.flushMessageQueue();
      this.startBatching();
    };
    
    this.ws.onmessage = async (event) => {
      const messages = await this.parseMessage(event.data);
      
      for (const message of messages) {
        this.handleMessage(message);
      }
    };
    
    this.ws.onclose = () => {
      this.handleReconnect();
    };
  }
  
  private async parseMessage(data: string | Blob): Promise<Message[]> {
    if (this.options.compression && data instanceof Blob) {
      // Decompress if needed
      const arrayBuffer = await data.arrayBuffer();
      const decompressed = await this.decompress(arrayBuffer);
      return JSON.parse(decompressed);
    }
    
    const parsed = JSON.parse(data as string);
    return Array.isArray(parsed) ? parsed : [parsed];
  }
  
  private handleMessage(message: Message) {
    const subscribers = this.subscribers.get(message.type);
    
    if (subscribers) {
      // Use requestAnimationFrame for UI updates
      requestAnimationFrame(() => {
        subscribers.forEach(callback => callback(message.data));
      });
    }
  }
  
  send(message: Partial<Message>) {
    const fullMessage: Message = {
      id: this.generateId(),
      timestamp: Date.now(),
      type: '',
      data: null,
      ...message
    };
    
    if (this.options.batchInterval) {
      this.messageBuffer.push(fullMessage);
    } else {
      this.sendImmediate(fullMessage);
    }
  }
  
  private sendImmediate(message: Message | Message[]) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const data = this.options.compression
        ? this.compress(JSON.stringify(message))
        : JSON.stringify(message);
      
      this.ws.send(data);
    } else {
      // Queue for later
      if (Array.isArray(message)) {
        this.messageQueue.push(...message);
      } else {
        this.messageQueue.push(message);
      }
    }
  }
  
  private startBatching() {
    if (!this.options.batchInterval) return;
    
    this.flushInterval = setInterval(() => {
      if (this.messageBuffer.length > 0) {
        this.sendImmediate(this.messageBuffer);
        this.messageBuffer = [];
      }
    }, this.options.batchInterval);
  }
  
  subscribe(type: string, callback: (data: any) => void) {
    if (!this.subscribers.has(type)) {
      this.subscribers.set(type, new Set());
    }
    
    this.subscribers.get(type)!.add(callback);
    
    return () => {
      this.subscribers.get(type)?.delete(callback);
    };
  }
  
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  private async compress(data: string): Promise<ArrayBuffer> {
    const encoder = new TextEncoder();
    const stream = new CompressionStream('gzip');
    const writer = stream.writable.getWriter();
    
    writer.write(encoder.encode(data));
    writer.close();
    
    const chunks: Uint8Array[] = [];
    const reader = stream.readable.getReader();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    
    const compressed = new Uint8Array(
      chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    );
    
    let offset = 0;
    for (const chunk of chunks) {
      compressed.set(chunk, offset);
      offset += chunk.length;
    }
    
    return compressed.buffer;
  }
  
  private async decompress(data: ArrayBuffer): Promise<string> {
    const stream = new DecompressionStream('gzip');
    const writer = stream.writable.getWriter();
    
    writer.write(new Uint8Array(data));
    writer.close();
    
    const chunks: Uint8Array[] = [];
    const reader = stream.readable.getReader();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    
    const decompressed = new Uint8Array(
      chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    );
    
    let offset = 0;
    for (const chunk of chunks) {
      decompressed.set(chunk, offset);
      offset += chunk.length;
    }
    
    const decoder = new TextDecoder();
    return decoder.decode(decompressed);
  }
}
```

## Database Optimization

### Query Optimization

```sql
-- Optimized indexes for common queries
CREATE INDEX CONCURRENTLY idx_trades_symbol_timestamp 
ON trades(symbol, timestamp DESC) 
WHERE status = 'completed';

CREATE INDEX CONCURRENTLY idx_positions_user_active 
ON positions(user_id) 
WHERE closed_at IS NULL;

CREATE INDEX CONCURRENTLY idx_orders_user_status_created 
ON orders(user_id, status, created_at DESC);

-- Partial indexes for better performance
CREATE INDEX CONCURRENTLY idx_large_trades 
ON trades(symbol, timestamp) 
WHERE quantity > 1000;

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_analysis_composite 
ON analysis_results(user_id, symbol, analysis_type, created_at DESC);

-- Use BRIN indexes for time-series data
CREATE INDEX idx_trades_timestamp_brin 
ON trades USING BRIN(timestamp);

-- Optimize foreign key checks
ALTER TABLE orders 
ADD CONSTRAINT fk_orders_user 
FOREIGN KEY (user_id) 
REFERENCES users(id) 
DEFERRABLE INITIALLY DEFERRED;
```

### Connection Pooling

```python
# backend/app/services/db_pool.py
import asyncpg
from typing import Optional
import asyncio

class DatabasePool:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None
        self.read_pool: Optional[asyncpg.Pool] = None
        
    async def init_pools(self):
        """Initialize connection pools"""
        # Main pool for writes
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60,
            statement_cache_size=1000,
            max_cached_statement_lifetime=3600
        )
        
        # Separate pool for reads (can point to replicas)
        self.read_pool = await asyncpg.create_pool(
            self.dsn,  # Use read replica DSN in production
            min_size=20,
            max_size=100,
            max_queries=100000,
            command_timeout=30
        )
    
    async def execute_read(self, query: str, *args, timeout: float = 30):
        """Execute read query on read pool"""
        async with self.read_pool.acquire() as connection:
            return await connection.fetch(query, *args, timeout=timeout)
    
    async def execute_write(self, query: str, *args, timeout: float = 60):
        """Execute write query on main pool"""
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args, timeout=timeout)
    
    async def execute_batch(self, queries: List[Tuple[str, tuple]]):
        """Execute multiple queries in a transaction"""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                results = []
                for query, args in queries:
                    result = await connection.execute(query, *args)
                    results.append(result)
                return results
    
    async def copy_records(self, table: str, records: List[tuple]):
        """Bulk insert using COPY for maximum performance"""
        async with self.pool.acquire() as connection:
            await connection.copy_records_to_table(
                table,
                records=records,
                columns=['col1', 'col2', 'col3']
            )
```

### Query Result Caching

```python
# backend/app/services/query_cache.py
import hashlib
import json
from typing import Any, Optional, Callable
import redis.asyncio as aioredis
from functools import wraps

class QueryCache:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
    def cached_query(
        self,
        ttl: int = 300,
        key_prefix: str = "query"
    ):
        """Decorator for caching query results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    key_prefix, func.__name__, args, kwargs
                )
                
                # Try to get from cache
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
                
                # Execute query
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(
        self,
        prefix: str,
        func_name: str,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Generate deterministic cache key"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match=pattern, count=100
            )
            
            if keys:
                await self.redis.delete(*keys)
            
            if cursor == 0:
                break

# Usage example
class OptimizedDataService:
    def __init__(self, db_pool: DatabasePool, cache: QueryCache):
        self.db = db_pool
        self.cache = cache
    
    @cache.cached_query(ttl=600, key_prefix="market_data")
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ):
        """Get historical data with caching"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = $1 
            AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp
        """
        
        return await self.db.execute_read(
            query, symbol, start_date, end_date
        )
```

## Infrastructure Scaling

### Horizontal Scaling

```yaml
# kubernetes/backend-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: finplatform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 10
        periodSeconds: 60
```

### Load Balancing

```nginx
# nginx/load-balancer.conf
upstream backend_cluster {
    least_conn;  # Use least connections algorithm
    
    server backend-1:8000 weight=10 max_fails=3 fail_timeout=30s;
    server backend-2:8000 weight=10 max_fails=3 fail_timeout=30s;
    server backend-3:8000 weight=10 max_fails=3 fail_timeout=30s;
    
    # Health check
    check interval=5000 rise=2 fall=3 timeout=2000 type=http;
    check_http_send "GET /api/health HTTP/1.0\r\n\r\n";
    check_http_expect_alive http_2xx;
    
    # Connection pooling
    keepalive 100;
    keepalive_requests 1000;
    keepalive_timeout 60s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://backend_cluster;
        
        # Performance headers
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_request_buffering off;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Circuit breaker
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 10s;
    }
}
```

## Monitoring and Profiling

### Performance Metrics

```python
# backend/app/services/performance_metrics.py
import time
import asyncio
from typing import Callable, Any
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge
import psutil

# Prometheus metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

memory_usage = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes'
)

cpu_usage = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

class PerformanceMonitor:
    def __init__(self):
        self.start_monitoring()
        
    def start_monitoring(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._monitor_system_resources())
    
    async def _monitor_system_resources(self):
        """Monitor system resources"""
        while True:
            # Update memory usage
            memory = psutil.virtual_memory()
            memory_usage.set(memory.used)
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.set(cpu_percent)
            
            await asyncio.sleep(10)
    
    def track_request(self, method: str, endpoint: str):
        """Decorator to track request metrics"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = 200
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 500
                    raise
                finally:
                    duration = time.time() - start_time
                    
                    # Update metrics
                    request_count.labels(
                        method=method,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    
                    request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
            
            return wrapper
        return decorator

# APM Integration
from elasticapm import async_capture_span

class APMIntegration:
    @staticmethod
    def trace_method(name: str):
        """Decorator for APM tracing"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with async_capture_span(name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
```

### Performance Testing

```python
# tests/performance/load_test.py
import asyncio
import aiohttp
import time
from typing import List, Dict
import statistics

class LoadTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: List[Dict] = []
        
    async def run_load_test(
        self,
        endpoint: str,
        concurrent_requests: int,
        total_requests: int
    ):
        """Run load test on endpoint"""
        print(f"Running load test: {concurrent_requests} concurrent, {total_requests} total")
        
        sem = asyncio.Semaphore(concurrent_requests)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._make_request(session, endpoint, sem, i)
                for i in range(total_requests)
            ]
            
            start_time = time.time()
            await asyncio.gather(*tasks)
            total_time = time.time() - start_time
        
        # Calculate statistics
        self._print_results(total_time, total_requests)
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        sem: asyncio.Semaphore,
        request_id: int
    ):
        """Make single request with timing"""
        async with sem:
            start_time = time.time()
            
            try:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    await response.read()
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'request_id': request_id,
                        'status': response.status,
                        'duration': duration,
                        'success': response.status == 200
                    })
            except Exception as e:
                duration = time.time() - start_time
                self.results.append({
                    'request_id': request_id,
                    'status': 0,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })
    
    def _print_results(self, total_time: float, total_requests: int):
        """Print test results"""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        durations = [r['duration'] for r in successful]
        
        if durations:
            print(f"\nResults:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Requests/second: {total_requests / total_time:.2f}")
            print(f"Success rate: {len(successful) / len(self.results) * 100:.2f}%")
            print(f"\nResponse times (successful requests):")
            print(f"Min: {min(durations) * 1000:.2f}ms")
            print(f"Max: {max(durations) * 1000:.2f}ms")
            print(f"Mean: {statistics.mean(durations) * 1000:.2f}ms")
            print(f"Median: {statistics.median(durations) * 1000:.2f}ms")
            print(f"P95: {statistics.quantiles(durations, n=20)[18] * 1000:.2f}ms")
            print(f"P99: {statistics.quantiles(durations, n=100)[98] * 1000:.2f}ms")

# Run load test
async def main():
    tester = LoadTester("http://localhost:8000")
    
    # Warm up
    await tester.run_load_test("/api/health", 10, 100)
    
    # Actual test
    await tester.run_load_test("/api/data/quote/AAPL", 100, 10000)

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Checklist

### Backend Performance
- [ ] Async/await used throughout
- [ ] Database connection pooling configured
- [ ] Query optimization implemented
- [ ] Caching strategy in place
- [ ] GPU acceleration for compute-intensive tasks
- [ ] Memory management optimized
- [ ] Background tasks properly managed

### Frontend Performance
- [ ] Code splitting implemented
- [ ] Lazy loading for components
- [ ] Virtual scrolling for large lists
- [ ] WebSocket batching enabled
- [ ] Image optimization
- [ ] Bundle size minimized
- [ ] Service worker for caching

### Database Performance
- [ ] Indexes optimized
- [ ] Query plans analyzed
- [ ] Connection pooling configured
- [ ] Read replicas utilized
- [ ] Partitioning implemented
- [ ] Vacuum and analyze scheduled
- [ ] Slow query logging enabled

### Infrastructure Performance
- [ ] Horizontal scaling configured
- [ ] Load balancing optimized
- [ ] CDN implemented
- [ ] Caching layers deployed
- [ ] Monitoring in place
- [ ] Performance testing automated
- [ ] Resource limits set appropriately