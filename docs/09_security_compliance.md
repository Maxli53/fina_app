# Security and Compliance Guide

## Overview

This guide covers security best practices, compliance requirements, and implementation details for the Financial Time Series Analysis Platform.

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────┐
│                    External Security                        │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ DDoS Protection│  │ WAF Rules    │  │ Rate Limiting  │ │
│  └───────────────┘  └──────────────┘  └────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Network Security                         │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ TLS 1.3       │  │ VPN Access   │  │ Network Policies│ │
│  └───────────────┘  └──────────────┘  └────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Application Security                       │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Authentication│  │ Authorization│  │ Input Validation│ │
│  └───────────────┘  └──────────────┘  └────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Data Security                           │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Encryption    │  │ Key Management│  │ Data Masking   │ │
│  └───────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Authentication and Authorization

### JWT Implementation

```python
# backend/app/services/auth.py
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 1440  # 24 hours
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "finplatform"
        })
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        return encoded_jwt
    
    def decode_access_token(self, token: str) -> dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
```

### Role-Based Access Control (RBAC)

```python
# backend/app/models/auth.py
from enum import Enum
from typing import List

class Role(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    # Trading permissions
    TRADE_EXECUTE = "trade:execute"
    TRADE_VIEW = "trade:view"
    
    # Analysis permissions
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_VIEW = "analysis:view"
    ANALYSIS_DELETE = "analysis:delete"
    
    # Strategy permissions
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_EDIT = "strategy:edit"
    STRATEGY_DELETE = "strategy:delete"
    STRATEGY_DEPLOY = "strategy:deploy"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],  # All permissions
    Role.TRADER: [
        Permission.TRADE_EXECUTE,
        Permission.TRADE_VIEW,
        Permission.ANALYSIS_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EDIT,
        Permission.STRATEGY_DEPLOY
    ],
    Role.ANALYST: [
        Permission.ANALYSIS_CREATE,
        Permission.ANALYSIS_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EDIT,
        Permission.TRADE_VIEW
    ],
    Role.VIEWER: [
        Permission.TRADE_VIEW,
        Permission.ANALYSIS_VIEW,
        Permission.SYSTEM_MONITOR
    ]
}

# Permission decorator
def require_permission(permission: Permission):
    async def decorator(current_user: User = Depends(get_current_user)):
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return current_user
    return decorator
```

### Multi-Factor Authentication (MFA)

```python
# backend/app/services/mfa.py
import pyotp
import qrcode
from io import BytesIO
import base64

class MFAService:
    def __init__(self):
        self.issuer = "FinPlatform"
        
    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, email: str, secret: str) -> str:
        """Generate QR code for authenticator app"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email,
            issuer_name=self.issuer
        )
        
        qr = qrcode.QRCode(version=1, auto_commit=True)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes"""
        return [pyotp.random_base32()[:8] for _ in range(count)]
```

## Data Security

### Encryption at Rest

```python
# backend/app/services/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionService:
    def __init__(self, master_key: str):
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # Use proper salt management
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(master_key.encode())
        )
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_file(self, file_path: str) -> None:
        """Encrypt file in place"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.cipher.encrypt(data)
        
        with open(file_path + '.enc', 'wb') as f:
            f.write(encrypted)
        
        # Securely delete original
        os.remove(file_path)
```

### Database Security

```sql
-- Enable Row Level Security (RLS)
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY positions_user_policy ON positions
    FOR ALL
    TO application_user
    USING (user_id = current_setting('app.current_user_id')::INTEGER);

CREATE POLICY orders_user_policy ON orders
    FOR ALL
    TO application_user
    USING (user_id = current_setting('app.current_user_id')::INTEGER);

-- Encrypt sensitive columns
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt API keys
ALTER TABLE user_credentials 
ADD COLUMN encrypted_api_key TEXT;

UPDATE user_credentials 
SET encrypted_api_key = pgp_sym_encrypt(api_key, 'encryption_key');

-- Create audit table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(100),
    table_name VARCHAR(100),
    record_id INTEGER,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        user_id,
        action,
        table_name,
        record_id,
        old_values,
        new_values,
        ip_address,
        user_agent
    ) VALUES (
        current_setting('app.current_user_id')::INTEGER,
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        to_jsonb(OLD),
        to_jsonb(NEW),
        inet_client_addr(),
        current_setting('app.user_agent', true)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers
CREATE TRIGGER audit_positions
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

## API Security

### Input Validation

```python
# backend/app/services/validation.py
import re
from typing import Any, Dict
from pydantic import BaseModel, validator
import bleach

class SecurityValidator:
    @staticmethod
    def validate_sql_injection(value: str) -> str:
        """Prevent SQL injection"""
        dangerous_patterns = [
            r"(\b(DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)",
            r"(--|#|/\*|\*/)",
            r"(\x00|\x1a)",  # NULL bytes
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError("Potentially dangerous input detected")
        
        return value
    
    @staticmethod
    def validate_xss(value: str) -> str:
        """Prevent XSS attacks"""
        # Whitelist allowed tags and attributes
        allowed_tags = ['b', 'i', 'u', 'em', 'strong']
        allowed_attributes = {}
        
        cleaned = bleach.clean(
            value,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        return cleaned
    
    @staticmethod
    def validate_path_traversal(path: str) -> str:
        """Prevent path traversal attacks"""
        if '..' in path or path.startswith('/'):
            raise ValueError("Invalid path")
        
        # Normalize and validate
        safe_path = os.path.normpath(path)
        if not safe_path.startswith('data/'):
            raise ValueError("Path outside allowed directory")
        
        return safe_path

# Request validation models
class TradingRequest(BaseModel):
    symbol: str
    quantity: int
    order_type: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError('Invalid symbol format')
        return SecurityValidator.validate_sql_injection(v)
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0 or v > 1000000:
            raise ValueError('Invalid quantity')
        return v
```

### Rate Limiting

```python
# backend/app/middleware/rate_limit.py
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from typing import Optional

class RateLimiter:
    def __init__(
        self, 
        redis_client: aioredis.Redis,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.redis = redis_client
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
    
    async def check_rate_limit(
        self, 
        key: str, 
        cost: int = 1
    ) -> tuple[bool, Optional[int]]:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        
        # Check minute limit
        minute_key = f"rate:{key}:minute:{now.minute}"
        minute_count = await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        if minute_count > self.rpm:
            return False, 60
        
        # Check hour limit
        hour_key = f"rate:{key}:hour:{now.hour}"
        hour_count = await self.redis.incr(hour_key)
        await self.redis.expire(hour_key, 3600)
        
        if hour_count > self.rph:
            return False, 3600
        
        return True, None

# Rate limit middleware
async def rate_limit_middleware(request: Request, call_next):
    # Get user identifier
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        user_id = request.client.host
    
    # Check rate limit
    limiter = request.app.state.rate_limiter
    allowed, retry_after = await limiter.check_rate_limit(user_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    
    response = await call_next(request)
    return response
```

### API Key Management

```python
# backend/app/services/api_key.py
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional

class APIKeyService:
    def __init__(self, db_session):
        self.db = db_session
        
    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and hash"""
        # Generate secure random key
        api_key = f"fpl_{secrets.token_urlsafe(32)}"
        
        # Hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        return api_key, key_hash
    
    async def create_api_key(
        self,
        user_id: int,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key for user"""
        api_key, key_hash = self.generate_api_key()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store in database
        await self.db.execute(
            """
            INSERT INTO api_keys 
            (user_id, name, key_hash, permissions, expires_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user_id, name, key_hash, permissions, expires_at
        )
        
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return permissions"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        result = await self.db.fetchone(
            """
            SELECT user_id, permissions, expires_at
            FROM api_keys
            WHERE key_hash = $1 AND revoked = FALSE
            """,
            key_hash
        )
        
        if not result:
            return None
        
        # Check expiration
        if result['expires_at'] and result['expires_at'] < datetime.utcnow():
            return None
        
        # Update last used
        await self.db.execute(
            """
            UPDATE api_keys 
            SET last_used = NOW(), usage_count = usage_count + 1
            WHERE key_hash = $1
            """,
            key_hash
        )
        
        return {
            'user_id': result['user_id'],
            'permissions': result['permissions']
        }
```

## Compliance

### Financial Regulations

#### MiFID II Compliance

```python
# backend/app/services/compliance/mifid.py
from datetime import datetime
from typing import Dict, Any

class MiFIDCompliance:
    """MiFID II compliance implementation"""
    
    async def record_transaction_report(
        self,
        order: Dict[str, Any]
    ) -> None:
        """Record transaction for regulatory reporting"""
        report = {
            "transaction_reference": order['id'],
            "trading_venue": order['exchange'],
            "execution_timestamp": order['executed_at'],
            "instrument_id": order['symbol'],
            "price": order['price'],
            "quantity": order['quantity'],
            "buyer_id": order.get('buyer_id'),
            "seller_id": order.get('seller_id'),
            "trading_capacity": "DEAL",  # DEAL or AOTC
            "reporting_timestamp": datetime.utcnow()
        }
        
        # Store for regulatory reporting
        await self.store_transaction_report(report)
    
    async def best_execution_analysis(
        self,
        order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze best execution compliance"""
        # Compare execution price with market
        market_price = await self.get_market_price(
            order['symbol'], 
            order['executed_at']
        )
        
        slippage = abs(order['price'] - market_price) / market_price
        
        return {
            "order_id": order['id'],
            "execution_price": order['price'],
            "market_price": market_price,
            "slippage": slippage,
            "compliant": slippage < 0.005  # 0.5% threshold
        }
```

#### GDPR Compliance

```python
# backend/app/services/compliance/gdpr.py
from typing import Dict, Any
import json

class GDPRCompliance:
    """GDPR compliance implementation"""
    
    async def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """Export all user data for GDPR request"""
        data = {
            "user_profile": await self.get_user_profile(user_id),
            "trading_history": await self.get_trading_history(user_id),
            "analysis_history": await self.get_analysis_history(user_id),
            "audit_logs": await self.get_audit_logs(user_id),
            "exported_at": datetime.utcnow().isoformat()
        }
        
        return data
    
    async def anonymize_user_data(self, user_id: int) -> None:
        """Anonymize user data for GDPR deletion"""
        # Anonymize personal data
        await self.db.execute(
            """
            UPDATE users 
            SET 
                email = CONCAT('deleted_', id, '@anonymous.com'),
                name = 'Deleted User',
                phone = NULL,
                address = NULL,
                anonymized_at = NOW()
            WHERE id = $1
            """,
            user_id
        )
        
        # Anonymize related data
        await self.db.execute(
            """
            UPDATE audit_log 
            SET ip_address = '0.0.0.0'
            WHERE user_id = $1
            """,
            user_id
        )
    
    def get_privacy_policy_version(self) -> str:
        """Get current privacy policy version"""
        return "2024.01.25"
    
    async def record_consent(
        self,
        user_id: int,
        consent_type: str,
        granted: bool
    ) -> None:
        """Record user consent"""
        await self.db.execute(
            """
            INSERT INTO user_consent 
            (user_id, consent_type, granted, timestamp, policy_version)
            VALUES ($1, $2, $3, NOW(), $4)
            """,
            user_id, consent_type, granted, self.get_privacy_policy_version()
        )
```

### Audit Logging

```python
# backend/app/services/audit.py
from typing import Dict, Any, Optional
from datetime import datetime
import json

class AuditLogger:
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
        
    async def log_event(
        self,
        user_id: int,
        event_type: str,
        resource_type: str,
        resource_id: Optional[int],
        action: str,
        details: Dict[str, Any],
        ip_address: str,
        user_agent: str
    ) -> None:
        """Log audit event"""
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "details": json.dumps(details),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow()
        }
        
        # Store in database
        await self.db.execute(
            """
            INSERT INTO audit_log 
            (user_id, event_type, resource_type, resource_id, 
             action, details, ip_address, user_agent, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            *event.values()
        )
        
        # Also store in Redis for real-time monitoring
        await self.redis.xadd(
            "audit:stream",
            {"data": json.dumps(event)},
            maxlen=10000  # Keep last 10k events
        )
    
    async def get_audit_trail(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filters"""
        query = """
            SELECT * FROM audit_log 
            WHERE 1=1
        """
        params = []
        
        if 'user_id' in filters:
            query += " AND user_id = $%d" % (len(params) + 1)
            params.append(filters['user_id'])
        
        if 'event_type' in filters:
            query += " AND event_type = $%d" % (len(params) + 1)
            params.append(filters['event_type'])
        
        if 'start_date' in filters:
            query += " AND timestamp >= $%d" % (len(params) + 1)
            params.append(filters['start_date'])
        
        query += " ORDER BY timestamp DESC LIMIT $%d" % (len(params) + 1)
        params.append(limit)
        
        return await self.db.fetch(query, *params)

# Audit middleware
from fastapi import Request

async def audit_middleware(request: Request, call_next):
    """Audit all API requests"""
    start_time = datetime.utcnow()
    
    # Get request details
    user_id = request.state.user_id if hasattr(request.state, 'user_id') else None
    
    # Process request
    response = await call_next(request)
    
    # Log request
    if user_id and request.method in ['POST', 'PUT', 'DELETE']:
        await request.app.state.audit_logger.log_event(
            user_id=user_id,
            event_type="api_request",
            resource_type=request.url.path.split('/')[2] if len(request.url.path.split('/')) > 2 else 'unknown',
            resource_id=None,
            action=request.method,
            details={
                "path": str(request.url.path),
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            },
            ip_address=request.client.host,
            user_agent=request.headers.get('user-agent', '')
        )
    
    return response
```

## Security Headers

```python
# backend/app/middleware/security_headers.py
from fastapi import Request
from fastapi.responses import Response

async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # CSP header
    response.headers["Content-Security-Policy"] = " ".join([
        "default-src 'self';",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net;",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;",
        "font-src 'self' https://fonts.gstatic.com;",
        "img-src 'self' data: https:;",
        "connect-src 'self' wss: https:;"
    ])
    
    return response
```

## Incident Response

### Security Incident Workflow

```python
# backend/app/services/incident_response.py
from enum import Enum
from typing import Dict, Any, List

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentResponse:
    def __init__(self, notification_service, system_orchestrator):
        self.notifications = notification_service
        self.orchestrator = system_orchestrator
        
    async def handle_security_incident(
        self,
        incident_type: str,
        severity: IncidentSeverity,
        details: Dict[str, Any]
    ) -> None:
        """Handle security incident"""
        incident_id = await self.create_incident_record(
            incident_type, severity, details
        )
        
        # Immediate response based on severity
        if severity == IncidentSeverity.CRITICAL:
            await self.critical_incident_response(incident_id, details)
        elif severity == IncidentSeverity.HIGH:
            await self.high_incident_response(incident_id, details)
        else:
            await self.standard_incident_response(incident_id, details)
    
    async def critical_incident_response(
        self,
        incident_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Response for critical incidents"""
        # 1. Immediate containment
        if details.get('user_id'):
            await self.suspend_user_access(details['user_id'])
        
        if details.get('ip_address'):
            await self.block_ip_address(details['ip_address'])
        
        # 2. Stop trading if necessary
        if details.get('affects_trading'):
            await self.orchestrator.emergency_trading_halt()
        
        # 3. Notify security team
        await self.notifications.send_emergency_alert(
            "CRITICAL SECURITY INCIDENT",
            incident_id,
            details
        )
        
        # 4. Preserve evidence
        await self.preserve_incident_evidence(incident_id)
    
    async def preserve_incident_evidence(
        self,
        incident_id: str
    ) -> None:
        """Preserve evidence for investigation"""
        # Snapshot relevant logs
        await self.snapshot_logs(incident_id)
        
        # Capture system state
        await self.capture_system_state(incident_id)
        
        # Export audit trail
        await self.export_audit_trail(incident_id)
```

### Security Monitoring

```python
# backend/app/services/security_monitor.py
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

class SecurityMonitor:
    def __init__(self, redis_client, incident_response):
        self.redis = redis_client
        self.incident_handler = incident_response
        self.rules = self._load_detection_rules()
        
    def _load_detection_rules(self) -> List[Dict]:
        """Load security detection rules"""
        return [
            {
                "name": "brute_force_attack",
                "condition": lambda e: e['failed_logins'] > 5,
                "window": 300,  # 5 minutes
                "severity": IncidentSeverity.HIGH
            },
            {
                "name": "suspicious_api_usage",
                "condition": lambda e: e['api_calls'] > 1000,
                "window": 60,  # 1 minute
                "severity": IncidentSeverity.MEDIUM
            },
            {
                "name": "unauthorized_access_attempt",
                "condition": lambda e: e['403_errors'] > 10,
                "window": 300,
                "severity": IncidentSeverity.HIGH
            },
            {
                "name": "data_exfiltration_attempt",
                "condition": lambda e: e['data_downloaded'] > 1000000000,  # 1GB
                "window": 3600,  # 1 hour
                "severity": IncidentSeverity.CRITICAL
            }
        ]
    
    async def monitor_security_events(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Process events from Redis stream
                events = await self.redis.xread(
                    {"security:events": "$"},
                    block=1000
                )
                
                for stream, messages in events:
                    for message_id, data in messages:
                        await self.analyze_event(data)
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def analyze_event(self, event_data: Dict) -> None:
        """Analyze security event against rules"""
        for rule in self.rules:
            # Get events in time window
            window_events = await self.get_events_in_window(
                event_data['user_id'],
                rule['window']
            )
            
            # Check if rule triggered
            if rule['condition'](window_events):
                await self.incident_handler.handle_security_incident(
                    incident_type=rule['name'],
                    severity=rule['severity'],
                    details={
                        "user_id": event_data['user_id'],
                        "ip_address": event_data.get('ip_address'),
                        "trigger_event": event_data,
                        "window_events": window_events
                    }
                )
```

## Security Checklist

### Development Security
- [ ] All dependencies regularly updated
- [ ] Security linting enabled (bandit, safety)
- [ ] Secrets scanning in CI/CD
- [ ] Code review for security issues
- [ ] SAST/DAST tools integrated

### Infrastructure Security
- [ ] TLS 1.3 enabled
- [ ] Security headers configured
- [ ] WAF rules active
- [ ] DDoS protection enabled
- [ ] Network segmentation implemented

### Application Security
- [ ] Authentication required for all endpoints
- [ ] Authorization checks implemented
- [ ] Input validation on all inputs
- [ ] Output encoding enabled
- [ ] Rate limiting active

### Data Security
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enforced
- [ ] PII data masked in logs
- [ ] Secure key management
- [ ] Regular backups encrypted

### Compliance
- [ ] GDPR compliance verified
- [ ] MiFID II reporting active
- [ ] Audit logging enabled
- [ ] Data retention policies enforced
- [ ] Regular compliance audits

### Incident Response
- [ ] Incident response plan documented
- [ ] Security monitoring active
- [ ] Alert system configured
- [ ] Forensics tools ready
- [ ] Recovery procedures tested