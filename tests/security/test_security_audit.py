"""
Comprehensive security audit and penetration testing suite
"""

import asyncio
import aiohttp
import pytest
from typing import Dict, List, Any, Tuple
import json
import re
import hashlib
import base64
from datetime import datetime, timedelta
import secrets
import ssl
import socket
from urllib.parse import urlparse, urljoin
import subprocess
import os
from dataclasses import dataclass
from enum import Enum
import jwt
import logging

logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityVulnerability:
    """Security vulnerability details"""
    test_name: str
    severity: VulnerabilitySeverity
    description: str
    affected_endpoint: str
    evidence: Dict[str, Any]
    remediation: str
    cwe_id: str = None
    owasp_category: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "severity": self.severity.value,
            "description": self.description,
            "affected_endpoint": self.affected_endpoint,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "timestamp": datetime.utcnow().isoformat()
        }


class SecurityAuditor:
    """Comprehensive security auditing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.vulnerabilities: List[SecurityVulnerability] = []
        self.test_results: Dict[str, Any] = {}
        
    async def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        print("Starting comprehensive security audit...")
        
        # Authentication tests
        await self.test_authentication_security()
        
        # Authorization tests
        await self.test_authorization_security()
        
        # Input validation tests
        await self.test_input_validation()
        
        # Injection tests
        await self.test_injection_vulnerabilities()
        
        # XSS tests
        await self.test_xss_vulnerabilities()
        
        # CSRF tests
        await self.test_csrf_protection()
        
        # Security headers tests
        await self.test_security_headers()
        
        # SSL/TLS tests
        await self.test_ssl_tls_configuration()
        
        # API security tests
        await self.test_api_security()
        
        # Session management tests
        await self.test_session_management()
        
        # File upload tests
        await self.test_file_upload_security()
        
        # Rate limiting tests
        await self.test_rate_limiting()
        
        # Business logic tests
        await self.test_business_logic_security()
        
        # Generate report
        return self.generate_audit_report()
    
    async def test_authentication_security(self):
        """Test authentication mechanisms"""
        print("\n[*] Testing Authentication Security...")
        
        # Test weak passwords
        weak_passwords = ["123456", "password", "admin", "test", "demo"]
        for password in weak_passwords:
            async with aiohttp.ClientSession() as session:
                try:
                    response = await session.post(
                        f"{self.base_url}/api/auth/login",
                        json={"username": "admin", "password": password}
                    )
                    if response.status == 200:
                        self.add_vulnerability(
                            test_name="Weak Password Allowed",
                            severity=VulnerabilitySeverity.CRITICAL,
                            description="System accepts weak passwords",
                            affected_endpoint="/api/auth/login",
                            evidence={"password": password},
                            remediation="Implement strong password policy",
                            cwe_id="CWE-521"
                        )
                except:
                    pass
        
        # Test brute force protection
        async with aiohttp.ClientSession() as session:
            failed_attempts = 0
            for i in range(20):
                try:
                    response = await session.post(
                        f"{self.base_url}/api/auth/login",
                        json={"username": "testuser", "password": f"wrong{i}"}
                    )
                    if response.status != 429:  # Not rate limited
                        failed_attempts += 1
                except:
                    pass
            
            if failed_attempts >= 15:
                self.add_vulnerability(
                    test_name="No Brute Force Protection",
                    severity=VulnerabilitySeverity.HIGH,
                    description="No rate limiting on authentication endpoint",
                    affected_endpoint="/api/auth/login",
                    evidence={"failed_attempts": failed_attempts},
                    remediation="Implement rate limiting and account lockout",
                    cwe_id="CWE-307"
                )
        
        # Test password reset token randomness
        await self.test_token_randomness()
    
    async def test_authorization_security(self):
        """Test authorization mechanisms"""
        print("\n[*] Testing Authorization Security...")
        
        # Test horizontal privilege escalation
        async with aiohttp.ClientSession() as session:
            # Get tokens for two different users
            user1_token = await self.get_test_token("user1")
            user2_token = await self.get_test_token("user2")
            
            if user1_token and user2_token:
                # Try to access user2's data with user1's token
                headers = {"Authorization": f"Bearer {user1_token}"}
                response = await session.get(
                    f"{self.base_url}/api/users/user2/positions",
                    headers=headers
                )
                
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="Horizontal Privilege Escalation",
                        severity=VulnerabilitySeverity.HIGH,
                        description="Users can access other users' data",
                        affected_endpoint="/api/users/{id}/positions",
                        evidence={"status_code": response.status},
                        remediation="Implement proper access controls",
                        cwe_id="CWE-639"
                    )
        
        # Test vertical privilege escalation
        await self.test_privilege_escalation()
    
    async def test_injection_vulnerabilities(self):
        """Test for injection vulnerabilities"""
        print("\n[*] Testing Injection Vulnerabilities...")
        
        # SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "1' UNION SELECT * FROM users--",
            "admin'--",
            "1' AND SLEEP(5)--"
        ]
        
        endpoints = [
            "/api/data/search",
            "/api/analysis/results",
            "/api/trading/orders"
        ]
        
        for endpoint in endpoints:
            for payload in sql_payloads:
                async with aiohttp.ClientSession() as session:
                    try:
                        # Test GET parameters
                        response = await session.get(
                            f"{self.base_url}{endpoint}?query={payload}"
                        )
                        
                        # Check for SQL error messages
                        if response.status == 500:
                            text = await response.text()
                            if any(err in text.lower() for err in ['sql', 'syntax', 'query']):
                                self.add_vulnerability(
                                    test_name="SQL Injection",
                                    severity=VulnerabilitySeverity.CRITICAL,
                                    description="SQL injection vulnerability detected",
                                    affected_endpoint=endpoint,
                                    evidence={"payload": payload, "response": text[:200]},
                                    remediation="Use parameterized queries",
                                    cwe_id="CWE-89",
                                    owasp_category="A03:2021"
                                )
                    except:
                        pass
        
        # Command injection tests
        await self.test_command_injection()
        
        # LDAP injection tests
        await self.test_ldap_injection()
    
    async def test_xss_vulnerabilities(self):
        """Test for XSS vulnerabilities"""
        print("\n[*] Testing XSS Vulnerabilities...")
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'>"
        ]
        
        # Test reflected XSS
        for payload in xss_payloads:
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                    f"{self.base_url}/api/search?q={payload}"
                )
                
                if response.status == 200:
                    text = await response.text()
                    if payload in text:
                        self.add_vulnerability(
                            test_name="Reflected XSS",
                            severity=VulnerabilitySeverity.HIGH,
                            description="Reflected XSS vulnerability",
                            affected_endpoint="/api/search",
                            evidence={"payload": payload},
                            remediation="Encode all user input in responses",
                            cwe_id="CWE-79",
                            owasp_category="A03:2021"
                        )
        
        # Test stored XSS
        await self.test_stored_xss()
    
    async def test_csrf_protection(self):
        """Test CSRF protection"""
        print("\n[*] Testing CSRF Protection...")
        
        async with aiohttp.ClientSession() as session:
            # Get valid token
            token = await self.get_test_token("testuser")
            
            if token:
                # Try state-changing request without CSRF token
                headers = {"Authorization": f"Bearer {token}"}
                
                response = await session.post(
                    f"{self.base_url}/api/trading/orders",
                    json={"symbol": "AAPL", "quantity": 100, "side": "buy"},
                    headers=headers
                )
                
                # Check if CSRF token is required
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="Missing CSRF Protection",
                        severity=VulnerabilitySeverity.MEDIUM,
                        description="State-changing operations lack CSRF protection",
                        affected_endpoint="/api/trading/orders",
                        evidence={"method": "POST", "status": response.status},
                        remediation="Implement CSRF tokens for state-changing operations",
                        cwe_id="CWE-352"
                    )
    
    async def test_security_headers(self):
        """Test security headers"""
        print("\n[*] Testing Security Headers...")
        
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=",
            "Content-Security-Policy": "default-src",
            "Referrer-Policy": ["no-referrer", "strict-origin"]
        }
        
        async with aiohttp.ClientSession() as session:
            response = await session.get(self.base_url)
            
            for header, expected_values in required_headers.items():
                header_value = response.headers.get(header)
                
                if not header_value:
                    self.add_vulnerability(
                        test_name=f"Missing Security Header: {header}",
                        severity=VulnerabilitySeverity.MEDIUM,
                        description=f"Security header {header} is missing",
                        affected_endpoint="/",
                        evidence={"missing_header": header},
                        remediation=f"Add {header} header to all responses",
                        cwe_id="CWE-693"
                    )
                else:
                    # Check header value
                    if isinstance(expected_values, list):
                        if not any(val in header_value for val in expected_values):
                            self.add_vulnerability(
                                test_name=f"Weak Security Header: {header}",
                                severity=VulnerabilitySeverity.LOW,
                                description=f"Security header {header} has weak configuration",
                                affected_endpoint="/",
                                evidence={"header_value": header_value},
                                remediation=f"Strengthen {header} configuration",
                                cwe_id="CWE-693"
                            )
    
    async def test_ssl_tls_configuration(self):
        """Test SSL/TLS configuration"""
        print("\n[*] Testing SSL/TLS Configuration...")
        
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme == "https":
            try:
                # Test SSL certificate
                context = ssl.create_default_context()
                with socket.create_connection((parsed_url.hostname, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=parsed_url.hostname) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate expiration
                        not_after = datetime.strptime(
                            cert['notAfter'], 
                            '%b %d %H:%M:%S %Y %Z'
                        )
                        
                        if not_after < datetime.utcnow() + timedelta(days=30):
                            self.add_vulnerability(
                                test_name="SSL Certificate Expiring Soon",
                                severity=VulnerabilitySeverity.HIGH,
                                description="SSL certificate expires within 30 days",
                                affected_endpoint="/",
                                evidence={"expires": not_after.isoformat()},
                                remediation="Renew SSL certificate",
                                cwe_id="CWE-295"
                            )
                        
                        # Check TLS version
                        if ssock.version() < "TLSv1.2":
                            self.add_vulnerability(
                                test_name="Weak TLS Version",
                                severity=VulnerabilitySeverity.HIGH,
                                description="Server supports weak TLS versions",
                                affected_endpoint="/",
                                evidence={"tls_version": ssock.version()},
                                remediation="Disable TLS versions below 1.2",
                                cwe_id="CWE-326"
                            )
            except Exception as e:
                logger.error(f"SSL/TLS test error: {e}")
    
    async def test_api_security(self):
        """Test API-specific security"""
        print("\n[*] Testing API Security...")
        
        # Test API key security
        weak_api_keys = [
            "test123",
            "demo",
            "12345678",
            "apikey123"
        ]
        
        for key in weak_api_keys:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-Key": key}
                response = await session.get(
                    f"{self.base_url}/api/data/quote/AAPL",
                    headers=headers
                )
                
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="Weak API Key Accepted",
                        severity=VulnerabilitySeverity.CRITICAL,
                        description="System accepts weak/predictable API keys",
                        affected_endpoint="/api/*",
                        evidence={"api_key": key},
                        remediation="Enforce strong API key generation",
                        cwe_id="CWE-798"
                    )
        
        # Test API versioning
        await self.test_api_versioning()
        
        # Test API rate limiting per key
        await self.test_api_rate_limiting()
    
    async def test_session_management(self):
        """Test session management security"""
        print("\n[*] Testing Session Management...")
        
        async with aiohttp.ClientSession() as session:
            # Login and get token
            login_response = await session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": "testuser", "password": "testpass"}
            )
            
            if login_response.status == 200:
                data = await login_response.json()
                token = data.get("access_token")
                
                if token:
                    # Decode token to check expiration
                    try:
                        decoded = jwt.decode(
                            token, 
                            options={"verify_signature": False}
                        )
                        
                        exp = decoded.get("exp")
                        iat = decoded.get("iat")
                        
                        if exp and iat:
                            lifetime = exp - iat
                            if lifetime > 86400:  # More than 24 hours
                                self.add_vulnerability(
                                    test_name="Excessive Session Lifetime",
                                    severity=VulnerabilitySeverity.MEDIUM,
                                    description="Session tokens have excessive lifetime",
                                    affected_endpoint="/api/auth/login",
                                    evidence={"lifetime_seconds": lifetime},
                                    remediation="Reduce session lifetime to reasonable duration",
                                    cwe_id="CWE-613"
                                )
                    except:
                        pass
                    
                    # Test session fixation
                    await self.test_session_fixation()
    
    async def test_file_upload_security(self):
        """Test file upload security"""
        print("\n[*] Testing File Upload Security...")
        
        # Test dangerous file types
        dangerous_files = [
            ("malicious.exe", b"MZ\x90\x00", "application/x-msdownload"),
            ("shell.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("script.js", b"alert('XSS')", "application/javascript"),
            ("test.svg", b"<svg onload=alert('XSS')>", "image/svg+xml")
        ]
        
        for filename, content, content_type in dangerous_files:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field(
                    'file',
                    content,
                    filename=filename,
                    content_type=content_type
                )
                
                response = await session.post(
                    f"{self.base_url}/api/upload",
                    data=data
                )
                
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="Dangerous File Upload Allowed",
                        severity=VulnerabilitySeverity.CRITICAL,
                        description=f"System accepts dangerous file type: {filename}",
                        affected_endpoint="/api/upload",
                        evidence={"filename": filename, "content_type": content_type},
                        remediation="Implement strict file type validation",
                        cwe_id="CWE-434"
                    )
        
        # Test path traversal in filename
        await self.test_file_upload_path_traversal()
    
    async def test_rate_limiting(self):
        """Test rate limiting implementation"""
        print("\n[*] Testing Rate Limiting...")
        
        endpoints = [
            "/api/data/quote/AAPL",
            "/api/analysis/ml",
            "/api/trading/orders"
        ]
        
        for endpoint in endpoints:
            async with aiohttp.ClientSession() as session:
                requests_made = 0
                rate_limited = False
                
                # Make rapid requests
                for i in range(200):
                    response = await session.get(f"{self.base_url}{endpoint}")
                    requests_made += 1
                    
                    if response.status == 429:
                        rate_limited = True
                        break
                
                if not rate_limited:
                    self.add_vulnerability(
                        test_name="Missing Rate Limiting",
                        severity=VulnerabilitySeverity.MEDIUM,
                        description=f"No rate limiting on endpoint: {endpoint}",
                        affected_endpoint=endpoint,
                        evidence={"requests_made": requests_made},
                        remediation="Implement rate limiting",
                        cwe_id="CWE-770"
                    )
    
    async def test_business_logic_security(self):
        """Test business logic security"""
        print("\n[*] Testing Business Logic Security...")
        
        # Test negative quantity orders
        async with aiohttp.ClientSession() as session:
            token = await self.get_test_token("testuser")
            
            if token:
                headers = {"Authorization": f"Bearer {token}"}
                
                # Try negative quantity
                response = await session.post(
                    f"{self.base_url}/api/trading/orders",
                    json={"symbol": "AAPL", "quantity": -100, "side": "buy"},
                    headers=headers
                )
                
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="Negative Quantity Order Accepted",
                        severity=VulnerabilitySeverity.HIGH,
                        description="System accepts orders with negative quantities",
                        affected_endpoint="/api/trading/orders",
                        evidence={"quantity": -100},
                        remediation="Validate all numeric inputs",
                        cwe_id="CWE-20"
                    )
                
                # Test excessive quantity
                response = await session.post(
                    f"{self.base_url}/api/trading/orders",
                    json={"symbol": "AAPL", "quantity": 999999999, "side": "buy"},
                    headers=headers
                )
                
                if response.status == 200:
                    self.add_vulnerability(
                        test_name="No Maximum Order Size Limit",
                        severity=VulnerabilitySeverity.MEDIUM,
                        description="System lacks maximum order size validation",
                        affected_endpoint="/api/trading/orders",
                        evidence={"quantity": 999999999},
                        remediation="Implement reasonable order size limits",
                        cwe_id="CWE-20"
                    )
    
    # Helper methods
    async def get_test_token(self, username: str) -> str:
        """Get test authentication token"""
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": "testpass"}
            )
            if response.status == 200:
                data = await response.json()
                return data.get("access_token")
        return None
    
    async def test_token_randomness(self):
        """Test token randomness"""
        tokens = []
        for _ in range(10):
            token = await self.get_test_token("testuser")
            if token:
                tokens.append(token)
        
        # Check for patterns
        if len(set(tokens)) < len(tokens):
            self.add_vulnerability(
                test_name="Predictable Token Generation",
                severity=VulnerabilitySeverity.CRITICAL,
                description="Authentication tokens are predictable",
                affected_endpoint="/api/auth/login",
                evidence={"unique_tokens": len(set(tokens)), "total": len(tokens)},
                remediation="Use cryptographically secure random token generation",
                cwe_id="CWE-330"
            )
    
    async def test_privilege_escalation(self):
        """Test for privilege escalation"""
        # Implementation for vertical privilege escalation tests
        pass
    
    async def test_command_injection(self):
        """Test for command injection"""
        # Implementation for command injection tests
        pass
    
    async def test_ldap_injection(self):
        """Test for LDAP injection"""
        # Implementation for LDAP injection tests
        pass
    
    async def test_stored_xss(self):
        """Test for stored XSS"""
        # Implementation for stored XSS tests
        pass
    
    async def test_api_versioning(self):
        """Test API versioning security"""
        # Implementation for API versioning tests
        pass
    
    async def test_api_rate_limiting(self):
        """Test API-specific rate limiting"""
        # Implementation for API rate limiting tests
        pass
    
    async def test_session_fixation(self):
        """Test for session fixation"""
        # Implementation for session fixation tests
        pass
    
    async def test_file_upload_path_traversal(self):
        """Test for path traversal in file uploads"""
        # Implementation for file upload path traversal tests
        pass
    
    def add_vulnerability(
        self,
        test_name: str,
        severity: VulnerabilitySeverity,
        description: str,
        affected_endpoint: str,
        evidence: Dict[str, Any],
        remediation: str,
        cwe_id: str = None,
        owasp_category: str = None
    ):
        """Add vulnerability to findings"""
        vuln = SecurityVulnerability(
            test_name=test_name,
            severity=severity,
            description=description,
            affected_endpoint=affected_endpoint,
            evidence=evidence,
            remediation=remediation,
            cwe_id=cwe_id,
            owasp_category=owasp_category
        )
        self.vulnerabilities.append(vuln)
        
        # Print immediate feedback
        severity_symbol = {
            VulnerabilitySeverity.CRITICAL: "🔴",
            VulnerabilitySeverity.HIGH: "🟠",
            VulnerabilitySeverity.MEDIUM: "🟡",
            VulnerabilitySeverity.LOW: "🔵",
            VulnerabilitySeverity.INFO: "⚪"
        }
        
        print(f"{severity_symbol[severity]} {test_name}: {description}")
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Count vulnerabilities by severity
        severity_counts = {
            VulnerabilitySeverity.CRITICAL: 0,
            VulnerabilitySeverity.HIGH: 0,
            VulnerabilitySeverity.MEDIUM: 0,
            VulnerabilitySeverity.LOW: 0,
            VulnerabilitySeverity.INFO: 0
        }
        
        for vuln in self.vulnerabilities:
            severity_counts[vuln.severity] += 1
        
        # Calculate risk score
        risk_score = (
            severity_counts[VulnerabilitySeverity.CRITICAL] * 10 +
            severity_counts[VulnerabilitySeverity.HIGH] * 5 +
            severity_counts[VulnerabilitySeverity.MEDIUM] * 3 +
            severity_counts[VulnerabilitySeverity.LOW] * 1
        )
        
        report = {
            "audit_date": datetime.utcnow().isoformat(),
            "platform_url": self.base_url,
            "total_vulnerabilities": len(self.vulnerabilities),
            "severity_summary": {
                "critical": severity_counts[VulnerabilitySeverity.CRITICAL],
                "high": severity_counts[VulnerabilitySeverity.HIGH],
                "medium": severity_counts[VulnerabilitySeverity.MEDIUM],
                "low": severity_counts[VulnerabilitySeverity.LOW],
                "info": severity_counts[VulnerabilitySeverity.INFO]
            },
            "risk_score": risk_score,
            "risk_level": self._calculate_risk_level(risk_score),
            "vulnerabilities": [vuln.to_dict() for vuln in self.vulnerabilities],
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        with open("security_audit_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*60)
        print("SECURITY AUDIT SUMMARY")
        print("="*60)
        print(f"Total Vulnerabilities: {report['total_vulnerabilities']}")
        print(f"Critical: {report['severity_summary']['critical']}")
        print(f"High: {report['severity_summary']['high']}")
        print(f"Medium: {report['severity_summary']['medium']}")
        print(f"Low: {report['severity_summary']['low']}")
        print(f"Info: {report['severity_summary']['info']}")
        print(f"\nRisk Score: {report['risk_score']}")
        print(f"Risk Level: {report['risk_level']}")
        print(f"\nFull report saved to: security_audit_report.json")
        
        return report
    
    def _calculate_risk_level(self, risk_score: int) -> str:
        """Calculate overall risk level"""
        if risk_score >= 50:
            return "CRITICAL"
        elif risk_score >= 30:
            return "HIGH"
        elif risk_score >= 15:
            return "MEDIUM"
        elif risk_score >= 5:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Check for critical vulnerabilities
        critical_count = sum(
            1 for v in self.vulnerabilities 
            if v.severity == VulnerabilitySeverity.CRITICAL
        )
        
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical vulnerabilities immediately"
            )
        
        # Check for common vulnerability patterns
        vuln_types = {}
        for vuln in self.vulnerabilities:
            vuln_type = vuln.cwe_id or vuln.test_name
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1
        
        # Add specific recommendations
        if "CWE-89" in vuln_types:  # SQL Injection
            recommendations.append(
                "Implement parameterized queries throughout the application"
            )
        
        if "CWE-79" in vuln_types:  # XSS
            recommendations.append(
                "Implement comprehensive input validation and output encoding"
            )
        
        if "CWE-352" in vuln_types:  # CSRF
            recommendations.append(
                "Implement CSRF tokens for all state-changing operations"
            )
        
        # General recommendations
        recommendations.extend([
            "Conduct regular security audits and penetration testing",
            "Implement a Web Application Firewall (WAF)",
            "Enable comprehensive security logging and monitoring",
            "Establish a vulnerability disclosure program",
            "Provide security training for development team"
        ])
        
        return recommendations


# Penetration testing class
class PenetrationTester(SecurityAuditor):
    """Advanced penetration testing capabilities"""
    
    async def run_penetration_test(self) -> Dict[str, Any]:
        """Run comprehensive penetration test"""
        print("Starting penetration testing...")
        
        # Run basic security audit first
        await self.run_full_audit()
        
        # Additional penetration tests
        await self.test_authentication_bypass()
        await self.test_directory_traversal()
        await self.test_xxe_injection()
        await self.test_server_side_request_forgery()
        await self.test_insecure_deserialization()
        await self.test_race_conditions()
        await self.test_clickjacking()
        await self.test_subdomain_takeover()
        
        # Generate penetration test report
        return self.generate_pentest_report()
    
    async def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        print("\n[*] Testing Authentication Bypass...")
        
        # Test JWT none algorithm
        async with aiohttp.ClientSession() as session:
            # Create JWT with none algorithm
            header = {"alg": "none", "typ": "JWT"}
            payload = {"sub": "admin", "exp": datetime.utcnow().timestamp() + 3600}
            
            token = (
                base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=") +
                "." +
                base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=") +
                "."
            )
            
            headers = {"Authorization": f"Bearer {token}"}
            response = await session.get(
                f"{self.base_url}/api/admin/users",
                headers=headers
            )
            
            if response.status == 200:
                self.add_vulnerability(
                    test_name="JWT None Algorithm Bypass",
                    severity=VulnerabilitySeverity.CRITICAL,
                    description="JWT validation accepts 'none' algorithm",
                    affected_endpoint="/api/*",
                    evidence={"token": token[:50] + "..."},
                    remediation="Reject JWTs with 'none' algorithm",
                    cwe_id="CWE-345"
                )
    
    async def test_directory_traversal(self):
        """Test for directory traversal vulnerabilities"""
        print("\n[*] Testing Directory Traversal...")
        
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\win.ini",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in traversal_payloads:
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                    f"{self.base_url}/api/files?path={payload}"
                )
                
                if response.status == 200:
                    content = await response.text()
                    if "root:" in content or "[fonts]" in content:
                        self.add_vulnerability(
                            test_name="Directory Traversal",
                            severity=VulnerabilitySeverity.CRITICAL,
                            description="Directory traversal allows file system access",
                            affected_endpoint="/api/files",
                            evidence={"payload": payload},
                            remediation="Validate and sanitize file paths",
                            cwe_id="CWE-22"
                        )
    
    async def test_xxe_injection(self):
        """Test for XXE injection"""
        print("\n[*] Testing XXE Injection...")
        
        xxe_payload = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
        <data>&xxe;</data>"""
        
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/xml"}
            response = await session.post(
                f"{self.base_url}/api/data/import",
                data=xxe_payload,
                headers=headers
            )
            
            if response.status == 200:
                content = await response.text()
                if "root:" in content:
                    self.add_vulnerability(
                        test_name="XXE Injection",
                        severity=VulnerabilitySeverity.CRITICAL,
                        description="XML parser vulnerable to XXE injection",
                        affected_endpoint="/api/data/import",
                        evidence={"response_snippet": content[:100]},
                        remediation="Disable external entity processing",
                        cwe_id="CWE-611"
                    )
    
    def generate_pentest_report(self) -> Dict[str, Any]:
        """Generate penetration test report"""
        base_report = self.generate_audit_report()
        
        # Add penetration test specific information
        base_report["test_type"] = "Penetration Test"
        base_report["test_methodology"] = "OWASP Testing Guide v4"
        base_report["executive_summary"] = self._generate_executive_summary()
        
        return base_report
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary"""
        critical_count = sum(
            1 for v in self.vulnerabilities 
            if v.severity == VulnerabilitySeverity.CRITICAL
        )
        
        high_count = sum(
            1 for v in self.vulnerabilities 
            if v.severity == VulnerabilitySeverity.HIGH
        )
        
        if critical_count > 0:
            return (
                f"The penetration test identified {critical_count} CRITICAL and "
                f"{high_count} HIGH severity vulnerabilities that pose immediate "
                "risk to the platform. Immediate remediation is strongly recommended."
            )
        elif high_count > 0:
            return (
                f"The penetration test identified {high_count} HIGH severity "
                "vulnerabilities that should be addressed promptly to ensure "
                "platform security."
            )
        else:
            return (
                "The penetration test identified several medium and low severity "
                "vulnerabilities. While not critical, these should be addressed "
                "as part of regular security maintenance."
            )


# Test execution
async def run_security_tests():
    """Run all security tests"""
    # Run security audit
    auditor = SecurityAuditor()
    audit_report = await auditor.run_full_audit()
    
    # Run penetration test
    pentester = PenetrationTester()
    pentest_report = await pentester.run_penetration_test()
    
    return {
        "audit_report": audit_report,
        "pentest_report": pentest_report
    }


if __name__ == "__main__":
    asyncio.run(run_security_tests())