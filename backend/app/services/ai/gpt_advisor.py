"""
GPT Integration for PhD-level Financial Analysis Advisory
Provides intelligent assistance for analysis configuration, results interpretation, and trading recommendations
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from openai import AsyncOpenAI
import anthropic

logger = logging.getLogger(__name__)


class AdvisorRole(Enum):
    """Different advisory roles for specialized assistance"""
    QUANT_ANALYST = "quantitative_analyst"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_STRATEGIST = "portfolio_strategist"
    MARKET_RESEARCHER = "market_researcher"
    TRADING_ADVISOR = "trading_advisor"


@dataclass
class AnalysisContext:
    """Context for analysis advisory"""
    symbols: List[str]
    timeframe: str
    analysis_type: str
    objectives: List[str]
    risk_tolerance: str
    capital: float
    current_positions: Optional[List[Dict]] = None
    market_conditions: Optional[Dict] = None
    historical_performance: Optional[Dict] = None


@dataclass
class AdvisoryResponse:
    """Structured advisory response"""
    role: AdvisorRole
    recommendations: List[Dict[str, Any]]
    insights: List[str]
    warnings: List[str]
    confidence_level: float
    supporting_data: Dict[str, Any]
    timestamp: datetime


class GPTAdvisor:
    """PhD-level AI advisor for quantitative finance"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.system_prompts = self._initialize_system_prompts()
        
    def _initialize_system_prompts(self) -> Dict[AdvisorRole, str]:
        """Initialize specialized system prompts for each role"""
        return {
            AdvisorRole.QUANT_ANALYST: """You are a PhD-level quantitative analyst with expertise in:
                - Information theory and transfer entropy analysis (IDTxl)
                - Advanced statistical methods and time series analysis
                - Machine learning for financial markets
                - Neural network architectures for price prediction
                - Cross-asset correlation and causality analysis
                
                Provide detailed, mathematically rigorous advice on:
                - Optimal parameter selection for IDTxl analysis
                - Feature engineering for ML models
                - Model validation and backtesting strategies
                - Statistical significance and p-value interpretation
                - Risk-adjusted performance metrics
                
                Always explain complex concepts clearly and suggest practical implementations.""",
            
            AdvisorRole.RISK_MANAGER: """You are a PhD-level risk management expert specializing in:
                - Value at Risk (VaR) and Conditional VaR
                - Portfolio risk decomposition
                - Stress testing and scenario analysis
                - Correlation breakdowns and tail risk
                - Regulatory compliance (Basel III, MiFID II)
                
                Focus on:
                - Identifying hidden risks and correlations
                - Recommending position limits and stop-losses
                - Suggesting hedging strategies
                - Analyzing worst-case scenarios
                - Ensuring regulatory compliance""",
            
            AdvisorRole.PORTFOLIO_STRATEGIST: """You are a PhD-level portfolio strategist with expertise in:
                - Modern Portfolio Theory and beyond
                - Factor investing and smart beta
                - Dynamic asset allocation
                - Portfolio optimization techniques
                - Behavioral finance considerations
                
                Provide strategic advice on:
                - Optimal portfolio construction
                - Rebalancing strategies
                - Factor exposure management
                - Tax-efficient investing
                - Long-term wealth preservation""",
            
            AdvisorRole.MARKET_RESEARCHER: """You are a PhD-level market researcher specializing in:
                - Market microstructure analysis
                - Sentiment analysis and alternative data
                - Macroeconomic indicators
                - Cross-market relationships
                - Emerging market trends
                
                Focus on:
                - Identifying market regimes
                - Analyzing structural breaks
                - Predicting volatility clusters
                - Understanding liquidity dynamics
                - Spotting arbitrage opportunities""",
            
            AdvisorRole.TRADING_ADVISOR: """You are a PhD-level algorithmic trading expert with knowledge of:
                - High-frequency trading strategies
                - Market making algorithms
                - Order execution optimization
                - Transaction cost analysis
                - Trading signal generation
                
                Advise on:
                - Optimal entry and exit points
                - Order sizing and timing
                - Slippage minimization
                - Alpha generation strategies
                - Risk-reward optimization"""
        }
    
    async def get_analysis_recommendations(
        self,
        context: AnalysisContext,
        role: AdvisorRole = AdvisorRole.QUANT_ANALYST
    ) -> AdvisoryResponse:
        """Get PhD-level recommendations for analysis configuration"""
        
        # Prepare context for AI
        context_str = self._prepare_context_string(context)
        
        # Create conversation
        messages = [
            {"role": "system", "content": self.system_prompts[role]},
            {"role": "user", "content": f"""
            Analysis Context:
            {context_str}
            
            Please provide PhD-level recommendations for:
            1. Optimal analysis parameters and configuration
            2. Key insights to look for in the results
            3. Potential pitfalls and how to avoid them
            4. Integration with other analysis methods
            5. Trading strategy implications
            
            Be specific and quantitative where possible.
            """}
        ]
        
        # Get AI response
        response = await self._get_ai_response(messages)
        
        # Parse and structure response
        return self._parse_advisory_response(response, role)
    
    async def interpret_analysis_results(
        self,
        results: Dict[str, Any],
        context: AnalysisContext,
        role: AdvisorRole = AdvisorRole.QUANT_ANALYST
    ) -> AdvisoryResponse:
        """Interpret analysis results with PhD-level insights"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[role]},
            {"role": "user", "content": f"""
            Analysis Results:
            {json.dumps(results, indent=2)}
            
            Context:
            {self._prepare_context_string(context)}
            
            Please provide:
            1. PhD-level interpretation of these results
            2. Statistical significance assessment
            3. Hidden patterns or anomalies
            4. Actionable trading signals
            5. Risk considerations
            6. Confidence level in findings
            
            Focus on insights that only an expert would notice.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return self._parse_advisory_response(response, role)
    
    async def recommend_trading_strategy(
        self,
        analysis_results: Dict[str, Any],
        market_data: Dict[str, Any],
        risk_profile: Dict[str, Any]
    ) -> AdvisoryResponse:
        """Recommend comprehensive trading strategy based on analysis"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[AdvisorRole.TRADING_ADVISOR]},
            {"role": "user", "content": f"""
            Analysis Results:
            {json.dumps(analysis_results, indent=2)}
            
            Market Data:
            {json.dumps(market_data, indent=2)}
            
            Risk Profile:
            {json.dumps(risk_profile, indent=2)}
            
            Design a comprehensive trading strategy including:
            1. Entry and exit rules
            2. Position sizing formula
            3. Risk management parameters
            4. Expected performance metrics
            5. Market regime considerations
            6. Implementation timeline
            
            Provide specific, quantitative recommendations.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return self._parse_advisory_response(response, AdvisorRole.TRADING_ADVISOR)
    
    async def analyze_portfolio_risk(
        self,
        portfolio: List[Dict[str, Any]],
        market_conditions: Dict[str, Any],
        scenarios: Optional[List[Dict]] = None
    ) -> AdvisoryResponse:
        """Provide PhD-level portfolio risk analysis"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[AdvisorRole.RISK_MANAGER]},
            {"role": "user", "content": f"""
            Portfolio Positions:
            {json.dumps(portfolio, indent=2)}
            
            Market Conditions:
            {json.dumps(market_conditions, indent=2)}
            
            Stress Scenarios:
            {json.dumps(scenarios or [], indent=2)}
            
            Provide comprehensive risk analysis:
            1. Hidden correlation risks
            2. Tail risk assessment
            3. Liquidity risk analysis
            4. Scenario impact analysis
            5. Hedging recommendations
            6. Position limit suggestions
            
            Use advanced risk metrics and models.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return self._parse_advisory_response(response, AdvisorRole.RISK_MANAGER)
    
    async def optimize_portfolio_allocation(
        self,
        current_portfolio: List[Dict[str, Any]],
        constraints: Dict[str, Any],
        objectives: List[str]
    ) -> AdvisoryResponse:
        """Optimize portfolio allocation with advanced techniques"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[AdvisorRole.PORTFOLIO_STRATEGIST]},
            {"role": "user", "content": f"""
            Current Portfolio:
            {json.dumps(current_portfolio, indent=2)}
            
            Constraints:
            {json.dumps(constraints, indent=2)}
            
            Objectives:
            {json.dumps(objectives, indent=2)}
            
            Provide portfolio optimization recommendations:
            1. Optimal allocation weights
            2. Rebalancing strategy
            3. Factor exposure analysis
            4. Expected risk-return profile
            5. Implementation roadmap
            6. Tax optimization considerations
            
            Use advanced portfolio theory and practical insights.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return self._parse_advisory_response(response, AdvisorRole.PORTFOLIO_STRATEGIST)
    
    async def analyze_market_regime(
        self,
        market_data: Dict[str, Any],
        economic_indicators: Dict[str, Any],
        sentiment_data: Optional[Dict] = None
    ) -> AdvisoryResponse:
        """Identify current market regime and implications"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[AdvisorRole.MARKET_RESEARCHER]},
            {"role": "user", "content": f"""
            Market Data:
            {json.dumps(market_data, indent=2)}
            
            Economic Indicators:
            {json.dumps(economic_indicators, indent=2)}
            
            Sentiment Data:
            {json.dumps(sentiment_data or {}, indent=2)}
            
            Analyze the current market regime:
            1. Regime identification (bull/bear/transition)
            2. Volatility regime analysis
            3. Correlation regime assessment
            4. Liquidity conditions
            5. Sector rotation signals
            6. Forward-looking implications
            
            Provide quantitative evidence for conclusions.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return self._parse_advisory_response(response, AdvisorRole.MARKET_RESEARCHER)
    
    async def generate_research_report(
        self,
        topic: str,
        data: Dict[str, Any],
        analysis_type: str
    ) -> str:
        """Generate professional research report"""
        
        messages = [
            {"role": "system", "content": """You are a PhD-level financial researcher 
            writing for institutional investors. Create professional, detailed research 
            reports with academic rigor but practical applicability."""},
            {"role": "user", "content": f"""
            Generate a professional research report on:
            Topic: {topic}
            Analysis Type: {analysis_type}
            
            Data and Findings:
            {json.dumps(data, indent=2)}
            
            Structure the report with:
            1. Executive Summary
            2. Methodology
            3. Key Findings
            4. Statistical Analysis
            5. Risk Considerations
            6. Investment Implications
            7. Appendix with technical details
            
            Use professional financial writing style.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        return response
    
    async def interactive_consultation(
        self,
        user_id: str,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Interactive Q&A with the AI advisor"""
        
        # Maintain conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Add context if provided
        context_message = ""
        if context:
            context_message = f"\nCurrent Context: {json.dumps(context, indent=2)}"
        
        # Prepare messages with history
        messages = [
            {"role": "system", "content": """You are a PhD-level financial advisor 
            providing expert consultation. Be helpful, specific, and quantitative. 
            Explain complex concepts clearly."""}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history[user_id][-10:])  # Last 10 messages
        
        # Add current question
        messages.append({
            "role": "user", 
            "content": f"{question}{context_message}"
        })
        
        # Get response
        response = await self._get_ai_response(messages)
        
        # Update history
        self.conversation_history[user_id].append({"role": "user", "content": question})
        self.conversation_history[user_id].append({"role": "assistant", "content": response})
        
        return response
    
    async def validate_strategy_logic(
        self,
        strategy_code: str,
        strategy_description: str
    ) -> Dict[str, Any]:
        """Validate trading strategy logic and suggest improvements"""
        
        messages = [
            {"role": "system", "content": self.system_prompts[AdvisorRole.TRADING_ADVISOR]},
            {"role": "user", "content": f"""
            Review this trading strategy:
            
            Description: {strategy_description}
            
            Code/Logic:
            {strategy_code}
            
            Please:
            1. Identify logical flaws or bugs
            2. Assess statistical validity
            3. Evaluate risk management
            4. Suggest improvements
            5. Estimate expected performance
            6. Identify edge cases
            
            Be specific about issues and solutions.
            """}
        ]
        
        response = await self._get_ai_response(messages)
        
        # Parse validation results
        return {
            "valid": "critical flaw" not in response.lower(),
            "issues": self._extract_issues(response),
            "improvements": self._extract_improvements(response),
            "risk_assessment": self._extract_risk_assessment(response),
            "full_analysis": response
        }
    
    # Helper methods
    async def _get_ai_response(self, messages: List[Dict]) -> str:
        """Get response from AI model"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI API error: {e}")
            raise
    
    def _prepare_context_string(self, context: AnalysisContext) -> str:
        """Prepare context for AI consumption"""
        return f"""
        Symbols: {', '.join(context.symbols)}
        Timeframe: {context.timeframe}
        Analysis Type: {context.analysis_type}
        Objectives: {', '.join(context.objectives)}
        Risk Tolerance: {context.risk_tolerance}
        Capital: ${context.capital:,.2f}
        Current Positions: {len(context.current_positions or [])} positions
        Market Conditions: {context.market_conditions or 'Normal'}
        """
    
    def _parse_advisory_response(self, response: str, role: AdvisorRole) -> AdvisoryResponse:
        """Parse AI response into structured format"""
        # This is a simplified parser - in production, use more sophisticated NLP
        
        recommendations = []
        insights = []
        warnings = []
        
        # Extract sections (simplified)
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if "recommend" in line.lower():
                current_section = "recommendations"
            elif "insight" in line.lower() or "finding" in line.lower():
                current_section = "insights"
            elif "warning" in line.lower() or "risk" in line.lower():
                current_section = "warnings"
            elif line.startswith(("•", "-", "*", "1.", "2.", "3.")):
                # Add to current section
                cleaned_line = line.lstrip("•-*123456789. ")
                if current_section == "recommendations":
                    recommendations.append({"text": cleaned_line, "priority": "high"})
                elif current_section == "insights":
                    insights.append(cleaned_line)
                elif current_section == "warnings":
                    warnings.append(cleaned_line)
        
        # Calculate confidence based on response characteristics
        confidence = 0.85  # Base confidence
        if "high confidence" in response.lower():
            confidence = 0.95
        elif "moderate confidence" in response.lower():
            confidence = 0.75
        elif "low confidence" in response.lower():
            confidence = 0.60
        
        return AdvisoryResponse(
            role=role,
            recommendations=recommendations,
            insights=insights,
            warnings=warnings,
            confidence_level=confidence,
            supporting_data={"raw_response": response},
            timestamp=datetime.utcnow()
        )
    
    def _extract_issues(self, response: str) -> List[str]:
        """Extract issues from validation response"""
        issues = []
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ["issue", "problem", "flaw", "bug"]):
                issues.append(line.strip())
                # Also get the next line if it's a continuation
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith(("•", "-", "*")):
                    issues[-1] += " " + lines[i + 1].strip()
        
        return issues
    
    def _extract_improvements(self, response: str) -> List[str]:
        """Extract improvements from validation response"""
        improvements = []
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ["improve", "suggest", "enhance", "optimize"]):
                improvements.append(line.strip())
        
        return improvements
    
    def _extract_risk_assessment(self, response: str) -> Dict[str, Any]:
        """Extract risk assessment from response"""
        risk_assessment = {
            "level": "medium",
            "factors": [],
            "mitigation": []
        }
        
        if "high risk" in response.lower():
            risk_assessment["level"] = "high"
        elif "low risk" in response.lower():
            risk_assessment["level"] = "low"
        
        # Extract risk factors and mitigation (simplified)
        lines = response.split('\n')
        for line in lines:
            if "risk" in line.lower() and any(word in line.lower() for word in ["factor", "because", "due to"]):
                risk_assessment["factors"].append(line.strip())
            elif any(word in line.lower() for word in ["mitigate", "reduce risk", "hedge"]):
                risk_assessment["mitigation"].append(line.strip())
        
        return risk_assessment


class ClaudeAdvisor(GPTAdvisor):
    """Claude-based PhD-level advisor (alternative to GPT)"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-opus-20240229"
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.system_prompts = self._initialize_system_prompts()
    
    async def _get_ai_response(self, messages: List[Dict]) -> str:
        """Get response from Claude model"""
        try:
            # Convert to Claude format
            system_message = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_messages = [m for m in messages[1:] if m["role"] == "user"]
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                system=system_message,
                messages=[{"role": "user", "content": user_messages[-1]["content"]}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class AIAdvisorService:
    """Main service for AI advisory functionality"""
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        if provider == "openai":
            self.advisor = GPTAdvisor(api_key)
        elif provider == "anthropic":
            self.advisor = ClaudeAdvisor(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.cache: Dict[str, Any] = {}
        
    async def get_smart_analysis_config(
        self,
        symbols: List[str],
        objectives: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get AI-recommended analysis configuration"""
        
        # Create context
        context = AnalysisContext(
            symbols=symbols,
            timeframe=constraints.get("timeframe", "1d"),
            analysis_type=constraints.get("analysis_type", "comprehensive"),
            objectives=objectives,
            risk_tolerance=constraints.get("risk_tolerance", "moderate"),
            capital=constraints.get("capital", 100000)
        )
        
        # Get recommendations
        response = await self.advisor.get_analysis_recommendations(context)
        
        # Convert to configuration
        config = {
            "idtxl": {
                "enabled": True,
                "parameters": self._extract_idtxl_params(response),
            },
            "ml": {
                "enabled": True,
                "models": self._extract_ml_models(response),
                "features": self._extract_features(response)
            },
            "neural_networks": {
                "enabled": True,
                "architectures": self._extract_nn_architectures(response)
            },
            "risk_controls": self._extract_risk_controls(response),
            "execution_plan": self._extract_execution_plan(response)
        }
        
        return config
    
    def _extract_idtxl_params(self, response: AdvisoryResponse) -> Dict[str, Any]:
        """Extract IDTxl parameters from response"""
        # Default sophisticated parameters
        params = {
            "max_lag": 10,
            "min_lag": 1,
            "cmi_estimator": "JidtGaussianCMI",
            "n_perm_max_stat": 500,
            "n_perm_omnibus": 500,
            "n_perm_mi": 500,
            "alpha_max_stat": 0.05,
            "alpha_omnibus": 0.05,
            "alpha_mi": 0.05,
            "tail": "two",
            "lag_mode": "max",
            "p_value_correction": "fdr_bh"
        }
        
        # Update based on recommendations
        for rec in response.recommendations:
            if "lag" in rec["text"].lower():
                # Extract lag value if mentioned
                import re
                numbers = re.findall(r'\d+', rec["text"])
                if numbers:
                    params["max_lag"] = int(numbers[0])
        
        return params
    
    def _extract_ml_models(self, response: AdvisoryResponse) -> List[str]:
        """Extract recommended ML models"""
        models = ["random_forest", "xgboost"]  # Defaults
        
        model_keywords = {
            "random forest": "random_forest",
            "xgboost": "xgboost",
            "svm": "svm",
            "neural": "neural_network",
            "logistic": "logistic_regression"
        }
        
        for rec in response.recommendations:
            for keyword, model in model_keywords.items():
                if keyword in rec["text"].lower():
                    if model not in models:
                        models.append(model)
        
        return models
    
    def _extract_features(self, response: AdvisoryResponse) -> List[str]:
        """Extract recommended features"""
        features = ["returns", "volume", "volatility"]  # Defaults
        
        feature_keywords = {
            "rsi": "rsi",
            "macd": "macd",
            "bollinger": "bollinger_bands",
            "moving average": "sma",
            "momentum": "momentum",
            "volume ratio": "volume_ratio"
        }
        
        for insight in response.insights:
            for keyword, feature in feature_keywords.items():
                if keyword in insight.lower():
                    if feature not in features:
                        features.append(feature)
        
        return features
    
    def _extract_nn_architectures(self, response: AdvisoryResponse) -> List[Dict[str, Any]]:
        """Extract neural network architectures"""
        architectures = [
            {
                "type": "lstm",
                "layers": [128, 64, 32],
                "dropout": 0.2,
                "learning_rate": 0.001
            }
        ]
        
        # Add more architectures based on recommendations
        if any("transformer" in str(r).lower() for r in response.recommendations):
            architectures.append({
                "type": "transformer",
                "num_heads": 8,
                "hidden_dim": 256,
                "num_layers": 4
            })
        
        return architectures
    
    def _extract_risk_controls(self, response: AdvisoryResponse) -> Dict[str, Any]:
        """Extract risk control parameters"""
        controls = {
            "max_position_size": 0.1,  # 10% max per position
            "stop_loss": 0.02,  # 2% stop loss
            "max_drawdown": 0.15,  # 15% max drawdown
            "var_limit": 0.05  # 5% VaR limit
        }
        
        # Update based on warnings
        if any("conservative" in w.lower() for w in response.warnings):
            controls["max_position_size"] = 0.05
            controls["stop_loss"] = 0.01
        
        return controls
    
    def _extract_execution_plan(self, response: AdvisoryResponse) -> List[Dict[str, Any]]:
        """Extract execution plan steps"""
        plan = [
            {"step": 1, "action": "data_validation", "duration": "5m"},
            {"step": 2, "action": "feature_engineering", "duration": "10m"},
            {"step": 3, "action": "model_training", "duration": "30m"},
            {"step": 4, "action": "backtesting", "duration": "20m"},
            {"step": 5, "action": "risk_analysis", "duration": "15m"}
        ]
        
        return plan