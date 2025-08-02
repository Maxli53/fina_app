"""
API endpoints for AI Advisory services
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.services.ai.gpt_advisor import (
    AIAdvisorService, 
    AnalysisContext, 
    AdvisorRole,
    AdvisoryResponse
)
from app.services.auth import get_current_user
from app.models.auth import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-advisor", tags=["AI Advisory"])

# Initialize AI advisor service
ai_advisor = None

def get_ai_advisor():
    """Get AI advisor instance"""
    global ai_advisor
    if ai_advisor is None:
        # Initialize with API key from environment
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="AI service not configured")
        ai_advisor = AIAdvisorService(provider="openai", api_key=api_key)
    return ai_advisor


# Request/Response models
class AnalysisConfigRequest(BaseModel):
    """Request for analysis configuration recommendations"""
    symbols: List[str] = Field(..., description="Symbols to analyze")
    objectives: List[str] = Field(..., description="Analysis objectives")
    constraints: Dict[str, Any] = Field(default={}, description="Constraints and preferences")


class ContextualAnalysisRequest(BaseModel):
    """Request for contextual analysis based on current configuration"""
    context: Dict[str, Any] = Field(..., description="Current analysis context including active analysis type and configuration")
    requestType: str = Field("configuration_optimization", description="Type of contextual analysis requested")


class AnalysisInterpretationRequest(BaseModel):
    """Request for analysis results interpretation"""
    results: Dict[str, Any] = Field(..., description="Analysis results to interpret")
    context: Dict[str, Any] = Field(..., description="Analysis context")
    role: str = Field("quantitative_analyst", description="Advisory role")


class StrategyRecommendationRequest(BaseModel):
    """Request for trading strategy recommendations"""
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    market_data: Dict[str, Any] = Field(..., description="Current market data")
    risk_profile: Dict[str, Any] = Field(..., description="Risk profile and constraints")


class PortfolioRiskRequest(BaseModel):
    """Request for portfolio risk analysis"""
    portfolio: List[Dict[str, Any]] = Field(..., description="Portfolio positions")
    market_conditions: Dict[str, Any] = Field(..., description="Current market conditions")
    scenarios: Optional[List[Dict]] = Field(None, description="Stress test scenarios")


class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization"""
    current_portfolio: List[Dict[str, Any]] = Field(..., description="Current portfolio")
    constraints: Dict[str, Any] = Field(..., description="Optimization constraints")
    objectives: List[str] = Field(..., description="Optimization objectives")


class MarketRegimeRequest(BaseModel):
    """Request for market regime analysis"""
    market_data: Dict[str, Any] = Field(..., description="Market data")
    economic_indicators: Dict[str, Any] = Field(..., description="Economic indicators")
    sentiment_data: Optional[Dict] = Field(None, description="Sentiment data")


class ConsultationRequest(BaseModel):
    """Request for interactive consultation"""
    question: str = Field(..., description="User question")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class StrategyValidationRequest(BaseModel):
    """Request for strategy validation"""
    strategy_code: str = Field(..., description="Strategy code or logic")
    strategy_description: str = Field(..., description="Strategy description")


# API Endpoints
@router.post("/contextual-analysis")
async def get_contextual_analysis(
    request: ContextualAnalysisRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get contextual AI analysis based on current configuration"""
    try:
        context = request.context
        active_analysis = context.get("activeAnalysis", "idtxl")
        
        # Build comprehensive prompt for AI with full configuration details
        config_details = {
            "active_analysis": active_analysis,
            "symbols": context.get("symbols", []),
            "date_range": context.get("dateRange", {}),
            "configuration": {}
        }
        
        # Include the specific configuration based on active analysis
        if active_analysis == "idtxl" and context.get("idtxl"):
            config_details["configuration"] = context["idtxl"]
        elif active_analysis == "ml" and context.get("ml"):
            config_details["configuration"] = context["ml"]
        elif active_analysis == "neural" and context.get("neural"):
            config_details["configuration"] = context["neural"]
        elif active_analysis == "integrated" and context.get("integrated"):
            config_details["configuration"] = context["integrated"]
        
        # Create analysis context with full details
        analysis_context = AnalysisContext(
            symbols=context.get("symbols", []),
            timeframe=context.get("dateRange", {}).get("start", "1d"),
            analysis_type=active_analysis,
            objectives=["optimize_configuration", "maximize_signal_quality", "financial_time_series_prediction"],
            risk_tolerance="moderate",
            capital=100000.0,
            additional_context=config_details  # Pass full configuration to AI
        )
        
        # Get AI recommendations based on the full context
        response = await advisor.advisor.get_analysis_recommendations(
            context=analysis_context,
            role=AdvisorRole.QUANT_ANALYST
        )
        
        # The AI service should now return optimal configuration based on its analysis
        # Extract it from the response if available, otherwise use the AI's supporting data
        optimal_config = response.supporting_data.get("optimal_configuration", {})
        
        # If AI didn't provide optimal config in supporting data, try to extract from recommendations
        if not optimal_config and response.recommendations:
            # Look for configuration-related recommendations
            for rec in response.recommendations:
                if "optimal" in rec.get("text", "").lower() or "configuration" in rec.get("text", "").lower():
                    # AI might have included config in recommendation text
                    optimal_config = rec.get("config", {})
                    break
        
        return {
            "status": "success",
            "data": {
                "recommendations": response.recommendations,
                "insights": response.insights,
                "warnings": response.warnings,
                "confidence_level": response.confidence_level,
                "optimal_configuration": optimal_config,
                "ai_context": f"Analysis for {active_analysis} with {len(context.get('symbols', []))} symbols"
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error in contextual analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/config")
async def get_analysis_configuration(
    request: AnalysisConfigRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get AI-recommended analysis configuration"""
    try:
        config = await advisor.get_smart_analysis_config(
            symbols=request.symbols,
            objectives=request.objectives,
            constraints=request.constraints
        )
        
        return {
            "status": "success",
            "configuration": config,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/interpret")
async def interpret_analysis_results(
    request: AnalysisInterpretationRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get PhD-level interpretation of analysis results"""
    try:
        # Create context
        context = AnalysisContext(
            symbols=request.context.get("symbols", []),
            timeframe=request.context.get("timeframe", "1d"),
            analysis_type=request.context.get("analysis_type", "comprehensive"),
            objectives=request.context.get("objectives", []),
            risk_tolerance=request.context.get("risk_tolerance", "moderate"),
            capital=request.context.get("capital", 100000)
        )
        
        # Get role
        role = AdvisorRole[request.role.upper()]
        
        # Get interpretation
        response = await advisor.advisor.interpret_analysis_results(
            results=request.results,
            context=context,
            role=role
        )
        
        return {
            "status": "success",
            "interpretation": {
                "role": response.role.value,
                "recommendations": response.recommendations,
                "insights": response.insights,
                "warnings": response.warnings,
                "confidence_level": response.confidence_level
            },
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error interpreting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/recommend")
async def recommend_trading_strategy(
    request: StrategyRecommendationRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get comprehensive trading strategy recommendations"""
    try:
        response = await advisor.advisor.recommend_trading_strategy(
            analysis_results=request.analysis_results,
            market_data=request.market_data,
            risk_profile=request.risk_profile
        )
        
        return {
            "status": "success",
            "strategy": {
                "recommendations": response.recommendations,
                "insights": response.insights,
                "warnings": response.warnings,
                "confidence_level": response.confidence_level,
                "implementation_guide": response.supporting_data.get("implementation", {})
            },
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error recommending strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/risk-analysis")
async def analyze_portfolio_risk(
    request: PortfolioRiskRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get PhD-level portfolio risk analysis"""
    try:
        response = await advisor.advisor.analyze_portfolio_risk(
            portfolio=request.portfolio,
            market_conditions=request.market_conditions,
            scenarios=request.scenarios
        )
        
        return {
            "status": "success",
            "risk_analysis": {
                "recommendations": response.recommendations,
                "risk_factors": response.insights,
                "warnings": response.warnings,
                "confidence_level": response.confidence_level,
                "mitigation_strategies": [
                    rec for rec in response.recommendations 
                    if "hedge" in rec["text"].lower() or "mitigate" in rec["text"].lower()
                ]
            },
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/optimize")
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Get portfolio optimization recommendations"""
    try:
        response = await advisor.advisor.optimize_portfolio_allocation(
            current_portfolio=request.current_portfolio,
            constraints=request.constraints,
            objectives=request.objectives
        )
        
        return {
            "status": "success",
            "optimization": {
                "recommendations": response.recommendations,
                "insights": response.insights,
                "warnings": response.warnings,
                "confidence_level": response.confidence_level,
                "allocation_changes": response.supporting_data.get("allocation_changes", {})
            },
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market/regime")
async def analyze_market_regime(
    request: MarketRegimeRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Analyze current market regime"""
    try:
        response = await advisor.advisor.analyze_market_regime(
            market_data=request.market_data,
            economic_indicators=request.economic_indicators,
            sentiment_data=request.sentiment_data
        )
        
        return {
            "status": "success",
            "regime_analysis": {
                "current_regime": response.supporting_data.get("regime", "unknown"),
                "insights": response.insights,
                "warnings": response.warnings,
                "recommendations": response.recommendations,
                "confidence_level": response.confidence_level
            },
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_with_advisor(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Chat with AI advisor with full context"""
    try:
        message = request.get("message", "")
        context = request.get("context", {})
        
        # Extract analysis context from request
        analysis_context = context.get("analysis_context", {})
        current_tab = context.get("current_tab", "analysis")
        conversation_history = context.get("conversation_history", [])
        
        # Build comprehensive context for AI
        comprehensive_context = {
            "current_tab": current_tab,
            "analysis_configuration": analysis_context,
            "conversation_history": conversation_history
        }
        
        response = await advisor.advisor.interactive_consultation(
            user_id=str(current_user.id),
            question=message,
            context=comprehensive_context
        )
        
        return {
            "status": "success",
            "response": response,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consult")
async def interactive_consultation(
    request: ConsultationRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Interactive consultation with AI advisor (legacy endpoint)"""
    try:
        response = await advisor.advisor.interactive_consultation(
            user_id=str(current_user.id),
            question=request.question,
            context=request.context
        )
        
        return {
            "status": "success",
            "response": response,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error in consultation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/validate")
async def validate_strategy(
    request: StrategyValidationRequest,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor)
) -> Dict[str, Any]:
    """Validate trading strategy logic"""
    try:
        validation = await advisor.advisor.validate_strategy_logic(
            strategy_code=request.strategy_code,
            strategy_description=request.strategy_description
        )
        
        return {
            "status": "success",
            "validation": validation,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error validating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/generate")
async def generate_research_report(
    topic: str,
    data: Dict[str, Any],
    analysis_type: str,
    current_user: User = Depends(get_current_user),
    advisor: AIAdvisorService = Depends(get_ai_advisor),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Generate professional research report"""
    try:
        # Generate report ID
        report_id = f"report_{current_user.id}_{datetime.utcnow().timestamp()}"
        
        # Start report generation in background
        background_tasks.add_task(
            generate_report_background,
            advisor,
            topic,
            data,
            analysis_type,
            report_id
        )
        
        return {
            "status": "success",
            "message": "Report generation started",
            "report_id": report_id,
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting report generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_report_background(
    advisor: AIAdvisorService,
    topic: str,
    data: Dict[str, Any],
    analysis_type: str,
    report_id: str
):
    """Generate report in background"""
    try:
        report = await advisor.advisor.generate_research_report(
            topic=topic,
            data=data,
            analysis_type=analysis_type
        )
        
        # Save report (implement storage logic)
        # For now, just log success
        logger.info(f"Report {report_id} generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating report {report_id}: {e}")


@router.get("/report/{report_id}")
async def get_report_status(
    report_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get report generation status"""
    # Implement report status checking
    return {
        "status": "success",
        "report_id": report_id,
        "generation_status": "completed",  # or "in_progress", "failed"
        "download_url": f"/api/ai/report/{report_id}/download"
    }


# Health check
@router.get("/health")
async def ai_health_check() -> Dict[str, Any]:
    """Check AI service health"""
    try:
        advisor = get_ai_advisor()
        return {
            "status": "healthy",
            "service": "AI Advisory",
            "provider": "OpenAI",
            "features": [
                "analysis_configuration",
                "results_interpretation",
                "strategy_recommendation",
                "portfolio_optimization",
                "risk_analysis",
                "market_regime_analysis",
                "interactive_consultation",
                "strategy_validation",
                "report_generation"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }