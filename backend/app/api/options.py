"""
API endpoints for Options Pricing and Analysis
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.services.options.options_pricing import (
    OptionsPricingService,
    OptionContract,
    MarketData,
    OptionType,
    ExerciseStyle,
    VolatilitySurface
)
from app.services.auth import get_current_user
from app.models.auth import User
from app.services.data.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options", tags=["Options"])

# Initialize services
pricing_service = OptionsPricingService()
data_service = None  # Will be injected


def get_data_service():
    """Get data service instance"""
    global data_service
    if data_service is None:
        from sqlalchemy.ext.asyncio import AsyncSession
        import redis.asyncio as aioredis
        # Initialize with dummy session/redis for now
        # In production, properly inject these dependencies
        data_service = DataService(None, None)
    return data_service


# Request/Response models
class OptionPricingRequest(BaseModel):
    """Request for option pricing"""
    underlying: str = Field(..., description="Underlying symbol")
    strike: float = Field(..., description="Strike price")
    expiry: str = Field(..., description="Expiry date (YYYY-MM-DD)")
    option_type: str = Field(..., description="Option type (call/put)")
    exercise_style: str = Field("european", description="Exercise style (european/american)")
    
    # Optional market data override
    spot_price: Optional[float] = Field(None, description="Spot price override")
    volatility: Optional[float] = Field(None, description="Volatility override")
    risk_free_rate: Optional[float] = Field(None, description="Risk-free rate override")
    dividend_yield: Optional[float] = Field(0.0, description="Dividend yield")
    
    # Model selection
    model: str = Field("black-scholes", description="Pricing model")
    model_params: Optional[Dict[str, Any]] = Field({}, description="Model-specific parameters")


class ImpliedVolRequest(BaseModel):
    """Request for implied volatility calculation"""
    underlying: str = Field(..., description="Underlying symbol")
    strike: float = Field(..., description="Strike price")
    expiry: str = Field(..., description="Expiry date (YYYY-MM-DD)")
    option_type: str = Field(..., description="Option type (call/put)")
    exercise_style: str = Field("european", description="Exercise style")
    market_price: float = Field(..., description="Market price of option")
    
    # Optional market data
    spot_price: Optional[float] = Field(None, description="Spot price")
    risk_free_rate: Optional[float] = Field(None, description="Risk-free rate")
    dividend_yield: Optional[float] = Field(0.0, description="Dividend yield")


class OptionPosition(BaseModel):
    """Option position for portfolio pricing"""
    underlying: str
    strike: float
    expiry: str
    option_type: str
    exercise_style: str = "european"
    quantity: int
    
    # Optional overrides
    spot_price: Optional[float] = None
    volatility: Optional[float] = None


class PortfolioPricingRequest(BaseModel):
    """Request for portfolio pricing"""
    positions: List[OptionPosition] = Field(..., description="Option positions")
    model: str = Field("black-scholes", description="Pricing model")
    
    # Global market data overrides
    risk_free_rate: Optional[float] = Field(None, description="Risk-free rate")
    correlation_matrix: Optional[List[List[float]]] = Field(None, description="Correlation matrix")


class VolatilitySurfaceRequest(BaseModel):
    """Request for volatility surface"""
    underlying: str = Field(..., description="Underlying symbol")
    expiry_range: str = Field("1M-1Y", description="Expiry range")
    strike_range: str = Field("80%-120%", description="Strike range as % of spot")
    data_source: str = Field("market", description="Data source (market/model)")


class VaRRequest(BaseModel):
    """Request for VaR calculation"""
    positions: List[OptionPosition] = Field(..., description="Option positions")
    confidence_level: float = Field(0.95, description="Confidence level")
    horizon_days: int = Field(1, description="Time horizon in days")
    simulations: int = Field(10000, description="Number of simulations")


# API Endpoints
@router.post("/price")
async def price_option(
    request: OptionPricingRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Price a single option"""
    try:
        # Get market data
        if request.spot_price is None:
            # Fetch current market data
            quote = await data_svc.get_quote(request.underlying)
            spot_price = quote["price"]
        else:
            spot_price = request.spot_price
        
        # Default market parameters if not provided
        if request.volatility is None:
            # Calculate historical volatility or use implied vol
            volatility = 0.25  # Default 25% for now
        else:
            volatility = request.volatility
        
        if request.risk_free_rate is None:
            risk_free_rate = 0.05  # Default 5% for now
        else:
            risk_free_rate = request.risk_free_rate
        
        # Create option contract
        option = OptionContract(
            underlying=request.underlying,
            strike=request.strike,
            expiry=datetime.strptime(request.expiry, "%Y-%m-%d"),
            option_type=OptionType(request.option_type.lower()),
            exercise_style=ExerciseStyle(request.exercise_style.lower()),
            dividend_yield=request.dividend_yield
        )
        
        # Create market data
        market = MarketData(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=request.dividend_yield,
            timestamp=datetime.now()
        )
        
        # Price option
        result = pricing_service.price_option(
            option=option,
            market=market,
            model=request.model,
            **request.model_params
        )
        
        return {
            "status": "success",
            "pricing": {
                "price": result.price,
                "greeks": result.greeks.to_dict(),
                "model": result.model,
                "inputs": {
                    "spot": spot_price,
                    "strike": request.strike,
                    "volatility": volatility,
                    "risk_free_rate": risk_free_rate,
                    "time_to_expiry": (option.expiry - datetime.now()).days / 365.25,
                    "dividend_yield": request.dividend_yield
                },
                "additional_info": result.additional_info
            },
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error pricing option: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/implied-volatility")
async def calculate_implied_volatility(
    request: ImpliedVolRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Calculate implied volatility from market price"""
    try:
        # Get market data
        if request.spot_price is None:
            quote = await data_svc.get_quote(request.underlying)
            spot_price = quote["price"]
        else:
            spot_price = request.spot_price
        
        if request.risk_free_rate is None:
            risk_free_rate = 0.05  # Default
        else:
            risk_free_rate = request.risk_free_rate
        
        # Create option contract
        option = OptionContract(
            underlying=request.underlying,
            strike=request.strike,
            expiry=datetime.strptime(request.expiry, "%Y-%m-%d"),
            option_type=OptionType(request.option_type.lower()),
            exercise_style=ExerciseStyle(request.exercise_style.lower()),
            dividend_yield=request.dividend_yield
        )
        
        # Create market data (with dummy volatility)
        market = MarketData(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            volatility=0.25,  # Dummy value
            dividend_yield=request.dividend_yield
        )
        
        # Calculate implied volatility
        impl_vol = pricing_service.calculate_implied_volatility(
            option=option,
            market_price=request.market_price,
            market=market
        )
        
        # Price option with implied vol to get Greeks
        market.volatility = impl_vol
        pricing_result = pricing_service.price_option(option, market)
        
        return {
            "status": "success",
            "implied_volatility": {
                "value": impl_vol,
                "annualized_percent": impl_vol * 100,
                "market_price": request.market_price,
                "theoretical_price": pricing_result.price,
                "greeks_at_iv": pricing_result.greeks.to_dict()
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/price")
async def price_portfolio(
    request: PortfolioPricingRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Price portfolio of options"""
    try:
        # Default risk-free rate
        risk_free_rate = request.risk_free_rate or 0.05
        
        # Convert positions to option contracts
        options_list = []
        
        for position in request.positions:
            # Get market data for each underlying
            if position.spot_price is None:
                quote = await data_svc.get_quote(position.underlying)
                spot_price = quote["price"]
            else:
                spot_price = position.spot_price
            
            volatility = position.volatility or 0.25  # Default
            
            # Create option contract
            option = OptionContract(
                underlying=position.underlying,
                strike=position.strike,
                expiry=datetime.strptime(position.expiry, "%Y-%m-%d"),
                option_type=OptionType(position.option_type.lower()),
                exercise_style=ExerciseStyle(position.exercise_style.lower())
            )
            
            options_list.append((option, position.quantity))
        
        # For simplicity, use the first underlying's market data
        # In practice, handle multiple underlyings properly
        first_position = request.positions[0]
        if first_position.spot_price is None:
            quote = await data_svc.get_quote(first_position.underlying)
            spot_price = quote["price"]
        else:
            spot_price = first_position.spot_price
        
        market = MarketData(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            volatility=first_position.volatility or 0.25
        )
        
        # Price portfolio
        portfolio_result = pricing_service.price_portfolio(
            options=options_list,
            market=market,
            model=request.model
        )
        
        return {
            "status": "success",
            "portfolio": portfolio_result,
            "summary": {
                "total_positions": len(request.positions),
                "model_used": request.model,
                "risk_free_rate": risk_free_rate
            }
        }
        
    except Exception as e:
        logger.error(f"Error pricing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/var")
async def calculate_var(
    request: VaRRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Calculate Value at Risk for options portfolio"""
    try:
        # Convert positions to option contracts
        options_list = []
        
        # Get market data for first underlying (simplified)
        first_position = request.positions[0]
        quote = await data_svc.get_quote(first_position.underlying)
        
        market = MarketData(
            spot_price=quote["price"],
            risk_free_rate=0.05,  # Default
            volatility=0.25  # Default
        )
        
        for position in request.positions:
            option = OptionContract(
                underlying=position.underlying,
                strike=position.strike,
                expiry=datetime.strptime(position.expiry, "%Y-%m-%d"),
                option_type=OptionType(position.option_type.lower()),
                exercise_style=ExerciseStyle(position.exercise_style.lower())
            )
            options_list.append((option, position.quantity))
        
        # Calculate VaR
        var_result = pricing_service.calculate_var_options(
            options=options_list,
            market=market,
            confidence_level=request.confidence_level,
            horizon_days=request.horizon_days,
            simulations=request.simulations
        )
        
        return {
            "status": "success",
            "var_analysis": var_result,
            "parameters": {
                "confidence_level": request.confidence_level,
                "horizon_days": request.horizon_days,
                "simulations": request.simulations,
                "positions": len(request.positions)
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chains/{symbol}")
async def get_option_chain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Specific expiry date"),
    min_strike: Optional[float] = Query(None, description="Minimum strike"),
    max_strike: Optional[float] = Query(None, description="Maximum strike"),
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Get option chain for a symbol"""
    try:
        # Get current spot price
        quote = await data_svc.get_quote(symbol)
        spot_price = quote["price"]
        
        # Generate sample option chain (in production, fetch from market data provider)
        if min_strike is None:
            min_strike = spot_price * 0.8
        if max_strike is None:
            max_strike = spot_price * 1.2
        
        # Generate strikes
        strikes = [
            round(min_strike + i * 5, 0) 
            for i in range(int((max_strike - min_strike) / 5) + 1)
        ]
        
        # Generate expiries if not specified
        if expiry:
            expiries = [expiry]
        else:
            from datetime import timedelta
            today = datetime.now()
            expiries = [
                (today + timedelta(days=30)).strftime("%Y-%m-%d"),
                (today + timedelta(days=60)).strftime("%Y-%m-%d"),
                (today + timedelta(days=90)).strftime("%Y-%m-%d")
            ]
        
        # Build option chain
        chain = {
            "symbol": symbol,
            "spot_price": spot_price,
            "timestamp": datetime.now(),
            "expiries": {}
        }
        
        for exp in expiries:
            chain["expiries"][exp] = {
                "calls": {},
                "puts": {}
            }
            
            for strike in strikes:
                # Generate sample data (in production, use real market data)
                # Calls
                call_iv = 0.20 + 0.1 * abs(strike - spot_price) / spot_price
                call_option = OptionContract(
                    underlying=symbol,
                    strike=strike,
                    expiry=datetime.strptime(exp, "%Y-%m-%d"),
                    option_type=OptionType.CALL,
                    exercise_style=ExerciseStyle.EUROPEAN
                )
                
                call_market = MarketData(
                    spot_price=spot_price,
                    risk_free_rate=0.05,
                    volatility=call_iv
                )
                
                call_result = pricing_service.price_option(call_option, call_market)
                
                chain["expiries"][exp]["calls"][str(strike)] = {
                    "bid": round(call_result.price * 0.98, 2),
                    "ask": round(call_result.price * 1.02, 2),
                    "last": call_result.price,
                    "volume": 100,
                    "open_interest": 500,
                    "implied_volatility": call_iv,
                    "greeks": call_result.greeks.to_dict()
                }
                
                # Puts
                put_iv = 0.22 + 0.12 * abs(strike - spot_price) / spot_price
                put_option = OptionContract(
                    underlying=symbol,
                    strike=strike,
                    expiry=datetime.strptime(exp, "%Y-%m-%d"),
                    option_type=OptionType.PUT,
                    exercise_style=ExerciseStyle.EUROPEAN
                )
                
                put_market = MarketData(
                    spot_price=spot_price,
                    risk_free_rate=0.05,
                    volatility=put_iv
                )
                
                put_result = pricing_service.price_option(put_option, put_market)
                
                chain["expiries"][exp]["puts"][str(strike)] = {
                    "bid": round(put_result.price * 0.98, 2),
                    "ask": round(put_result.price * 1.02, 2),
                    "last": put_result.price,
                    "volume": 80,
                    "open_interest": 400,
                    "implied_volatility": put_iv,
                    "greeks": put_result.greeks.to_dict()
                }
        
        return {
            "status": "success",
            "chain": chain
        }
        
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility-surface/{symbol}")
async def get_volatility_surface(
    symbol: str,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Get volatility surface for a symbol"""
    try:
        # Get spot price
        quote = await data_svc.get_quote(symbol)
        spot_price = quote["price"]
        
        # Generate sample volatility surface
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 13)
        expiries = [30, 60, 90, 120, 180, 365]  # Days to expiry
        
        # Create volatility grid (smile effect)
        vol_surface = []
        for expiry in expiries:
            vol_row = []
            for strike in strikes:
                moneyness = strike / spot_price
                # Simple volatility smile
                base_vol = 0.20
                smile = 0.1 * (moneyness - 1.0)**2
                term_structure = 0.05 * np.sqrt(expiry / 365)
                vol = base_vol + smile + term_structure
                vol_row.append(vol)
            vol_surface.append(vol_row)
        
        # Create VolatilitySurface object
        vol_surface_obj = VolatilitySurface(
            strikes=strikes.tolist(),
            expiries=[e/365 for e in expiries],
            vols=np.array(vol_surface)
        )
        
        # Calibrate SABR model
        sabr_params = vol_surface_obj.calibrate_sabr(spot_price, 0.05)
        
        return {
            "status": "success",
            "surface": {
                "symbol": symbol,
                "spot_price": spot_price,
                "strikes": strikes.tolist(),
                "expiries": expiries,
                "volatilities": vol_surface,
                "sabr_parameters": sabr_params
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error generating volatility surface: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/analyze")
async def analyze_option_strategy(
    strategy_type: str,
    legs: List[OptionPosition],
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Analyze common option strategies"""
    try:
        # Strategy analysis logic
        # This is a placeholder - implement full strategy analysis
        
        # Get spot price for first underlying
        quote = await data_svc.get_quote(legs[0].underlying)
        spot_price = quote["price"]
        
        # Calculate P&L profile
        price_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 100)
        pnl_profile = []
        
        for price in price_range:
            total_pnl = 0
            
            for leg in legs:
                # Simple P&L calculation
                if leg.option_type.lower() == "call":
                    intrinsic = max(0, price - leg.strike)
                else:
                    intrinsic = max(0, leg.strike - price)
                
                # Assume we bought at theoretical price
                # In practice, use actual trade prices
                pnl = (intrinsic - 5.0) * leg.quantity  # Dummy premium
                total_pnl += pnl
            
            pnl_profile.append(total_pnl)
        
        # Calculate key metrics
        max_profit = max(pnl_profile)
        max_loss = min(pnl_profile)
        breakeven_indices = [i for i, pnl in enumerate(pnl_profile) if abs(pnl) < 0.1]
        breakeven_prices = [price_range[i] for i in breakeven_indices]
        
        return {
            "status": "success",
            "strategy_analysis": {
                "type": strategy_type,
                "legs": len(legs),
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven_prices": breakeven_prices,
                "current_spot": spot_price,
                "pnl_profile": {
                    "prices": price_range.tolist(),
                    "pnl": pnl_profile
                }
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health")
async def options_health_check() -> Dict[str, Any]:
    """Check options service health"""
    return {
        "status": "healthy",
        "service": "Options Pricing",
        "models": ["black-scholes", "binomial", "monte-carlo"],
        "features": [
            "european_options",
            "american_options",
            "implied_volatility",
            "greeks_calculation",
            "portfolio_pricing",
            "var_calculation",
            "volatility_surface",
            "strategy_analysis"
        ]
    }