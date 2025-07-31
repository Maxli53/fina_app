from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.services.data.yahoo_finance import YahooFinanceService
from app.models.data import TimeSeriesData, DataQualityReport

router = APIRouter()

# Initialize service
yf_service = YahooFinanceService()


class SymbolSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for symbols")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")


class DataRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, description="List of symbols to fetch")
    start_date: Optional[datetime] = Field(None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    interval: str = Field("1d", description="Data interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo")


@router.post("/search")
async def search_symbols(request: SymbolSearchRequest) -> List[Dict[str, Any]]:
    """Search for financial symbols"""
    try:
        results = await yf_service.search_symbols(request.query, request.limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search")
async def search_symbols_get(
    query: str = Query(..., description="Search query for symbols"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return")
) -> List[Dict[str, Any]]:
    """Search for financial symbols (GET version)"""
    try:
        results = await yf_service.search_symbols(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/fetch")
async def fetch_data(request: DataRequest) -> Dict[str, TimeSeriesData]:
    """Fetch historical data for multiple symbols"""
    try:
        # Default to last 90 days if no dates specified
        if not request.start_date:
            request.start_date = datetime.now() - timedelta(days=90)
        if not request.end_date:
            request.end_date = datetime.now()
            
        results = {}
        for symbol in request.symbols:
            data = await yf_service.fetch_historical_data(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                interval=request.interval
            )
            results[symbol] = data
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval")
) -> TimeSeriesData:
    """Get historical data for a single symbol (GET version)"""
    try:
        # Parse dates
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.now() - timedelta(days=90)
            
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()
            
        data = await yf_service.fetch_historical_data(
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
            interval=interval
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")


@router.get("/quality/{symbol}")
async def check_data_quality(
    symbol: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
) -> DataQualityReport:
    """Check data quality for a symbol"""
    try:
        # Fetch data first
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        data = await yf_service.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        # Assess quality
        quality_report = await yf_service.assess_data_quality(data)
        return quality_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


@router.get("/supported-intervals")
async def get_supported_intervals() -> List[str]:
    """Get list of supported data intervals"""
    return ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]


@router.get("/market-status")
async def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    try:
        status = await yf_service.get_market_status()
        return status
    except Exception as e:
        # Fallback to simple time-based status without external API calls
        import pytz
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        
        current_hour = now_ny.hour
        is_weekday = now_ny.weekday() < 5
        
        if is_weekday and 9 <= current_hour < 16:
            market_status = "open"
        elif is_weekday and 4 <= current_hour < 9:
            market_status = "pre-market"
        elif is_weekday and 16 <= current_hour < 20:
            market_status = "post-market"
        else:
            market_status = "closed"
            
        return {
            "US_MARKET": {
                "market": "US",
                "status": market_status,
                "current_time": now_ny.replace(tzinfo=None).isoformat(),
                "timezone": "America/New_York"
            },
            "note": f"Market status calculated from current NY time. Network error: {str(e)}"
        }