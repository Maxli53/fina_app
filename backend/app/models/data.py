from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class DataInterval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"


class OHLCV(BaseModel):
    """Open, High, Low, Close, Volume data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


class TimeSeriesData(BaseModel):
    """Container for time series data"""
    symbol: str
    interval: DataInterval
    data: List[OHLCV]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment"""
    completeness: float = Field(..., ge=0, le=1, description="Ratio of non-missing data points")
    consistency: float = Field(..., ge=0, le=1, description="Ratio of consistent data points")
    timeliness: float = Field(..., ge=0, le=1, description="Data freshness score")
    accuracy: float = Field(..., ge=0, le=1, description="Data accuracy score")
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (self.completeness + self.consistency + self.timeliness + self.accuracy) / 4


class DataQualityReport(BaseModel):
    """Comprehensive data quality report"""
    symbol: str
    period_start: datetime
    period_end: datetime
    metrics: DataQualityMetrics
    issues: List[str] = Field(default_factory=list)
    missing_dates: List[datetime] = Field(default_factory=list)
    outliers: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketStatus(BaseModel):
    """Market trading status"""
    market: str
    status: str  # "open", "closed", "pre-market", "post-market"
    current_time: datetime
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    timezone: str