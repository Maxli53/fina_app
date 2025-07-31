import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytz

from app.models.data import (
    TimeSeriesData, OHLCV, DataInterval, 
    DataQualityMetrics, DataQualityReport, MarketStatus
)


class YahooFinanceService:
    """Service for fetching data from Yahoo Finance"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for symbols matching query"""
        def _search():
            # Yahoo Finance doesn't have a direct search API
            # We'll use a workaround with Ticker info
            results = []
            
            # Common suffixes for different markets
            suffixes = ['', '.L', '.TO', '.AX', '.HK', '.SI']
            
            for suffix in suffixes:
                try:
                    ticker = yf.Ticker(f"{query.upper()}{suffix}")
                    info = ticker.info
                    
                    if info and 'symbol' in info:
                        results.append({
                            "symbol": info.get('symbol', ''),
                            "name": info.get('longName', info.get('shortName', '')),
                            "exchange": info.get('exchange', ''),
                            "type": info.get('quoteType', ''),
                            "currency": info.get('currency', '')
                        })
                        
                        if len(results) >= limit:
                            break
                            
                except Exception:
                    continue
            
            return results[:limit]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _search)
    
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> TimeSeriesData:
        """Fetch historical OHLCV data for a symbol"""
        def _fetch():
            ticker = yf.Ticker(symbol)
            
            # Convert interval to yfinance format
            yf_interval = interval
            if interval == "1m":
                # For minute data, limit to last 7 days
                max_start = end_date - timedelta(days=7)
                start_date_adj = max(start_date, max_start)
            else:
                start_date_adj = start_date
            
            # Fetch data - try period first, then fall back to date range
            try:
                # Calculate approximate period
                days_diff = (end_date - start_date_adj).days
                if days_diff <= 7:
                    period = "1wk"
                elif days_diff <= 30:
                    period = "1mo"
                elif days_diff <= 90:
                    period = "3mo"
                elif days_diff <= 180:
                    period = "6mo"
                elif days_diff <= 365:
                    period = "1y"
                else:
                    period = "2y"
                
                df = ticker.history(period=period, interval=yf_interval, auto_adjust=False)
                
                # Filter to requested date range
                if not df.empty:
                    # Ensure timezone consistency
                    if df.index.tz is None:
                        # If data has no timezone, assume UTC
                        df.index = df.index.tz_localize('UTC')
                    
                    # Convert filter dates to same timezone as data
                    start_ts = pd.Timestamp(start_date_adj).tz_localize('UTC')
                    end_ts = pd.Timestamp(end_date).tz_localize('UTC')
                    
                    if df.index.tz != pytz.UTC:
                        start_ts = start_ts.tz_convert(df.index.tz)
                        end_ts = end_ts.tz_convert(df.index.tz)
                    
                    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            except:
                # Fallback to date-based fetch
                df = ticker.history(
                    start=start_date_adj.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=yf_interval,
                    auto_adjust=False
                )
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Convert to OHLCV format
            ohlcv_data = []
            for index, row in df.iterrows():
                # Convert timezone-aware timestamp to naive datetime
                timestamp = index.to_pydatetime()
                if timestamp.tzinfo is not None:
                    # Convert to UTC and make naive for consistency
                    timestamp = timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
                
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adjusted_close=float(row.get('Adj Close', row['Close']))
                )
                ohlcv_data.append(ohlcv)
            
            # Get ticker info for metadata
            info = ticker.info
            
            return TimeSeriesData(
                symbol=symbol,
                interval=DataInterval(interval),
                data=ohlcv_data,
                metadata={
                    "currency": info.get('currency', 'USD'),
                    "exchange": info.get('exchange', ''),
                    "timezone": info.get('exchangeTimezoneName', ''),
                    "regularMarketPrice": info.get('regularMarketPrice', None),
                    "fiftyDayAverage": info.get('fiftyDayAverage', None),
                    "twoHundredDayAverage": info.get('twoHundredDayAverage', None)
                }
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _fetch)
    
    async def assess_data_quality(self, data: TimeSeriesData) -> DataQualityReport:
        """Assess the quality of time series data"""
        def _assess():
            df = pd.DataFrame([
                {
                    'timestamp': d.timestamp,
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                } for d in data.data
            ])
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate completeness
            expected_periods = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=self._get_pandas_freq(data.interval)
            )
            missing_dates = list(set(expected_periods) - set(df.index))
            completeness = 1 - (len(missing_dates) / len(expected_periods))
            
            # Calculate consistency
            inconsistent = 0
            for _, row in df.iterrows():
                if row['high'] < row['low']:
                    inconsistent += 1
                if row['high'] < row['open'] or row['high'] < row['close']:
                    inconsistent += 1
                if row['low'] > row['open'] or row['low'] > row['close']:
                    inconsistent += 1
            consistency = 1 - (inconsistent / (len(df) * 3))
            
            # Calculate timeliness
            latest_date = df.index.max()
            
            # Handle timezone-aware timestamps properly
            if hasattr(latest_date, 'tz') and latest_date.tz is not None:
                # Convert to naive UTC datetime for comparison
                latest_date = latest_date.tz_convert(pytz.UTC).tz_localize(None)
            
            now = datetime.utcnow()  # Use UTC time for consistency
            days_old = (now - latest_date).days
            timeliness = max(0, 1 - (days_old / 30))  # Penalize if older than 30 days
            
            # Detect outliers using IQR method
            outliers = []
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                for idx in df[outlier_mask].index:
                    outliers.append({
                        "timestamp": idx.isoformat(),
                        "column": col,
                        "value": float(df.loc[idx, col])
                    })
            
            # Calculate accuracy (simplified - based on reasonable price ranges)
            accuracy = min(consistency, 0.95)  # Simplified metric
            
            # Generate recommendations
            recommendations = []
            if completeness < 0.95:
                recommendations.append(f"Data has {len(missing_dates)} missing periods. Consider gap filling.")
            if consistency < 0.98:
                recommendations.append("Inconsistent OHLC relationships detected. Review data source.")
            if len(outliers) > len(df) * 0.05:
                recommendations.append("High number of outliers detected. Consider outlier treatment.")
            if timeliness < 0.9:
                recommendations.append("Data is not up-to-date. Refresh data source.")
            
            issues = []
            if missing_dates:
                issues.append(f"{len(missing_dates)} missing data points")
            if inconsistent > 0:
                issues.append(f"{inconsistent} inconsistent OHLC relationships")
            
            return DataQualityReport(
                symbol=data.symbol,
                period_start=df.index.min(),
                period_end=df.index.max(),
                metrics=DataQualityMetrics(
                    completeness=completeness,
                    consistency=consistency,
                    timeliness=timeliness,
                    accuracy=accuracy
                ),
                issues=issues,
                missing_dates=missing_dates[:10],  # Limit to first 10
                outliers=outliers[:20],  # Limit to first 20
                recommendations=recommendations
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _assess)
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        def _get_status():
            # Use NY timezone for market hours (avoid external API calls)
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.now(ny_tz)
            
            # Simple market status logic based on NY time
            current_hour = now_ny.hour
            is_weekday = now_ny.weekday() < 5
            
            if is_weekday and 9 <= current_hour < 16:
                status = "open"
            elif is_weekday and 4 <= current_hour < 9:
                status = "pre-market"
            elif is_weekday and 16 <= current_hour < 20:
                status = "post-market"
            else:
                status = "closed"
            
            # Try to get market info, but don't fail if it doesn't work
            market_price = None
            try:
                ticker = yf.Ticker("SPY")
                info = ticker.info
                market_price = info.get('regularMarketPrice', None)
            except Exception:
                # Ignore SSL/network errors - just return status without price
                pass
            
            return {
                "US_MARKET": MarketStatus(
                    market="US",
                    status=status,
                    current_time=now_ny.replace(tzinfo=None),  # Make naive for consistency
                    timezone="America/New_York"
                ).dict(),
                "market_price_spy": market_price,
                "note": "Market status based on NY time. Price data may be unavailable due to network restrictions."
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _get_status)
    
    def _get_pandas_freq(self, interval: DataInterval) -> str:
        """Convert DataInterval to pandas frequency string"""
        mapping = {
            DataInterval.ONE_MINUTE: "1T",
            DataInterval.FIVE_MINUTES: "5T",
            DataInterval.FIFTEEN_MINUTES: "15T",
            DataInterval.THIRTY_MINUTES: "30T",
            DataInterval.ONE_HOUR: "1H",
            DataInterval.ONE_DAY: "1D",
            DataInterval.ONE_WEEK: "1W",
            DataInterval.ONE_MONTH: "1M"
        }
        return mapping.get(interval, "1D")