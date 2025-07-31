"""Test timezone handling fixes"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.data.yahoo_finance import YahooFinanceService

async def test_timezone_handling():
    """Test the timezone handling improvements"""
    print("Testing Timezone Handling Fixes")
    print("=" * 40)
    
    service = YahooFinanceService()
    
    # Test 1: Fetch data and check timestamps
    print("\n1. Testing data fetch with timezone handling...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Try multiple symbols in case of SSL/connection issues
        symbols_to_try = ["AAPL", "MSFT", "GOOGL", "SPY"]
        data = None
        
        for symbol in symbols_to_try:
            try:
                print(f"   Trying to fetch data for {symbol}...")
                data = await service.fetch_historical_data(symbol, start_date, end_date)
                break
            except Exception as symbol_error:
                print(f"   Failed to fetch {symbol}: {symbol_error}")
                continue
        
        if data is None:
            raise Exception("Could not fetch data from any symbol - likely connection issue")
        
        print(f"   Symbol: {data.symbol}")
        print(f"   Data points: {len(data.data)}")
        print(f"   First timestamp: {data.data[0].timestamp}")
        print(f"   Last timestamp: {data.data[-1].timestamp}")
        print(f"   Timezone in metadata: {data.metadata.get('timezone', 'Not specified')}")
        
        # Check if timestamps are timezone-naive (as intended)
        first_ts = data.data[0].timestamp
        if first_ts.tzinfo is None:
            print("   [OK] Timestamps are timezone-naive (UTC) as expected")
        else:
            print(f"   [WARN] Timestamps still have timezone info: {first_ts.tzinfo}")
            
    except Exception as e:
        print(f"   [ERROR] Error fetching data: {e}")
        return  # Exit early if data fetch fails
    
    # Test 2: Data quality assessment
    print("\n2. Testing data quality assessment with timezone handling...")
    try:
        quality_report = await service.assess_data_quality(data)
        
        print(f"   Completeness: {quality_report.metrics.completeness:.3f}")
        print(f"   Consistency: {quality_report.metrics.consistency:.3f}")
        print(f"   Timeliness: {quality_report.metrics.timeliness:.3f}")
        print(f"   Period: {quality_report.period_start} to {quality_report.period_end}")
        
        # Check if period dates are timezone-naive
        if quality_report.period_start.tzinfo is None:
            print("   [OK] Quality report dates are timezone-naive as expected")
        else:
            print(f"   [WARN] Quality report dates have timezone info: {quality_report.period_start.tzinfo}")
            
    except Exception as e:
        print(f"   [ERROR] Error in quality assessment: {e}")
    
    # Test 3: Market status with timezone
    print("\n3. Testing market status with proper timezone handling...")
    try:
        market_status = await service.get_market_status()
        us_market = market_status.get("US_MARKET", {})
        
        print(f"   Market: {us_market.get('market', 'Unknown')}")
        print(f"   Status: {us_market.get('status', 'Unknown')}")
        print(f"   Current time: {us_market.get('current_time', 'Unknown')}")
        print(f"   Timezone: {us_market.get('timezone', 'Unknown')}")
        
        current_time = us_market.get('current_time')
        if isinstance(current_time, str):
            print("   [OK] Current time is properly formatted as string")
        elif hasattr(current_time, 'tzinfo') and current_time.tzinfo is None:
            print("   [OK] Current time is timezone-naive as expected")
        else:
            print(f"   [WARN] Current time format: {type(current_time)}")
            
    except Exception as e:
        print(f"   [ERROR] Error getting market status: {e}")
    
    print("\n" + "=" * 40)
    print("Timezone handling test completed!")

if __name__ == "__main__":
    asyncio.run(test_timezone_handling())