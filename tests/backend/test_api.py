import requests
import json
from datetime import datetime, timedelta

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/api/health/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_search_symbols():
    """Test symbol search"""
    print("Testing symbol search...")
    response = requests.post(
        f"{BASE_URL}/api/data/search",
        json={"query": "AAPL", "limit": 5}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_fetch_data():
    """Test data fetching"""
    print("Testing data fetch...")
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    response = requests.post(
        f"{BASE_URL}/api/data/fetch",
        json={
            "symbols": ["AAPL", "MSFT"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "interval": "1d"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        for symbol, series in data.items():
            print(f"\n{symbol}: {len(series['data'])} data points")
            if series['data']:
                print(f"  First: {series['data'][0]['timestamp']}")
                print(f"  Last: {series['data'][-1]['timestamp']}")
                print(f"  Latest close: ${series['data'][-1]['close']:.2f}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_data_quality():
    """Test data quality check"""
    print("Testing data quality check...")
    response = requests.get(f"{BASE_URL}/api/data/quality/AAPL")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        quality = response.json()
        print(f"Overall Score: {quality['metrics']['overall_score']:.2f}")
        print(f"Completeness: {quality['metrics']['completeness']:.2f}")
        print(f"Consistency: {quality['metrics']['consistency']:.2f}")
        print(f"Issues: {quality['issues']}")
        print(f"Recommendations: {quality['recommendations']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_idtxl_analysis():
    """Test IDTxl analysis"""
    print("Testing IDTxl analysis...")
    
    # First fetch some data
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    
    # Start analysis
    response = requests.post(
        f"{BASE_URL}/api/analysis/idtxl/start",
        json={
            "data": {"symbols": ["AAPL", "MSFT", "GOOGL"]},
            "config": {
                "analysis_type": "both",
                "max_lag": 5,
                "estimator": "kraskov",
                "significance_level": 0.05,
                "permutations": 200,
                "variables": ["AAPL", "MSFT", "GOOGL"],
                "k_neighbors": 3
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        task_id = result['task_id']
        print(f"Analysis started. Task ID: {task_id}")
        
        # Check status
        import time
        for i in range(10):
            time.sleep(1)
            status_response = requests.get(f"{BASE_URL}/api/analysis/status/{task_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"Status: {status['status']}")
                if status['status'] == 'completed':
                    # Get results
                    results_response = requests.get(f"{BASE_URL}/api/analysis/results/{task_id}")
                    if results_response.status_code == 200:
                        print(f"Results: {json.dumps(results_response.json(), indent=2)}")
                    break
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing Financial Analysis Platform API")
    print("=" * 50)
    
    test_health_check()
    test_search_symbols()
    test_fetch_data()
    test_data_quality()
    test_idtxl_analysis()