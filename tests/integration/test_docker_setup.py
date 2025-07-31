"""
Test script to verify Docker setup and basic functionality
"""
import requests
import time
import sys

def test_backend_health():
    """Test if backend health endpoint is working"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_backend_docs():
    """Test if API documentation is accessible"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation not accessible: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def test_data_endpoint():
    """Test data retrieval endpoint"""
    try:
        # Test symbol search
        response = requests.get("http://localhost:8000/api/data/search?query=AAPL", timeout=10)
        if response.status_code == 200:
            print("✅ Data search endpoint working")
            data = response.json()
            print(f"   Found {len(data)} results")
            return True
        else:
            print(f"❌ Data search endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Data endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Docker Setup for Financial Platform")
    print("=" * 50)
    
    print("\n⏳ Waiting for services to start...")
    time.sleep(3)
    
    tests = [
        ("Backend Health Check", test_backend_health),
        ("API Documentation", test_backend_docs),
        ("Data Endpoints", test_data_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker setup is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the logs for more details.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Docker containers are running: python dev.py status")
        print("2. Check container logs: python dev.py logs")
        print("3. Restart services: python dev.py restart")
        return 1

if __name__ == "__main__":
    sys.exit(main())