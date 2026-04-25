"""
Test script for CinnaGuard API
Run this after starting the FastAPI server
"""
import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("\n" + "="*50)
    print("TEST 1: Root Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_predict(image_path):
    """Test prediction endpoint"""
    print("\n" + "="*50)
    print("TEST 2: Predict Endpoint")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prediction Result:")
        print(f"   ID: {result['id']}")
        print(f"   Disease: {result['disease']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"   Severity: {result['severity']}")
        print(f"   Created At: {result['created_at']}")
        print(f"\n   All Probabilities:")
        for disease, prob in result['all_probabilities'].items():
            print(f"      {disease}: {prob*100:.2f}%")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_history():
    """Test history endpoint"""
    print("\n" + "="*50)
    print("TEST 3: History Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/history?limit=5")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Found {data['count']} predictions")
        
        if data['count'] > 0:
            print(f"\nLatest predictions:")
            for i, pred in enumerate(data['predictions'][:3], 1):
                print(f"\n{i}. ID: {pred['id']}")
                print(f"   Disease: {pred['disease']}")
                print(f"   Confidence: {pred['confidence']*100:.2f}%")
                print(f"   Severity: {pred['severity']}")
                print(f"   Created: {pred['created_at']}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*50)
    print("TEST 4: Statistics Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\n✅ Statistics:")
        print(f"   Total Scans: {stats['total_scans']}")
        print(f"   Accuracy: {stats['accuracy']:.1f}%")
        print(f"   Healthy Leaves: {stats['healthy_leaves']}")
        print(f"   Critical Cases: {stats['critical_cases']}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("CINNAMON GUARD API TESTING")
    print("="*50)
    print(f"Testing API at: {BASE_URL}")
    
    results = []
    
    # Test 1: Root
    try:
        results.append(("Root", test_root()))
    except Exception as e:
        print(f"❌ Root test failed: {e}")
        results.append(("Root", False))
    
    # Test 2: Predict (modify this path to your test image)
    test_image = r"C:\Users\Thulani\Desktop\project\dataset\Yellow_leaf_spots\2025_09_23_13_44_IMG_0023.jpg"
    try:
        results.append(("Predict", test_predict(test_image)))
    except Exception as e:
        print(f"❌ Predict test failed: {e}")
        results.append(("Predict", False))
    
    # Test 3: History
    try:
        results.append(("History", test_history()))
    except Exception as e:
        print(f"❌ History test failed: {e}")
        results.append(("History", False))
    
    # Test 4: Stats
    try:
        results.append(("Stats", test_stats()))
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        results.append(("Stats", False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:15s}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()