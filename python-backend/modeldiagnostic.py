"""
Diagnostic script to check why the model isn't loading
Run this before starting the server
Logs are saved to: diagnostic_log.txt
"""
import os
import sys
from datetime import datetime

# Create log file
log_file = "diagnostic_log.txt"
log = open(log_file, "w", encoding="utf-8")

def print_log(message):
    """Print to both console and file"""
    print(message)
    log.write(message + "\n")
    log.flush()

print_log("=" * 60)
print_log("CINNAGUARD MODEL DIAGNOSTIC")
print_log(f"Time: {datetime.now()}")
print_log("=" * 60)

# 1. Check Python version
print_log(f"\n1. Python Version: {sys.version}")
print_log(f"   Python executable: {sys.executable}")

# 2. Check if TensorFlow is installed
print_log("\n2. Checking TensorFlow...")
try:
    import tensorflow as tf
    print_log(f"   ✅ TensorFlow Version: {tf.__version__}")
    print_log(f"   TensorFlow location: {tf.__file__}")
except ImportError as e:
    print_log(f"   ❌ TensorFlow: NOT INSTALLED")
    print_log(f"   Error: {e}")
    print_log("   Fix: pip install tensorflow")
    log.close()
    sys.exit(1)

# 3. Check model path
MODEL_PATH = r"D:\Professor.lk\cinnoman leaf\python-backend\best_cinnamon_model.keras"
print_log(f"\n3. Checking model path:")
print_log(f"   Path: {MODEL_PATH}")
print_log(f"   Current directory: {os.getcwd()}")
print_log(f"   File exists: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print_log("   ❌ MODEL FILE NOT FOUND!")
    print_log("\n   Looking for .keras files in current directory and subdirectories...")
    found_count = 0
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".keras") or file.endswith(".h5"):
                full_path = os.path.abspath(os.path.join(root, file))
                file_size = os.path.getsize(full_path) / (1024 * 1024)
                print_log(f"   Found [{found_count+1}]: {full_path} ({file_size:.2f} MB)")
                found_count += 1
    
    if found_count == 0:
        print_log("   ❌ No .keras or .h5 model files found!")
    
    print_log("\n   💡 Solution: Copy the correct path above and update MODEL_PATH in main.py")
    log.close()
    sys.exit(1)

# 4. Get file info
file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
print_log(f"   ✅ File found!")
print_log(f"   File size: {file_size:.2f} MB")

# 5. Try to load the model
print_log(f"\n4. Attempting to load model...")
print_log(f"   Loading with compile=False...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print_log(f"   ✅ MODEL LOADED SUCCESSFULLY!")
    print_log(f"   Input shape: {model.input_shape}")
    print_log(f"   Output shape: {model.output_shape}")
    print_log(f"   Total layers: {len(model.layers)}")
    print_log(f"   Model type: {type(model).__name__}")
except Exception as e:
    print_log(f"   ❌ FAILED TO LOAD MODEL!")
    print_log(f"   Error type: {type(e).__name__}")
    print_log(f"   Error message: {str(e)}")
    print_log("\n   Full traceback:")
    import traceback
    print_log(traceback.format_exc())
    print_log("\n   💡 Possible solutions:")
    print_log("   - Model might be corrupted, try re-downloading it")
    print_log("   - Model might be from a different TensorFlow version")
    print_log("   - Try: pip install --upgrade tensorflow")
    log.close()
    sys.exit(1)

# 6. Test prediction
print_log(f"\n5. Testing prediction with dummy data...")
try:
    import numpy as np
    # Create dummy image (224x224x3)
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_image = tf.keras.applications.efficientnet.preprocess_input(dummy_image)
    
    print_log(f"   Input shape: {dummy_image.shape}")
    predictions = model.predict(dummy_image, verbose=0)
    print_log(f"   ✅ Prediction test passed!")
    print_log(f"   Output shape: {predictions.shape}")
    print_log(f"   Output values: {predictions[0]}")
    print_log(f"   Sum of probabilities: {np.sum(predictions[0]):.4f}")
except Exception as e:
    print_log(f"   ❌ Prediction test failed!")
    print_log(f"   Error: {e}")
    import traceback
    print_log(traceback.format_exc())
    log.close()
    sys.exit(1)

# 7. Check other dependencies
print_log(f"\n6. Checking other dependencies...")
dependencies = {
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'mysql.connector': 'mysql-connector-python',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'numpy': 'numpy',
    'pydantic': 'pydantic'
}

missing = []
for module, package in dependencies.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print_log(f"   ✅ {package} (version: {version})")
    except ImportError:
        print_log(f"   ❌ {package} - Run: pip install {package}")
        missing.append(package)

if missing:
    print_log(f"\n   ⚠️  Missing packages: {', '.join(missing)}")
    print_log(f"   Run: pip install {' '.join(missing)}")

# 8. Check MySQL connection
print_log(f"\n7. Checking MySQL connection...")
try:
    import mysql.connector
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password=''
    )
    print_log(f"   ✅ MySQL connection successful!")
    conn.close()
except Exception as e:
    print_log(f"   ⚠️  MySQL connection failed: {e}")
    print_log(f"   Make sure MySQL/XAMPP is running")

# Summary
print_log("\n" + "=" * 60)
print_log("DIAGNOSIS COMPLETE!")
print_log("=" * 60)

if model is not None:
    print_log("\n✅ ALL CHECKS PASSED!")
    print_log("✅ Your server should work now.")
    print_log("\nNext steps:")
    print_log("1. Run: python main.py")
    print_log("2. Visit: http://localhost:8000")
    print_log("3. Test with: POST http://localhost:8000/predict")
else:
    print_log("\n⚠️  Some issues found. Check the errors above.")

print_log(f"\n📄 Full log saved to: {os.path.abspath(log_file)}")
print_log("=" * 60)

log.close()
print(f"\n✅ Log file created: {os.path.abspath(log_file)}")
print("Please share this file if you need help!")