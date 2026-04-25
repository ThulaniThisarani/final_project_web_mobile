from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import base64
from typing import List, Optional
import mysql.connector
from mysql.connector import Error
import os
from pydantic import BaseModel, EmailStr
import hashlib
import logging
import traceback
import io

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = (224, 224)
MODEL_PATH = r"D:\project\best_cinnamon_model.keras"

CLASS_NAMES = [
    "Black_Sooty_Mold",
    "Blight_Disease",
    "Leaf_Gall_Disease",
    "Yellow_leaf_spots"
]

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',  
    'database': 'cinnamon_db',
    'port': 3307
}

# ============================================================
# INITIALIZE FASTAPI
# ============================================================
app = FastAPI(title="CinnaGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CUSTOM LAYER FOR COMPATIBILITY
# ============================================================
@tf.keras.utils.register_keras_serializable()
class CompatibleRandomFlip(tf.keras.layers.Layer):
    """Custom RandomFlip that ignores data_format parameter"""
    def __init__(self, mode='horizontal', seed=None, **kwargs):
        # Remove data_format if present
        kwargs.pop('data_format', None)
        super().__init__(**kwargs)
        self.mode = mode
        self.seed = seed
    
    def call(self, inputs, training=None):
        # During inference, just return inputs unchanged
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "seed": self.seed})
        return config

# Create custom objects for all augmentation layers
CUSTOM_OBJECTS = {
    'RandomFlip': CompatibleRandomFlip,
    'RandomRotation': tf.keras.layers.Lambda(lambda x: x),  # Identity function
    'RandomZoom': tf.keras.layers.Lambda(lambda x: x),
    'RandomContrast': tf.keras.layers.Lambda(lambda x: x),
    'RandomBrightness': tf.keras.layers.Lambda(lambda x: x),
}

# ============================================================
# LOAD KERAS MODEL
# ============================================================
model = None
model_error = None

def load_model():
    """Load the Keras model with custom objects for compatibility"""
    global model, model_error
    try:
        if not os.path.exists(MODEL_PATH):
            model_error = f"Model file not found at: {MODEL_PATH}"
            logger.error(f"❌ {model_error}")
            return False
        
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"📦 Model file found! Size: {file_size:.2f} MB")
        logger.info(f"⏳ Loading model with custom objects...")
        
        # Load with custom objects to handle version incompatibilities
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        
        logger.info("✅ Keras model loaded successfully!")
        logger.info(f"📊 Model input shape: {model.input_shape}")
        logger.info(f"📊 Model output shape: {model.output_shape}")
        logger.info(f"🏷️  Classes: {CLASS_NAMES}")
        
        # Test prediction
        logger.info("🧪 Testing model prediction...")
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_input = tf.keras.applications.efficientnet.preprocess_input(test_input)
        test_pred = model.predict(test_input, verbose=0)
        logger.info(f"✅ Model test passed! Output shape: {test_pred.shape}")
        
        return True
        
    except Exception as e:
        model_error = str(e)
        logger.error(f"❌ Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False

# Try to load model on startup
load_model()

# ============================================================
# PYDANTIC MODELS
# ============================================================
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# ============================================================
# DATABASE FUNCTIONS
# ============================================================
def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Initialize database and create tables"""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            cursor.execute("CREATE DATABASE IF NOT EXISTS cinnamon_db")
            cursor.execute("USE cinnamon_db")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_email (email)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    disease VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    image_data LONGTEXT NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    all_probabilities JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_created_at (created_at),
                    INDEX idx_disease (disease),
                    INDEX idx_user_id (user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            connection.commit()
            cursor.close()
            connection.close()
            logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================
def load_image_from_upload(file_contents: bytes) -> np.ndarray:
    """Load image from uploaded file bytes, supporting multiple formats including HEIC"""
    try:
        # First try with PIL (supports more formats including HEIC)
        image = Image.open(io.BytesIO(file_contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"✅ Image loaded via PIL. Shape: {img_bgr.shape}")
        return img_bgr
        
    except Exception as pil_error:
        logger.warning(f"PIL failed: {pil_error}. Trying OpenCV...")
        
        try:
            nparr = np.frombuffer(file_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("OpenCV failed to decode image")
            
            logger.info(f"✅ Image loaded via OpenCV. Shape: {img.shape}")
            return img
            
        except Exception as cv_error:
            logger.error(f"Both PIL and OpenCV failed")
            raise ValueError("Failed to load image. Supported formats: JPG, PNG, HEIC, WebP")

def predict_leaf_disease(img_array):
    """Predict disease from image array using Keras model"""
    try:
        logger.info(f"Input image shape: {img_array.shape}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        
        # Convert to float32 and add batch dimension
        img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        # EfficientNet preprocessing
        img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)
        
        # Predict
        predictions = model.predict(img_preprocessed, verbose=0)[0]
        
        # Get predicted class
        class_index = np.argmax(predictions)
        disease_name = CLASS_NAMES[class_index]
        confidence = float(predictions[class_index])
        
        # Create probability dictionary
        all_probabilities = {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)}
        
        logger.info(f"✅ Prediction: {disease_name} ({confidence:.2%})")
        return disease_name, confidence, all_probabilities
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise

def get_severity(disease_name: str, confidence: float) -> str:
    """Determine severity based on disease and confidence"""
    high_severity = ["blight_disease", "black_sooty_mold"]
    medium_severity = ["yellow_leaf_spots", "leaf_gall_disease"]
    
    disease_lower = disease_name.lower()
    
    if any(hs in disease_lower for hs in high_severity):
        return "high" if confidence > 0.8 else "medium"
    elif any(ms in disease_lower for ms in medium_severity):
        return "medium" if confidence > 0.7 else "low"
    else:
        return "low"

# ============================================================
# API ENDPOINTS
# ============================================================
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CinnaGuard API is running",
        "version": "2.0.0",
        "model": "Keras/TensorFlow EfficientNet",
        "model_loaded": model is not None,
        "model_error": model_error if model is None else None,
        "tensorflow_version": tf.__version__,
        "supported_formats": ["JPG", "PNG", "HEIC", "WebP"],
        "endpoints": {
            "register": "/register",
            "login": "/login",
            "predict": "/predict",
            "history": "/history",
            "stats": "/stats",
            "reload_model": "/reload-model"
        }
    }

@app.post("/reload-model")
async def reload_model():
    """Manually reload the model"""
    success = load_model()
    if success:
        return {"message": "Model reloaded successfully", "model_loaded": True}
    else:
        return {"message": f"Failed to load model: {model_error}", "model_loaded": False}

@app.post("/register")
async def register_user(user: UserRegister):
    """Register a new user"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = hash_password(user.password)
        
        cursor.execute("""
            INSERT INTO users (name, email, password)
            VALUES (%s, %s, %s)
        """, (user.name, user.email, hashed_password))
        
        connection.commit()
        user_id = cursor.lastrowid
        
        cursor.execute("SELECT created_at FROM users WHERE id = %s", (user_id,))
        created_at = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return {
            "id": user_id,
            "name": user.name,
            "email": user.email,
            "created_at": created_at.isoformat(),
            "message": "User registered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/login")
async def login_user(user: UserLogin):
    """Login user"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        hashed_password = hash_password(user.password)
        
        cursor.execute("""
            SELECT id, name, email, created_at 
            FROM users 
            WHERE email = %s AND password = %s
        """, (user.email, hashed_password))
        
        user_data = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        return {
            "id": user_data['id'],
            "name": user_data['name'],
            "email": user_data['email'],
            "created_at": user_data['created_at'].isoformat(),
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), user_id: Optional[int] = None):
    """Predict disease from uploaded image - Supports JPG, PNG, HEIC, WebP"""
    logger.info(f"📥 Prediction request. File: {file.filename}, User: {user_id}")
    
    if model is None:
        logger.error("❌ Model not loaded")
        raise HTTPException(
            status_code=500, 
            detail=f"Model not loaded. Error: {model_error}"
        )
    
    try:
        # Read image
        logger.info("📖 Reading image...")
        contents = await file.read()
        logger.info(f"✅ File read. Size: {len(contents)} bytes")
        
        # Load image
        img = load_image_from_upload(contents)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Predict
        logger.info("🔮 Predicting...")
        disease, confidence, all_probs = predict_leaf_disease(img)
        severity = get_severity(disease, confidence)
        
        # Convert to base64
        logger.info("💾 Converting image...")
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{image_base64}"
        
        # Save to database
        logger.info("💿 Saving to database...")
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        import json
        
        cursor.execute("""
            INSERT INTO predictions 
            (user_id, disease, confidence, image_data, severity, all_probabilities)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, disease, confidence, image_data, severity, json.dumps(all_probs)))
        
        connection.commit()
        prediction_id = cursor.lastrowid
        
        cursor.close()
        connection.close()
        
        logger.info(f"✅ Complete! ID: {prediction_id}")
        
        return {
            "id": prediction_id,
            "disease": disease,
            "confidence": confidence,
            "severity": severity,
            "image_data": image_data,
            "all_probabilities": all_probs,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history")
async def get_history(limit: int = 50, user_id: Optional[int] = None):
    """Get prediction history"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        if user_id:
            cursor.execute("""
                SELECT id, disease, confidence, image_data, severity,
                       all_probabilities, created_at
                FROM predictions WHERE user_id = %s
                ORDER BY created_at DESC LIMIT %s
            """, (user_id, limit))
        else:
            cursor.execute("""
                SELECT id, disease, confidence, image_data, severity,
                       all_probabilities, created_at
                FROM predictions
                ORDER BY created_at DESC LIMIT %s
            """, (limit,))
        
        predictions = cursor.fetchall()
        
        import json
        for pred in predictions:
            pred['created_at'] = pred['created_at'].isoformat()
            if pred['all_probabilities']:
                pred['all_probabilities'] = json.loads(pred['all_probabilities'])
        
        cursor.close()
        connection.close()
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/stats")
async def get_stats(user_id: Optional[int] = None):
    """Get statistics"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        where_clause = "WHERE user_id = %s" if user_id else ""
        params = (user_id,) if user_id else ()
        
        cursor.execute(f"SELECT COUNT(*) as total FROM predictions {where_clause}", params)
        total_scans = cursor.fetchone()['total']
        
        cursor.execute(f"SELECT AVG(confidence) * 100 as avg_conf FROM predictions {where_clause}", params)
        accuracy = cursor.fetchone()['avg_conf'] or 0
        
        cursor.execute(f"SELECT COUNT(*) as count FROM predictions {where_clause} {'AND' if user_id else 'WHERE'} severity = 'low'", params)
        healthy_leaves = cursor.fetchone()['count']
        
        cursor.execute(f"SELECT COUNT(*) as count FROM predictions {where_clause} {'AND' if user_id else 'WHERE'} severity = 'high'", params)
        critical_cases = cursor.fetchone()['count']
        
        cursor.close()
        connection.close()
        
        return {
            "total_scans": total_scans,
            "accuracy": round(accuracy, 1),
            "healthy_leaves": healthy_leaves,
            "critical_cases": critical_cases
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete a prediction"""
    try:
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s", (prediction_id,))
        connection.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        cursor.close()
        connection.close()
        
        return {"message": "Prediction deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)