from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_trained_model():
    """Load the trained CNN model"""
    global model
    model_path = "models/mnist_cnn_model.h5"
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return True
    else:
        print(f"Model not found at {model_path}")
        return False

def preprocess_image(image_data):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert base64 to PIL Image if needed
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            image = Image.open(image_data)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 with proper interpolation
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # For canvas drawings: white strokes on black background is correct for MNIST
        # For uploaded images: check if inversion is needed
        # Canvas images typically have low mean (mostly black with white strokes)
        # Uploaded images might have high mean (white background with dark digits)
        
        # If the image has a high mean (bright), it's likely a scanned/photo digit
        # that needs inversion to match MNIST format
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize to 0-1
        img_array = img_array.astype(np.float32) / 255.0
        
        # Apply slight smoothing to reduce noise
        from scipy import ndimage
        img_array = ndimage.gaussian_filter(img_array, sigma=0.5)
        
        # Ensure proper contrast
        img_array = np.clip(img_array, 0, 1)
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "MNIST CNN API is running",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image"""
    global model
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"error": "No image provided"}), 400
        
        # Handle file upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({"error": "No image selected"}), 400
            
            # Preprocess image
            processed_image = preprocess_image(image_file)
        
        # Handle base64 image data
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities
        probabilities = {
            str(i): float(predictions[0][i]) 
            for i in range(10)
        }
        
        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/model/info')
def model_info():
    """Get model information"""
    global model
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        return jsonify({
            "model_summary": str(model.summary()),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_trained_model():
        print("Warning: Model not loaded. Train and save a model first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)