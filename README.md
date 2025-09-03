# MNIST Digit Classifier

A machine learning web application that classifies handwritten digits (0-9) using a CNN model. The project includes both a standalone Streamlit app and a Flask API with Streamlit frontend.

## Project Structure

```
├── app.py                    # Flask API server
├── streamlit_app.py         # Streamlit frontend (connects to Flask API)
├── streamlit_deploy.py      # Standalone Streamlit app (no API needed)
├── requirements.txt         # Python dependencies
├── start_service.bat       # Windows batch file to start Flask API
├── models/
│   └── mnist_cnn_model.h5  # Trained CNN model
└── venv/                   # Virtual environment (created during setup)
```

## Setup Instructions

### 1. Create Virtual Environment

First, create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# For Git Bash or PowerShell on Windows
venv/Scripts/activate
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Running the Applications

### Option 1: Standalone Streamlit App (Recommended for Quick Start)

Run the standalone version that doesn't require a separate API server:

```bash
streamlit run streamlit_deploy.py
```

This will:
- Load the CNN model directly
- Start the Streamlit web interface
- Open your browser to `http://localhost:8501`

### Option 2: Flask API + Streamlit Frontend

For a more production-like setup with separate API and frontend:

#### Step 1: Start the Flask API Server

```bash
# Option A: Run directly
python app.py

# Option B: Use the batch file (Windows)
start_service.bat
```

The Flask API will start on `http://localhost:5000`

#### Step 2: Start the Streamlit Frontend

In a new terminal (with venv activated):

```bash
streamlit run streamlit_app.py
```

The Streamlit app will start on `http://localhost:8501`

## Usage

1. **Draw a Digit**: Use the canvas to draw a digit (0-9) with white strokes on black background
2. **Adjust Brush Size**: Use the slider to change brush thickness (10-30 pixels)
3. **Classify**: Click the "Classify" button to get AI predictions
4. **View Results**: See the predicted digit, confidence score, and probability distribution
5. **Clear Canvas**: Start over with a new digit

## Features

- **Interactive Canvas**: Draw digits with adjustable brush size
- **Real-time Preview**: See how your drawing looks after processing (28x28 pixels)
- **Confidence Scoring**: Get confidence levels for predictions
- **Probability Distribution**: View probabilities for all digits (0-9)
- **Top 3 Predictions**: See the most likely digit candidates
- **Visual Charts**: Bar chart showing prediction probabilities

## Technical Details

- **Model**: Convolutional Neural Network (CNN) trained on MNIST dataset
- **Input**: 28x28 grayscale images
- **Output**: Probability distribution over 10 digit classes (0-9)
- **Frontend**: Streamlit with drawable canvas component
- **API**: Flask with CORS support for cross-origin requests

## Troubleshooting

### Model Not Found Error
If you see "Model not found" error, ensure the trained model exists at `models/mnist_cnn_model.h5`

### API Connection Error (streamlit_app.py)
- Make sure Flask API is running on `http://localhost:5000`
- Check that both services are running simultaneously
- Verify no firewall is blocking the connection

### Canvas Not Working
- Try refreshing the page
- Clear browser cache
- Use a different browser (Chrome/Firefox recommended)

### Virtual Environment Issues
```bash
# Deactivate current environment
deactivate

# Remove and recreate venv
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Development

To modify or extend the application:

1. **Model Changes**: Update the model file in `models/` directory
2. **API Endpoints**: Modify `app.py` for new API functionality  
3. **Frontend**: Edit `streamlit_app.py` or `streamlit_deploy.py` for UI changes
4. **Dependencies**: Add new packages to `requirements.txt`

## Performance Tips

- Use bold, centered digits for best accuracy
- Draw digits similar to handwritten style
- Ensure good contrast between strokes and background
- Try different brush sizes for optimal results