import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:5000"

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def predict_digit(image_data):
    """Send image to Flask API for prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"image_data": image_data},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def check_api_status():
    """Check if Flask API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def process_canvas_image(canvas_result):
    """Process the canvas drawing into a format suitable for prediction"""
    if canvas_result.image_data is not None:
        # Get the image data from canvas
        img_array = canvas_result.image_data
        
        # Check if something is drawn (not just empty canvas)
        if np.sum(img_array) > 0:
            # Convert to PIL Image
            image = Image.fromarray(img_array.astype('uint8'), 'RGBA')
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Convert to numpy array for processing
            img_np = np.array(image)
            
            # Resize to 28x28 with proper interpolation
            image = Image.fromarray(img_np)
            image = image.resize((28, 28), Image.LANCZOS)
            
            return image
    return None

# Main app
def main():
    st.title("🔢 MNIST Digit Classifier")
    st.markdown("Draw a digit (0-9) in the canvas below and get predictions!")
    
    # Check API status
    if not check_api_status():
        st.error("Flask API is not running. Please start the Flask server first.")
        st.code("python app.py")
        return
    
    st.success("Connected to Flask API")
    
    # Initialize session state
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw a Digit")
        
        # Canvas controls
        stroke_width = st.slider("Brush Size", 10, 30, 20)
        
        # Drawing canvas with simpler configuration
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",  # White stroke
            background_color="#000000",  # Black background
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Clear Canvas"):
                st.session_state.canvas_key += 1
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                st.rerun()
        
        with col_btn2:
            classify_button = st.button("Classify", type="primary")
        
        # Process and show the drawn image
        processed_image = process_canvas_image(canvas_result)
        if processed_image:
            st.subheader("Preview (28x28)")
            # Show the processed image (enlarged for visibility)
            display_img = processed_image.resize((140, 140), Image.NEAREST)
            st.image(display_img, caption="Processed for AI")
    
    with col2:
        st.subheader("Results")
        
        # Classify when button is pressed and image exists
        if classify_button and processed_image:
            with st.spinner("Classifying..."):
                # Encode image
                img_base64 = encode_image_to_base64(processed_image)
                
                # Get prediction
                result = predict_digit(img_base64)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Store results in session state
                    st.session_state.prediction_result = result
        
        # Display results if available
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            predicted_digit = result["predicted_digit"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            
            # Main prediction
            st.markdown(f"## Predicted: **{predicted_digit}**")
            st.markdown(f"**Confidence: {confidence:.1%}**")
            
            # Confidence indicator
            if confidence > 0.8:
                st.success("🟢 High Confidence")
            elif confidence > 0.5:
                st.warning("🟡 Medium Confidence")
            else:
                st.error("🔴 Low Confidence")
            
            # Top 3 predictions
            st.subheader("Top 3")
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for i, (digit, prob) in enumerate(sorted_probs[:3]):
                medal = ["🥇", "🥈", "🥉"][i]
                st.write(f"{medal} **{digit}**: {prob:.1%}")
            
            # Simple bar chart
            st.subheader("All Probabilities")
            prob_data = {int(k): v for k, v in probabilities.items()}
            
            # Create chart data
            digits = list(range(10))
            probs = [prob_data.get(i, 0) for i in digits]
            
            # Simple matplotlib chart
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['red' if i == predicted_digit else 'lightblue' for i in digits]
            ax.bar(digits, probs, color=colors)
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_xticks(digits)
            st.pyplot(fig)
        
        elif processed_image:
            st.info("Click 'Classify' to get prediction!")
        else:
            st.info("Draw a digit to get started!")
            
            st.markdown("""
            **Drawing Tips:**
            - Use thick white strokes
            - Draw large, centered digits  
            - Make clear, bold lines
            - Try different brush sizes
            """)
    
    # Simple instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Draw** a digit (0-9) in the black canvas using white color
        2. **Adjust** brush size if needed (10-30 pixels)
        3. **Click** 'Classify' to get AI prediction
        4. **Clear** canvas to try another digit
        
        **For best results:**
        - Draw digits similar to handwriting
        - Use bold, connected strokes
        - Center the digit in canvas
        - Make digits large enough to fill space
        """)

if __name__ == "__main__":
    main()