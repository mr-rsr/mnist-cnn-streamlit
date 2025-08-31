@echo off
echo Starting MNIST Digit Classifier Services...
echo.

echo Starting Flask API...
start "Flask API" cmd /k "python app.py"

echo Waiting for Flask API to start...
timeout /t 15 /nobreak > nul

echo Starting Streamlit UI...
start "Streamlit UI" cmd /k "streamlit run streamlit_app.py --server.fileWatcherType none --server.runOnSave false"

echo.
echo Both services are starting...
echo Flask API: http://localhost:5000
echo Streamlit UI: http://localhost:8501
echo.
echo Press any key to exit...
pause > nul