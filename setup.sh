#!/bin/bash
# Setup script for Indic-OCR project

echo "========================================"
echo "Indic-OCR Setup Script"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/{raw,processed,train,val,test}/{devanagari,malayalam,tamil}
mkdir -p models
mkdir -p outputs
mkdir -p logs
mkdir -p uploads

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short || echo "Some tests may fail if OCR engines are not installed"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the Streamlit web app, run:"
echo "  streamlit run app/streamlit_app.py"
echo ""
echo "To start the Flask web app, run:"
echo "  python app/flask_app.py"
echo ""
echo "For training, run:"
echo "  python train.py --help"
echo ""
