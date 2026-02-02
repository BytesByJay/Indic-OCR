"""
Flask Web Application for Indic-OCR
====================================

Alternative web interface using Flask.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_ocr_components():
    """Initialize and return OCR components."""
    components = {}
    
    try:
        from src.preprocessing import ImagePreprocessor
        components['preprocessor'] = ImagePreprocessor()
    except Exception as e:
        components['preprocessor'] = None
        print(f"Preprocessor not available: {e}")
    
    try:
        from src.script_classifier import ScriptClassifier
        components['classifier'] = ScriptClassifier(model_type='cnn')
    except Exception as e:
        components['classifier'] = None
        print(f"Classifier not available: {e}")
    
    try:
        from src.ocr_engine import OCREngine
        components['ocr_engine'] = OCREngine(engine='paddleocr', use_gpu=False)
    except Exception as e:
        components['ocr_engine'] = None
        print(f"OCR Engine not available: {e}")
    
    return components


# Load components at startup
components = None


@app.before_request
def initialize_components():
    """Initialize components before first request."""
    global components
    if components is None:
        components = get_ocr_components()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    """
    Perform OCR on uploaded image.
    
    Returns JSON with extracted text and metadata.
    """
    result = {
        'success': False,
        'text': '',
        'script': None,
        'confidence': None,
        'error': None
    }
    
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            # Check for base64 image
            if 'image_base64' in request.json:
                image_data = base64.b64decode(request.json['image_base64'])
                image = Image.open(io.BytesIO(image_data))
            else:
                result['error'] = 'No image provided'
                return jsonify(result), 400
        else:
            file = request.files['image']
            
            if file.filename == '':
                result['error'] = 'No file selected'
                return jsonify(result), 400
            
            if not allowed_file(file.filename):
                result['error'] = 'File type not allowed'
                return jsonify(result), 400
            
            image = Image.open(file)
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Preprocess
        if components.get('preprocessor'):
            processed = components['preprocessor'].preprocess_for_ocr(img_array)
        else:
            processed = img_array
        
        # Detect script
        if components.get('classifier'):
            script, confidence = components['classifier'].predict(processed)
            result['script'] = script
            result['confidence'] = float(confidence)
        
        # Perform OCR
        if components.get('ocr_engine'):
            ocr_result = components['ocr_engine'].recognize(
                processed,
                language=result.get('script')
            )
            result['text'] = ocr_result.get('text', '')
            result['lines'] = ocr_result.get('lines', [])
        else:
            result['text'] = '[OCR Engine not available]'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        return jsonify(result), 500
    
    return jsonify(result)


@app.route('/api/status')
def api_status():
    """Return API status and component availability."""
    return jsonify({
        'status': 'running',
        'components': {
            'preprocessor': components.get('preprocessor') is not None,
            'classifier': components.get('classifier') is not None,
            'ocr_engine': components.get('ocr_engine') is not None
        }
    })


# HTML template for the main page
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indic-OCR | Multi-Language OCR System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #a0a0a0;
            font-size: 1.1rem;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            margin-bottom: 20px;
            color: #4fc3f7;
        }
        .upload-area {
            border: 2px dashed #4fc3f7;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: rgba(79, 195, 247, 0.1);
        }
        .upload-area input {
            display: none;
        }
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .btn {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            color: #1a1a2e;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result-area {
            min-height: 200px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 20px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .metrics {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric {
            background: rgba(79, 195, 247, 0.2);
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4fc3f7;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #a0a0a0;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #4fc3f7;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #a0a0a0;
            margin-top: 50px;
        }
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📝 Indic-OCR</h1>
            <p class="subtitle">Multi-Language OCR System for Indian Regional Scripts</p>
        </header>
        
        <div class="main-content">
            <div class="card">
                <h2>📤 Upload Image</h2>
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="imageInput" accept="image/*">
                    <p>📷 Click or drag an image here</p>
                    <p style="color: #a0a0a0; font-size: 0.9rem; margin-top: 10px;">
                        Supported: JPG, PNG, BMP, TIFF
                    </p>
                </div>
                <img id="previewImage" class="preview-image" style="display: none;">
                <button class="btn" id="extractBtn" disabled>🚀 Extract Text</button>
            </div>
            
            <div class="card">
                <h2>📄 Extracted Text</h2>
                <div class="metrics" id="metrics" style="display: none;">
                    <div class="metric">
                        <div class="metric-value" id="scriptValue">-</div>
                        <div class="metric-label">Script</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="confidenceValue">-</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px;">Processing image...</p>
                </div>
                <div class="result-area" id="resultArea">
                    Upload an image to extract text...
                </div>
            </div>
        </div>
        
        <footer>
            <p>Indic-OCR | MCA Minor Project (21CSA697A)</p>
            <p>Author: Dhananjayan H | Roll No: AA.SC.P2MCA24070151</p>
        </footer>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const previewImage = document.getElementById('previewImage');
        const extractBtn = document.getElementById('extractBtn');
        const resultArea = document.getElementById('resultArea');
        const loading = document.getElementById('loading');
        const metrics = document.getElementById('metrics');
        
        let selectedFile = null;
        
        uploadArea.addEventListener('click', () => imageInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(79, 195, 247, 0.2)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'transparent';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'transparent';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                extractBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        extractBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            loading.style.display = 'block';
            resultArea.style.display = 'none';
            metrics.style.display = 'none';
            extractBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/api/ocr', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                
                if (data.success) {
                    resultArea.textContent = data.text || 'No text extracted';
                    
                    if (data.script) {
                        metrics.style.display = 'flex';
                        document.getElementById('scriptValue').textContent = data.script;
                        document.getElementById('confidenceValue').textContent = 
                            data.confidence ? (data.confidence * 100).toFixed(1) + '%' : '-';
                    }
                } else {
                    resultArea.textContent = 'Error: ' + (data.error || 'Unknown error');
                }
            } catch (error) {
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                resultArea.textContent = 'Error: ' + error.message;
            }
            
            extractBtn.disabled = false;
        });
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
templates_dir = Path(__file__).parent / 'templates'
templates_dir.mkdir(exist_ok=True)

with open(templates_dir / 'index.html', 'w') as f:
    f.write(INDEX_HTML)


if __name__ == '__main__':
    print("=" * 60)
    print("Indic-OCR Flask Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
