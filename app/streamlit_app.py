"""
Indic-OCR Web Application
=========================

Streamlit-based web interface for the Multi-Language OCR System
for Indian Regional Scripts.

Author: Dhananjayan H
Roll No: AA.SC.P2MCA24070151
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Indic-OCR | Multi-Language OCR System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f7ff;
        border: 1px solid #cce0ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .script-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px;
    }
    .devanagari { background-color: #FF6B6B; color: white; }
    .malayalam { background-color: #4ECDC4; color: white; }
    .tamil { background-color: #45B7D1; color: white; }
</style>
""", unsafe_allow_html=True)


def load_ocr_components():
    """Load OCR components with caching."""
    components = {}
    
    try:
        from src.preprocessing import ImagePreprocessor
        components['preprocessor'] = ImagePreprocessor()
        st.session_state['preprocessor_loaded'] = True
    except Exception as e:
        st.warning(f"Preprocessor not loaded: {e}")
        components['preprocessor'] = None
    
    try:
        from src.script_classifier import ScriptClassifier
        components['classifier'] = ScriptClassifier(model_type="cnn")
        st.session_state['classifier_loaded'] = True
    except Exception as e:
        st.warning(f"Script classifier not loaded: {e}")
        components['classifier'] = None
    
    try:
        from src.ocr_engine import OCREngine
        components['ocr_engine'] = OCREngine(engine="paddleocr", use_gpu=False)
        st.session_state['ocr_loaded'] = True
    except Exception as e:
        st.warning(f"OCR engine not loaded: {e}")
        components['ocr_engine'] = None
    
    return components


def process_image(image, components, options):
    """Process uploaded image through OCR pipeline."""
    results = {
        'text': '',
        'script': None,
        'script_confidence': 0.0,
        'processing_time': 0.0,
        'lines': [],
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Preprocess if enabled
        if options.get('preprocess', True) and components['preprocessor']:
            processed_img = components['preprocessor'].preprocess_for_ocr(img_array)
        else:
            processed_img = img_array
        
        # Detect script if enabled
        if options.get('detect_script', True) and components['classifier']:
            script, confidence = components['classifier'].predict(processed_img)
            results['script'] = script
            results['script_confidence'] = confidence
        
        # Perform OCR
        if components['ocr_engine']:
            ocr_result = components['ocr_engine'].recognize(
                processed_img, 
                language=results['script']
            )
            results['text'] = ocr_result.get('text', '')
            results['lines'] = ocr_result.get('lines', [])
        else:
            results['text'] = "[OCR Engine not available - Demo Mode]"
        
    except Exception as e:
        results['error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    
    return results


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="main-header">📝 Indic-OCR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Language OCR System for Indian Regional Scripts</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Indic-OCR", width=150)
        st.markdown("---")
        
        st.header("⚙️ Settings")
        
        # OCR Options
        st.subheader("OCR Options")
        enable_preprocessing = st.checkbox("Enable Preprocessing", value=True, 
                                          help="Apply image preprocessing for better accuracy")
        detect_script = st.checkbox("Auto-detect Script", value=True,
                                   help="Automatically detect the script/language")
        
        # Language Selection (if not auto-detecting)
        if not detect_script:
            selected_language = st.selectbox(
                "Select Language",
                ["Devanagari (Hindi)", "Malayalam", "Tamil", "English"],
                index=0
            )
        else:
            selected_language = None
        
        st.markdown("---")
        
        # Supported Scripts
        st.subheader("📚 Supported Scripts")
        st.markdown("""
        - 🔴 **Devanagari** (Hindi)
        - 🟢 **Malayalam**
        - 🔵 **Tamil**
        """)
        
        st.markdown("---")
        
        # Project Info
        st.subheader("ℹ️ Project Info")
        st.markdown("""
        **Author:** Dhananjayan H  
        **Roll No:** AA.SC.P2MCA24070151  
        **Course:** MCA Minor Project
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing text in supported Indian scripts"
        )
        
        # Sample images
        st.markdown("---")
        st.subheader("📸 Or try sample images")
        
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            if st.button("🔴 Hindi Sample"):
                st.info("Sample Hindi image would be loaded here")
        
        with sample_col2:
            if st.button("🟢 Malayalam Sample"):
                st.info("Sample Malayalam image would be loaded here")
        
        with sample_col3:
            if st.button("🔵 Tamil Sample"):
                st.info("Sample Tamil image would be loaded here")
        
        # Display uploaded image
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {uploaded_file.type}
            """)
    
    with col2:
        st.header("📄 Extracted Text")
        
        if uploaded_file:
            # Process button
            if st.button("🚀 Extract Text", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Load components
                    components = {
                        'preprocessor': None,
                        'classifier': None,
                        'ocr_engine': None
                    }
                    
                    # Try to load actual components
                    try:
                        from src.preprocessing import ImagePreprocessor
                        components['preprocessor'] = ImagePreprocessor()
                    except:
                        pass
                    
                    try:
                        from src.script_classifier import ScriptClassifier
                        components['classifier'] = ScriptClassifier(model_type="cnn")
                    except:
                        pass
                    
                    # Process options
                    options = {
                        'preprocess': enable_preprocessing,
                        'detect_script': detect_script,
                        'language': selected_language
                    }
                    
                    # Process image
                    image = Image.open(uploaded_file)
                    results = process_image(image, components, options)
                
                # Display results
                if results['error']:
                    st.error(f"Error: {results['error']}")
                else:
                    # Metrics row
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("⏱️ Processing Time", f"{results['processing_time']:.2f}s")
                    
                    with metric_col2:
                        if results['script']:
                            st.metric("📜 Detected Script", results['script'])
                    
                    with metric_col3:
                        if results['script_confidence']:
                            st.metric("🎯 Confidence", f"{results['script_confidence']*100:.1f}%")
                    
                    st.markdown("---")
                    
                    # Text result
                    st.subheader("Extracted Text")
                    
                    if results['text']:
                        st.text_area(
                            "OCR Result",
                            value=results['text'],
                            height=200,
                            label_visibility="collapsed"
                        )
                        
                        # Copy button
                        st.download_button(
                            label="📋 Download Text",
                            data=results['text'],
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No text could be extracted from the image.")
                    
                    # Line-by-line results
                    if results['lines']:
                        with st.expander("📝 Line-by-Line Results"):
                            for i, line in enumerate(results['lines'], 1):
                                st.markdown(f"**Line {i}:** {line.get('text', '')}")
                                if 'confidence' in line:
                                    st.progress(line['confidence'])
        else:
            st.info("👆 Upload an image to extract text")
            
            # Demo text area
            st.markdown("### 💡 Demo Output")
            st.text_area(
                "Sample Output",
                value="नमस्ते, यह एक परीक्षण है।\n\nThis is a sample output showing how the extracted text would appear.",
                height=150,
                label_visibility="collapsed",
                disabled=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Indic-OCR | Multi-Language OCR System for Indian Regional Scripts</p>
        <p>MCA Minor Project (21CSA697A) | Department of Computer Science</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
