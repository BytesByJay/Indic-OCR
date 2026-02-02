# Indic-OCR: Multi-Language OCR System for Indian Regional Scripts

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Development-yellow.svg)

A machine learning-based OCR solution that can detect, classify, and accurately recognize text from multiple Indian regional scripts and convert it into editable Unicode text.

## 📋 Project Information

- **Title:** Multi-Language OCR System for Indian Regional Scripts (Indic-OCR)
- **Author:** Dhananjayan H
- **Roll No:** AA.SC.P2MCA24070151
- **Course:** MCA Minor Project (21CSA697A)

## 🎯 Objectives

1. Develop a multi-language OCR system capable of identifying the script and extracting text from handwritten or printed Indian regional languages
2. Convert extracted text into Unicode digital text
3. Support applications like digitization of academic notes, historical records, government documents, and accessibility enhancement

## ✨ Features

- **Multi-Script Support:** Devanagari (Hindi), Malayalam, Tamil
- **Script Detection:** Automatic identification of the input script
- **Image Preprocessing:** Advanced preprocessing for improved accuracy
- **Deep Learning OCR:** State-of-the-art recognition using PaddleOCR/TrOCR
- **Web Interface:** User-friendly Streamlit-based interface
- **Evaluation Tools:** CER/WER metrics for performance assessment

## 🏗️ Project Structure

```
Indic-OCR/
├── app/
│   └── streamlit_app.py       # Web interface
├── config/
│   └── config.yaml            # Configuration file
├── data/
│   ├── raw/                   # Raw dataset
│   ├── processed/             # Processed images
│   ├── train/                 # Training set
│   ├── val/                   # Validation set
│   └── test/                  # Test set
├── models/                    # Saved models
├── notebooks/                 # Jupyter notebooks
├── outputs/                   # OCR outputs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Image preprocessing
│   ├── script_classifier.py   # Script identification model
│   ├── ocr_engine.py          # OCR recognition
│   ├── dataset.py             # Dataset utilities
│   ├── evaluation.py          # Evaluation metrics
│   └── utils.py               # Utility functions
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── train.py                   # Training script
├── inference.py               # Inference script
└── README.md                  # Documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
cd "MCA Project/Indic-OCR"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support (optional):
```bash
pip install paddlepaddle-gpu  # For CUDA 11.x
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset

### Supported Datasets

1. **Public Datasets:**
   - [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/rishianand/devanagari-character-set)
   - [Tamil Handwritten Dataset](https://www.kaggle.com/datasets/sudalairajkumar/tamil-nlp)
   - BHaratWrites Dataset

2. **Self-collected Data:**
   - Scanned documents
   - Handwritten notes
   - Printed materials

### Data Preparation

```bash
python -c "from src.dataset import DatasetManager; dm = DatasetManager(); dm.split_dataset()"
```

## 🎓 Training

### Train Script Classifier

```bash
python train.py --task script_classifier --epochs 50 --batch_size 32
```

### Train OCR Model (Fine-tuning)

```bash
python train.py --task ocr --model paddleocr --language hindi
```

## 🔮 Inference

### Command Line

```bash
python inference.py --image path/to/image.png --output results.txt
```

### Python API

```python
from src import ImagePreprocessor, ScriptClassifier, OCREngine

# Initialize components
preprocessor = ImagePreprocessor()
classifier = ScriptClassifier()
ocr = OCREngine()

# Process image
image = preprocessor.preprocess("document.png")
script, confidence = classifier.predict(image)
result = ocr.recognize(image, language=script)

print(f"Detected Script: {script}")
print(f"Extracted Text: {result['text']}")
```

## 🌐 Web Interface

Launch the Streamlit web application:

```bash
cd Indic-OCR
streamlit run app/streamlit_app.py
```

Access the interface at `http://localhost:8501`

## 📈 Evaluation

### Run Evaluation

```bash
python -c "from src.evaluation import evaluate_ocr_results; evaluate_ocr_results(predictions, ground_truths)"
```

### Metrics

- **CER (Character Error Rate):** Measures character-level accuracy
- **WER (Word Error Rate):** Measures word-level accuracy
- **Accuracy:** Percentage of correctly recognized samples

## 📅 Timeline & Milestones

| Week | Milestone |
|------|-----------|
| 1 | Literature review & requirements analysis |
| 2 | Dataset collection & preprocessing |
| 3 | Script identification model training |
| 4 | OCR engine integration & testing |
| 5 | UI development & deployment |
| 6 | Accuracy evaluation & improvements |
| 7 | Final testing, documentation & presentation |

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python |
| Libraries | OpenCV, PaddleOCR/TrOCR, TensorFlow/PyTorch |
| IDE | VS Code, Jupyter Notebook |
| Interface | Streamlit |
| Version Control | GitHub |
| Hardware | Laptop (8GB+ RAM) + Google Colab GPU |

## 📚 Learning Outcomes

- **ML & DL:** Model training, image classification, evaluation
- **Computer Vision:** Preprocessing, thresholding, deskewing, feature extraction
- **OCR Systems:** Text detection, recognition, and Unicode conversion
- **Research Methodology:** Dataset preparation, benchmarking, literature review
- **Web Development:** Creating a functional front-end for OCR usage

## 📖 References

1. PaddleOCR Documentation: https://github.com/PaddlePaddle/PaddleOCR
2. TrOCR Paper: https://arxiv.org/abs/2109.10282
3. OpenCV Documentation: https://docs.opencv.org/
4. Streamlit Documentation: https://docs.streamlit.io/

## 📄 License

This project is developed purely for academic and research-related purposes.

## 👤 Author

**Dhananjayan H**  
Roll No: AA.SC.P2MCA24070151  
Department of Computer Science

---

*Last Updated: November 2024*
