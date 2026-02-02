"""
AI/ML Learning Guide for Indic-OCR Project
=============================================

A beginner-friendly guide to understand AI, Machine Learning, 
and Deep Learning concepts used in the OCR project.

Author: Learning Module
Last Updated: February 2026
"""

# ============================================================================
# PART 1: FUNDAMENTAL CONCEPTS
# ============================================================================

"""
1. WHAT IS ARTIFICIAL INTELLIGENCE (AI)?
========================================

AI is the broad field of creating machines/software that can:
- Learn from data
- Make decisions
- Solve problems
- Perform tasks that normally require human intelligence

Examples:
- ChatGPT (language understanding)
- Self-driving cars (object detection)
- Netflix recommendations (pattern recognition)
- Indic-OCR (text recognition)


2. WHAT IS MACHINE LEARNING (ML)?
==================================

Machine Learning is a subset of AI that focuses on:
- Learning patterns from data (without explicit programming)
- Making predictions based on learned patterns
- Improving performance through experience

Instead of programming every rule, you:
1. Collect data
2. Train a model on that data
3. Use the model to make predictions

Real example from your project:
- Traditional approach: Code rules for every character in every script
- ML approach: Show the model thousands of character images and let it learn


3. WHY IS ML BETTER THAN TRADITIONAL PROGRAMMING?
===================================================

Traditional Programming:
    Input ─→ [Explicit Rules Written by Humans] ─→ Output
    Problem: Impossible to write rules for all variations

Machine Learning:
    Data ─→ [Learning Algorithm] ─→ Model ─→ Predictions
    Benefit: Learns from examples automatically


4. KEY TERMS YOU'LL ENCOUNTER
=============================

DATA:
    The information used to train models
    Example: 1,000 images of the letter "क" (Devanagari)

FEATURES:
    The characteristics extracted from data
    Example: Pixel values, edges, corners in an image

LABEL:
    The correct answer for training data
    Example: Image of "क" has label = "क"

MODEL:
    The learned pattern from data
    Like a "brain" that makes predictions

TRAINING:
    The process of teaching the model using data
    
PREDICTION/INFERENCE:
    Using the trained model to make new predictions
    Example: Show your model a new Hindi image and it predicts the text

ACCURACY:
    How correct the model is
    Example: 95% accuracy = 95 out of 100 predictions are correct
"""

# ============================================================================
# PART 2: TYPES OF MACHINE LEARNING
# ============================================================================

"""
THREE MAIN TYPES OF ML:
=======================

1. SUPERVISED LEARNING (Most Common)
    - You have data with correct answers (labels)
    - Model learns to predict the answer
    
    Example: Script Classification in your project
    - You show: Image of Hindi text ─→ Label: "Devanagari"
    - Model learns to recognize Devanagari images
    
    Types:
    a) Classification: Predict a category (e.g., which script?)
    b) Regression: Predict a number (e.g., temperature, price)


2. UNSUPERVISED LEARNING
    - You have data WITHOUT correct answers
    - Model finds hidden patterns
    
    Example: Grouping similar documents
    - You don't tell it "these are news, these are blogs"
    - It discovers these groups itself


3. REINFORCEMENT LEARNING
    - Agent learns by trial and error with rewards/penalties
    
    Example: Game AI
    - Agent plays game, gets rewarded for good moves
    - Learns to play better through experience


FOR YOUR INDIC-OCR PROJECT:
===========================

You're mainly using:
✓ SUPERVISED LEARNING for script classification
✓ SUPERVISED LEARNING for character recognition
"""

# ============================================================================
# PART 3: DEEP LEARNING (DL)
# ============================================================================

"""
WHAT IS DEEP LEARNING?
=======================

Deep Learning is a subset of Machine Learning that:
- Uses multiple layers of artificial "neurons"
- Mimics how the human brain works
- Automatically finds important features

WHY "DEEP"?
- It has many layers (deep = many layers)
- Each layer extracts increasingly complex patterns

LAYERS IN A NEURAL NETWORK:
===========================

Input Layer:
    Receives raw data (e.g., pixel values of image)
    
Hidden Layers:
    Process and transform the data
    Layer 1: Detects simple patterns (edges, corners)
    Layer 2: Combines to detect shapes
    Layer 3: Combines to detect objects (letters)
    
Output Layer:
    Produces final prediction
    Example: "This is the letter क"


VISUAL EXAMPLE:
===============

Image of "क" (50x50 pixels)
    ↓
[Input Layer: 2500 pixels values]
    ↓
[Hidden Layer 1: Detects edges & corners]
    ↓
[Hidden Layer 2: Detects shapes & strokes]
    ↓
[Hidden Layer 3: Detects characters]
    ↓
[Output Layer: क with 95% confidence]


HOW DOES IT LEARN?
===================

1. Initialize: Start with random weights (like untrained neurons)

2. Forward Pass: Send input through layers to get prediction
   Input → Layer 1 → Layer 2 → Layer 3 → Prediction

3. Calculate Error: Compare prediction to correct answer
   Error = |Prediction - Actual|

4. Backward Pass: Send error back through layers
   Update weights to reduce error (this is called BACKPROPAGATION)

5. Repeat: Do steps 2-4 many times until model improves


WHY DEEP LEARNING FOR OCR?
===========================

✓ Can learn complex patterns automatically
✓ Works well with images
✓ Better accuracy than traditional methods
✓ Can handle variations (handwriting, different fonts, etc.)
"""

# ============================================================================
# PART 4: CONVOLUTIONAL NEURAL NETWORKS (CNN)
# ============================================================================

"""
WHAT IS CNN?
============

CNN is a type of Deep Learning model specifically designed for images.

WHY NOT JUST USE REGULAR NEURAL NETWORKS?
==========================================

Regular Neural Network:
    - Treats image as flat list of pixels
    - Loses spatial relationships
    - Requires too many weights
    
CNN:
    - Preserves spatial relationships (nearby pixels matter)
    - Uses "convolution" to find patterns
    - More efficient


HOW CNN WORKS?
==============

1. CONVOLUTION LAYER
   ─────────────────
   - Slides small filter over image
   - Calculates pattern matches
   - Finds edges, corners, textures
   
   Visual example:
   
   Input Image:          Filter (looking for edges):    Output (feature map):
   ┌─────────────┐      ┌──────┐                       ┌──────┐
   │ 0 0 0 0 0   │      │1  0 -1│                      │ 1 0  │
   │ 0 1 2 1 0   │  +   │1  0 -1│  =                   │ 2 1  │
   │ 0 2 4 2 0   │      │1  0 -1│                      │ 2 0  │
   │ 0 1 2 1 0   │      └──────┘                       └──────┘
   │ 0 0 0 0 0   │
   └─────────────┘

2. ACTIVATION LAYER (ReLU)
   ──────────────────────
   - Adds non-linearity
   - Helps model learn complex patterns
   - Simple formula: if x > 0, keep it; if x < 0, make it 0


3. POOLING LAYER
   ──────────────
   - Reduces image size
   - Keeps important information
   - Makes computation faster
   
   Example (Max Pooling):
   Input:           Output:
   ┌──────┐         ┌──┐
   │2 4 1 │         │4 1│
   │1 3 2 │    →    │3 5│
   │3 5 2 │         └──┘
   │1 0 1 │
   └──────┘


CNN ARCHITECTURE FOR YOUR PROJECT:
===================================

Your image goes through:

Input: Hindi Text Image (224 x 224 pixels)
    ↓
Conv Layer 1: Find basic features (edges) → 32 filters
    ↓
Pooling: Reduce size
    ↓
Conv Layer 2: Find mid-level features (strokes) → 64 filters
    ↓
Pooling: Reduce size
    ↓
Conv Layer 3: Find high-level features (shapes) → 128 filters
    ↓
Pooling: Reduce size
    ↓
Flatten: Convert to 1D array
    ↓
Fully Connected Layer: Learn which script (3 options)
    ↓
Output: "Devanagari" with 95% confidence
"""

# ============================================================================
# PART 5: TRAINING PROCESS (HANDS-ON UNDERSTANDING)
# ============================================================================

"""
UNDERSTANDING TRAINING WITH AN ANALOGY
========================================

Learning to Identify Scripts is like:

HUMAN LEARNING:
1. See 100 examples of Devanagari text → Get feeling for it
2. See 100 examples of Tamil text → Understand differences
3. See 100 examples of Malayalam text → Recognize all three
4. Practice on new examples → Improve accuracy

ML TRAINING:
1. Load 100 Devanagari images with label "Devanagari"
2. Load 100 Tamil images with label "Tamil"
3. Load 100 Malayalam images with label "Malayalam"
4. Feed them to the model
5. Model adjusts internal weights
6. Test on new images → Check accuracy


KEY METRICS TO UNDERSTAND:
===========================

EPOCH:
    One complete pass through all training data
    Example: 1 epoch = model sees all 300 images once
    
    Why multiple epochs?
    - First epoch: Random performance (untrained)
    - Second epoch: Better (starting to learn)
    - Tenth epoch: Good accuracy (well-trained)
    - 100th epoch: Might be overfitting (too tailored to training data)


BATCH SIZE:
    How many images to show before updating weights
    
    Small batch (32 images):
    - More updates = slower training
    - Noisier = might miss some patterns
    
    Large batch (256 images):
    - Fewer updates = faster training
    - Smoother = more stable


LEARNING RATE:
    How much to adjust weights at each step
    
    Too small (0.00001):
    - Very slow training
    - Takes forever to improve
    
    Too large (1.0):
    - Jumps over good solutions
    - Might not converge
    
    Just right (0.001):
    - Steady improvement
    - Reaches good accuracy


LOSS vs ACCURACY:
=================

LOSS: How wrong the model is
    - Measured: Usually ranges from 0 to 1+
    - Goal: Make it as SMALL as possible
    - If loss = 0.05: Model is very confident and correct

ACCURACY: How many predictions are correct
    - Measured: Percentage (0-100%)
    - Goal: Make it as LARGE as possible
    - If accuracy = 95%: 95 out of 100 predictions correct


DURING TRAINING, YOU'LL SEE:
===========================

Epoch 1:  Loss = 2.34, Accuracy = 40%
Epoch 2:  Loss = 1.89, Accuracy = 58%
Epoch 3:  Loss = 1.23, Accuracy = 72%
Epoch 4:  Loss = 0.89, Accuracy = 82%
Epoch 5:  Loss = 0.56, Accuracy = 91%
...
Epoch 50: Loss = 0.12, Accuracy = 96%

✓ Loss decreasing = Model improving
✓ Accuracy increasing = Model improving
"""

# ============================================================================
# PART 6: YOUR PROJECT COMPONENTS EXPLAINED
# ============================================================================

"""
YOUR INDIC-OCR PIPELINE:
=========================

COMPONENT 1: IMAGE PREPROCESSING
─────────────────────────────────

WHY DO WE NEED IT?
    Real-world images are messy:
    - Different lighting
    - Noise/graininess
    - Uneven background
    - Skewed/rotated text
    
    Models work best with clean, normalized data


WHAT IT DOES:

a) GRAYSCALE CONVERSION
   Color Image (RGB) → Black & White
   
   Why? 
   - Reduces complexity (3 channels → 1 channel)
   - Color doesn't matter for text
   - Faster processing

b) DENOISING
   Removes graininess/spots
   
   Image with noise:  →  Image denoised:
   ┌──────────┐         ┌──────────┐
   │█ █  ████│         │█████████│
   │████ █ ██│         │██████████│
   │██ █ ████│         │██████████│
   │████ █ ██│         │██████████│
   └──────────┘         └──────────┘

c) DESKEWING (Rotation correction)
   Rotated text → Straight text
   
   Skewed:    →  Straight:
   ╱╱╱╱╱          ─────
   ╱╱╱╱╱          ─────

d) BINARIZATION (Make it Black & White)
   Grayscale → Pure black and white
   
   Gray image:    →  Binary image:
   ┌────────┐        ┌────────┐
   │░░░▒░░░░│        │░░░█░░░░│
   │░░▒░▒░░░│   →    │░░█░█░░░│
   │░▒░░░▒░░│        │░█░░░█░░│
   │▒░░░░░▒░│        │█░░░░░█░│
   └────────┘        └────────┘

WHY THESE STEPS?
    Better preprocessing → Better model input → Better accuracy


COMPONENT 2: SCRIPT CLASSIFIER
──────────────────────────────

WHAT IT DOES:
    Looks at image and decides: Is this Devanagari, Tamil, or Malayalam?

HOW IT WORKS:
    Input: Preprocessed image
    ↓
    CNN processes the image
    ↓
    Output: 3 probabilities (confidence for each script)
    
    Example output:
    Devanagari: 92% ← Highest, so answer is "Devanagari"
    Tamil:      5%
    Malayalam:  3%


COMPONENT 3: OCR ENGINE (Text Recognition)
─────────────────────────────────────────

WHAT IT DOES:
    Takes the detected text image and extracts the actual characters

HOW IT WORKS:
    Uses advanced models like PaddleOCR or TrOCR
    
    Input: Image of "नमस्ते"
    ↓
    Character detection (where are the characters?)
    ↓
    Character recognition (what is each character?)
    ↓
    Output: Text string "नमस्ते"


COMPONENT 4: EVALUATION METRICS
────────────────────────────────

HOW DO WE KNOW IF OCR IS GOOD?

CER (Character Error Rate):
    Measures character-level accuracy
    
    Predicted: "नमस्ते"
    Actual:    "नमस्ते"
    CER = 0%   (Perfect!)
    
    Predicted: "नमस्त"
    Actual:    "नमस्ते"
    CER = 14%  (1 character missing out of 7)

WER (Word Error Rate):
    Measures word-level accuracy
    
    Predicted: "नमस्ते भारत"
    Actual:    "नमस्ते भारत"
    WER = 0%   (Perfect!)
    
    Predicted: "नमस्ते भारत खेल"
    Actual:    "नमस्ते भारत"
    WER = 50%  (1 wrong word out of 2)

ACCURACY:
    Percentage of correct predictions
    
    Tested on 100 images, got 95 correct
    Accuracy = 95%
"""

# ============================================================================
# PART 7: PRACTICAL IMPLEMENTATION GUIDE
# ============================================================================

"""
HOW TO RUN YOUR PROJECT:
=========================

STEP 1: COLLECT DATA
────────────────────
    - Find/create images of Hindi, Tamil, Malayalam text
    - Label them (e.g., "devanagari", "tamil", "malayalam")
    - Put them in: data/train/devanagari/, data/train/tamil/, etc.

STEP 2: TRAIN SCRIPT CLASSIFIER
────────────────────────────────
    python train.py --task script_classifier --epochs 50
    
    What happens:
    1. Reads all images
    2. Splits into training/validation sets
    3. For 50 epochs:
       - Shows images to model
       - Model makes predictions
       - Calculates error
       - Updates weights (learning)
    4. Saves trained model

STEP 3: TEST/INFERENCE
──────────────────────
    python inference.py --image test_image.png
    
    What happens:
    1. Loads trained model
    2. Preprocesses image
    3. Runs through classifier
    4. Runs through OCR engine
    5. Outputs recognized text

STEP 4: EVALUATE
────────────────
    Check CER/WER on test set
    If accuracy < 90%, need more training data or better model


STEP 5: DEPLOY
──────────────
    streamlit run app/streamlit_app.py
    
    Now anyone can use your OCR system!
"""

# ============================================================================
# PART 8: COMMON PROBLEMS AND SOLUTIONS
# ============================================================================

"""
PROBLEM 1: MODEL HAS LOW ACCURACY (40-50%)
============================================

Causes:
1. Insufficient training data (need 100+ images per script)
2. Model underfitting (not enough layers/parameters)
3. Bad preprocessing (images still noisy)
4. Wrong hyperparameters

Solutions:
✓ Collect more data
✓ Train for more epochs
✓ Use larger model (ResNet instead of simple CNN)
✓ Improve preprocessing


PROBLEM 2: MODEL WORKS GREAT ON TRAINING DATA BUT BAD ON NEW DATA
==================================================================

This is called OVERFITTING
The model memorized training data instead of learning general patterns

Causes:
1. Too much training data variation
2. Model too complex
3. Training for too many epochs

Solutions:
✓ Use regularization (L1/L2 penalty)
✓ Use dropout (randomly deactivate neurons)
✓ Use early stopping (stop when validation accuracy plateaus)
✓ Use data augmentation (rotate, flip images)


PROBLEM 3: TRAINING IS VERY SLOW
=================================

Causes:
1. No GPU (using CPU instead)
2. Large batch size
3. Too much preprocessing

Solutions:
✓ Use Google Colab (free GPU)
✓ Reduce batch size (but not too small)
✓ Use lighter preprocessing


PROBLEM 4: OCR ACCURACY IS LOW ON HANDWRITTEN TEXT
====================================================

Causes:
Training data was mostly printed text
Handwritten has too much variation

Solutions:
✓ Add handwritten samples to training data
✓ Use data augmentation
✓ Train separate model for handwriting
"""

# ============================================================================
# PART 9: KEY PYTHON CONCEPTS YOU'LL NEED
# ============================================================================

"""
NUMPY - Numerical Computing
============================

What: Working with arrays and matrices
Why: Deep learning uses lots of matrix operations

Example:
    import numpy as np
    
    image = np.array([[1, 2, 3],      # 2D array (matrix)
                      [4, 5, 6]])
    
    # Basic operations
    image_doubled = image * 2           # Multiply all elements
    image_mean = np.mean(image)         # Average value
    image_normalized = image / 255.0    # Normalize to 0-1


PANDAS - Data Handling
======================

What: Working with tabular data (like spreadsheets)
Why: Managing datasets, labels, evaluation results

Example:
    import pandas as pd
    
    data = pd.DataFrame({
        'image': ['image1.png', 'image2.png'],
        'label': ['hindi', 'tamil'],
        'accuracy': [0.95, 0.92]
    })
    
    # Access data
    print(data['accuracy'])  # Get accuracy column
    print(data.iloc[0])      # Get first row


OPENCV - Image Processing
==========================

What: Manipulating images (used in preprocessing)
Why: Essential for preparing images before model

Example:
    import cv2
    
    image = cv2.imread('photo.png')           # Read
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     # Denoise
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # Binarize


PYTORCH - Deep Learning Framework
==================================

What: Library for building and training neural networks
Why: Makes implementing ML models much easier

Example:
    import torch
    import torch.nn as nn
    
    # Define model
    model = nn.Sequential(
        nn.Linear(784, 128),      # Input layer
        nn.ReLU(),                # Activation
        nn.Linear(128, 10)        # Output layer
    )
    
    # Train
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        output = model(input_data)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()            # Backpropagation
        optimizer.step()           # Update weights


TENSORFLOW/KERAS - Another Deep Learning Framework
====================================================

Alternative to PyTorch
Easier to use for beginners

Example:
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=50)
"""

# ============================================================================
# PART 10: NEXT STEPS FOR LEARNING
# ============================================================================

"""
ROADMAP TO MASTERY:
====================

WEEK 1: FUNDAMENTALS (You are here!)
    ✓ Read this guide
    ✓ Understand basic ML concepts
    ✓ Watch: 3Blue1Brown Neural Networks series (YouTube)

WEEK 2: HANDS-ON PRACTICE
    ✓ Run the example notebook: notebooks/exploration.ipynb
    ✓ Experiment with preprocessing.py
    ✓ Try different parameters in train.py

WEEK 3: DEEP DIVE INTO CNN
    ✓ Understand convolution operations
    ✓ Modify the model architecture in script_classifier.py
    ✓ Train and compare results

WEEK 4: OCR SPECIFICS
    ✓ Understand how PaddleOCR works
    ✓ Fine-tune on your custom data
    ✓ Evaluate with metrics

WEEK 5-6: OPTIMIZATION & DEPLOYMENT
    ✓ Debug low accuracy issues
    ✓ Collect more data
    ✓ Deploy web app
    ✓ Write documentation


RECOMMENDED LEARNING RESOURCES:
================================

Free Online Courses:
1. Fast.ai - Practical Deep Learning for Coders
   (https://course.fast.ai/) - BEST FOR BEGINNERS
   
2. Andrew Ng's Machine Learning Course
   (Coursera) - Classic foundation course
   
3. Deep Learning Specialization by Ng
   (Coursera) - More advanced

YouTube Channels:
1. 3Blue1Brown - Beautiful visualizations of neural networks
2. StatQuest with Josh Starmer - Statistics & ML explained clearly
3. Andrej Karpathy - Deep Learning expert
4. Code.org - Quick tutorials

Books:
1. "Hands-On Machine Learning" by Aurélien Géron
2. "Deep Learning" by Goodfellow, Bengio, Courville


PRACTICE DATASETS:
==================

1. MNIST (Digits) - Start here
2. CIFAR-10 (Images) - Next level
3. ImageNet (Large) - Professional level
4. Your own collected data - Real-world learning


KAGGLE COMPETITIONS:
====================

Join kaggle.com to:
- Download datasets
- Compete with others
- Learn from solutions
- Build portfolio

Start with: Beginner-friendly competitions
"""

# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
QUICK REFERENCE CHEAT SHEET:
=============================

AI: Broad field of making intelligent machines
 ├─ Machine Learning: Learning from data
 │   ├─ Supervised: Learning with labeled data
 │   ├─ Unsupervised: Finding patterns without labels
 │   └─ Reinforcement: Learning through rewards
 └─ Deep Learning: Neural networks with many layers
     └─ CNN: Convolutional networks (best for images)

TRAINING PHASES:
    Preprocessing → Training → Validation → Testing → Deployment

NEURAL NETWORK LAYERS:
    Input → Hidden → Hidden → ... → Output
    (more hidden layers = "deeper")

IMPORTANT METRICS:
    Loss: How wrong (lower is better)
    Accuracy: How right (higher is better)
    CER: Character error rate
    WER: Word error rate

TRAINING PARAMETERS:
    Epoch: Full pass through data
    Batch: Samples shown before update
    Learning Rate: How much to adjust weights
    Dropout: Regularization to prevent overfitting

YOUR PROJECT PIPELINE:
    Raw Image → Preprocess → Classify Script → Recognize Text → Output

COMMON MISTAKES:
    ✗ Not enough data
    ✗ Overfitting to training data
    ✗ Bad preprocessing
    ✗ Wrong hyperparameters
    ✗ Not evaluating properly

REMEMBER:
    - Start simple, then add complexity
    - More data usually beats better algorithms
    - Monitor both training and validation metrics
    - Preprocessing is often more important than model choice
    - Share your work and learn from others
"""

if __name__ == "__main__":
    print("=" * 70)
    print("INDIC-OCR: AI/ML LEARNING GUIDE")
    print("=" * 70)
    print("\nThis file is a comprehensive guide to AI/ML fundamentals.")
    print("\nTo read it:")
    print("1. Open this file in a text editor")
    print("2. Read through each section")
    print("3. Run the example notebook: notebooks/exploration.ipynb")
    print("4. Experiment with the code in src/ directory")
    print("\nGood luck with your learning journey!")
    print("=" * 70)
