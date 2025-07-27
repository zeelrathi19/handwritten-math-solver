Handwritten Mathematical Equation Solver Using AI

An intelligent system that recognizes and solves handwritten mathematical equations using Convolutional Neural Networks (CNN) and computer vision techniques.

-> Overview

This project implements an end-to-end solution for analyzing and solving handwritten mathematical expressions using artificial intelligence. The system can process images containing handwritten mathematical equations, recognize individual digits and operators, and automatically compute the results.

-> Features

- Character Recognition: Recognizes digits (0-9) and operators (+, -, ×)
- Image Processing: Advanced preprocessing including noise reduction, binarization, and           segmentation
- CNN-based Classification: Deep learning model with 98.5% accuracy
- Real-time Solving: Automatic evaluation of recognized mathematical expressions
- Robust Segmentation: Handles connected and overlapping characters
- CROHME Dataset: Trained on Competition on Recognition of Online Handwritten Mathematical        Expressions dataset

-> Supported Operations

- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators

-> Technology Stack

- Python 3.7+
- TensorFlow/Keras - Deep learning framework
- OpenCV - Computer vision and image processing
- NumPy - Numerical computations
- PIL (Pillow) - Image handling
- Pandas - Data manipulation
- Matplotlib - Visualization

-> Requirements

- tensorflow>=2.0.0
- keras>=2.4.0
- opencv-python>=4.5.0
- numpy>=1.19.0
- pillow>=8.0.0
- pandas>=1.3.0
- matplotlib>=3.3.0
  
-> Installation

1. Clone the repository
git clone https://github.com/yourusername/handwritten-math-solver.git
cd handwritten-math-solver

2. Create a virtual environment (recommended)
python -m venv math_solver_env
source math_solver_env/bin/activate  # On Windows: math_solver_env\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Download the pre-trained model

- Ensure model_final.json and model_final.h5 are in the project directory
- Or train your own model using the provided dataset

-> Usage
Quick Start

1. Prepare your image

- Place your handwritten mathematical equation image in the project directory
- Supported formats: PNG, JPG, JPEG
- Ensure clear handwriting with good contrast

2. Run the solver
python math_solver.py

3. View results

- The system will display the recognized equation
- Automatic evaluation and result will be shown


Example Usage
# Load and process image

# The system will:
# 1. Preprocess the image
# 2. Segment individual characters
# 3. Classify each character using CNN
# 4. Reconstruct the equation
# 5. Evaluate the mathematical expression

# Output example:
# "The evaluation of the image gives equation: 2+3*4"
# "The evaluation of the image gives --> 2+3*4 = 14"

System Architecture
1. Image Preprocessing

- RGB to Grayscale conversion
- Binarization using Otsu's method
- Noise reduction and filtering
- Character segmentation using contour detection

2. Feature Extraction

- Contour analysis for character boundaries
- Bounding box detection for individual symbols
- Image normalization to 28×28 pixels
- Connected component analysis

3. CNN Classification

- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Output layer with 13 classes (0-9, +, -, ×)

4. Equation Reconstruction & Solving

- Character sequence assembly
- Mathematical expression parsing
- Automatic evaluation using Python's eval()

-> Model Performance

- Training Accuracy: 98.5%
- Dataset Size: 47,504 images
- Classes: 13 (digits 0-9, operators +, -, ×)
- Image Size: 28×28 grayscale
- Training Method: CNN with multiple conv-pool layers

-> Dataset Information

- Source: Modified CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions)
- Total Images: 47,504
- Image Format: 28×28 grayscale JPG files
- Classes Distribution:

   Digits 0-9: ~4,000 images each
   Operators (+, -, ×): ~3,000-4,000 images each


- Preprocessing: Balanced dataset to prevent bias

Training Process

Data Preprocessing Steps:

- Image Loading: Read images from organized folders
- Inversion: Convert white background to black
- Thresholding: Binary image conversion
- Contour Detection: Find character boundaries
- Cropping: Extract individual characters
- Resizing: Normalize to 28×28 pixels
- Label Assignment: Map to corresponding classes

-> Model Training:
# CNN Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(1,28,28)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(13, activation='softmax')  # 13 classes
])

-> Performance Optimization

Efficient Segmentation: Smart contour analysis to handle overlapping characters
Optimized CNN: Balanced model complexity for accuracy vs speed
Memory Management: Efficient image processing pipeline
Error Handling: Robust preprocessing for various image qualities

-> Future Enhancements

 Extended Operators: Division (÷), equals (=), parentheses
 Fraction Recognition: Support for complex fractions
 Multi-line Equations: Handle multiple equation lines
 Handwriting Styles: Improve recognition across different writing styles
 Mobile App: Deploy as mobile application
 Real-time Processing: Live camera-based equation solving
 Advanced Math: Support for algebraic expressions and calculus
 Uncertainty Estimation: Confidence scores for predictions

-> Contributing
Contributions are welcome! Here's how you can help:

- Expand Dataset: Add more handwriting samples
- Improve Accuracy: Enhance the CNN model
- Add Features: Implement new mathematical operations
- Optimize Performance: Speed up processing
- Better UI: Create a graphical user interface

-> Steps to Contribute:

- Fork the repository
- Create a feature branch (git checkout -b feature/new-operator)
- Commit your changes (git commit -am 'Add division operator support')
- Push to the branch (git push origin feature/new-operator)
- Create a Pull Request

-> Research Background
This project is based on cutting-edge research in:

- Computer Vision: Image processing and character recognition
- Deep Learning: Convolutional Neural Networks for classification
- Pattern Recognition: Handwriting analysis and feature extraction
- Digital Document Analysis: Mathematical expression understanding

-> Key References:

- CROHME Competition datasets and methodologies
- CNN architectures for character recognition
- Image preprocessing techniques for handwritten text
- Mathematical expression parsing and evaluation

-> Limitations

- Simple Expressions: Currently supports basic arithmetic only
- Single Line: Works with single-line equations
- Clear Handwriting: Requires reasonably legible handwriting
- Limited Operators: Supports +, -, × operators only
- No Variables: Does not handle algebraic variables
- Image Quality: Sensitive to image quality and lighting

-> Author
Zeel Rathi

M.Sc. (Integrated) AIML/Data Science Student
Gujarat University
Department of AIML & Data Science
School of Emerging Science and Technology

-> Acknowledgments

- CROHME Competition for providing the handwritten math dataset
- Gujarat University for academic support and resources
- TensorFlow/Keras Team for the deep learning framework
- OpenCV Community for computer vision tools
- Research Supervisors: Dr. Ravi Gor and Rashmi Ma'am for guidance


-> Project Metrics

Dataset Processing: 47,504 images processed
Model Training: 98.5% accuracy achieved
Testing: Comprehensive evaluation on diverse handwriting samples


⭐ If this project helped you, please give it a star! ⭐
This project was developed as part of B.Sc. dissertation for M.Sc. (Integrated) Five Years Program in AIML/Data Science at Gujarat University, June 2022.
