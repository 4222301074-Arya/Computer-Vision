# EMNIST Letter Classification using HOG + SVM

## Overview
This project implements a handwritten letter classification system using the **EMNIST Letters Dataset**, **Histogram of Oriented Gradients (HOG)** feature extraction, and **Support Vector Machine (SVM)** classification.

The notebook demonstrates the complete machine learning workflow:
- Dataset preparation
- Feature extraction using HOG
- Model training using SVM
- Hyperparameter tuning with GridSearchCV
- Model evaluation using accuracy, classification report, and confusion matrix

---

## Dataset
The project uses the **EMNIST Letters** dataset.

Dataset contents:
- Handwritten English alphabet letters (a-z)
- Grayscale images with size **28x28 pixels**
- Labels converted into classes from **0–25**

Dataset files used:
```bash
emnist-letters-train-images-idx3-ubyte
emnist-letters-train-labels-idx1-ubyte
```

---

## Project Structure

```bash
EMNIST_Letter_Classification_HOG_SVM/
│
├── dataset/
│   ├── emnist-letters-train-images-idx3-ubyte
│   └── emnist-letters-train-labels-idx1-ubyte
│
├── output/
│   ├── Confusion_Matrix.png
│   └── Result.png
│
└── EMNIST_Letter_Classification_HOG_SVM.ipynb
```

---

## Libraries Used

Install required libraries:

```bash
pip install numpy matplotlib seaborn scikit-image scikit-learn mlxtend
```

Libraries:
- NumPy
- Matplotlib
- Seaborn
- Scikit-image
- Scikit-learn
- Mlxtend

---

## Methodology

### 1. Dataset Preparation
The dataset is loaded using `loadlocal_mnist()`.

Steps performed:
- Load EMNIST images and labels
- Convert labels from 1–26 into 0–25
- Randomly select 100 samples for each class
- Split dataset into training and testing sets

Train-test split:
- 80% Training Data
- 20% Testing Data

---

### 2. Feature Extraction using HOG
HOG (Histogram of Oriented Gradients) is used to extract image features.

Parameters:

```python
orientations = 8
pixels_per_cell = (4, 4)
cells_per_block = (2, 2)
```

HOG captures:
- Edge direction
- Shape information
- Local gradient patterns

The extracted features are then used as input for the SVM classifier.

---

### 3. Classification using SVM
The classification model uses **Support Vector Machine (SVM)**.

Hyperparameter tuning is performed using:

```python
GridSearchCV
```

Parameter search:

```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
```

The best model is selected automatically based on cross-validation performance.

---

## Evaluation
The trained model is evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix

Example evaluation:

```python
print("Accuracy :", accuracy_score(y_test, y_test_pred))
```

The confusion matrix visualization helps analyze prediction performance for each letter class.

---

## Output

### Result Visualization
The notebook generates:
- Predicted classification results
- Confusion matrix heatmap
- HOG feature visualization

Output images:

```bash
output/Result.png
output/Confusion_Matrix.png
```

---

## How to Run

### 1. Clone Repository

```bash
git clone <repository-url>
cd EMNIST_Letter_Classification_HOG_SVM
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib seaborn scikit-image scikit-learn mlxtend
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
EMNIST_Letter_Classification_HOG_SVM.ipynb
```

---

## Workflow Summary

```text
EMNIST Dataset
       ↓
Dataset Preparation
       ↓
HOG Feature Extraction
       ↓
SVM Training
       ↓
GridSearchCV Optimization
       ↓
Model Evaluation
       ↓
Prediction Results
```

---

## Advantages of HOG + SVM

### HOG Advantages
- Good for edge detection
- Effective for shape-based recognition
- Lightweight feature extraction

### SVM Advantages
- Strong classification performance
- Effective in high-dimensional spaces
- Works well with small-to-medium datasets

---

## Future Improvements
Possible future developments:
- Add CNN-based deep learning comparison
- Real-time handwritten letter recognition
- Data augmentation
- Model export for deployment
- GUI-based prediction system

---

## Author
Created for Computer Vision and Machine Learning study purposes.

---

## License
This project is intended for educational and research purposes.
