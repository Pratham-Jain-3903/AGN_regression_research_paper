# Advanced ML Classification Pipeline

A comprehensive machine learning pipeline specialized for classification tasks with advanced feature selection, model evaluation, and class imbalance handling capabilities.

## Features

- Automated EDA with visualization
- Advanced feature selection using multiple techniques
- Intelligent class imbalance detection and handling:
  - SMOTE for moderate imbalance
  - SMOTEENN for severe imbalance
  - Class weights for mild imbalance
- Model training with PyCaret:
  - Multiple classification algorithms
  - Automated hyperparameter tuning
  - Model ensembling (Blending and Stacking)
- Classification-specific evaluations:
  - ROC-AUC curves
  - Precision-Recall curves
  - Confusion matrices
  - Class distribution analysis
- PDF report generation
- Test data prediction

## Installation

```bash
pip install -r requirements.txt
pip install imbalanced-learn
```

## Usage

```bash
python pipeline_classification.py --train_path data/train.csv --test_path data/test.csv
```

## Output Structure

```
.
├── data/
│   ├── processed/
│   │   └── classifications/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── classifications/
├── predictions/
│   ├── test_predictions_model_*.csv
│   └── prediction_distribution.csv
├── viz/
│   ├── eda/
│   │   └── classifications/
│   ├── feature_selection/
│   │   └── classifications/
│   └── models/
│       ├── model_comparison.png
│       └── prediction_distribution.png
└── pipeline_results.pdf
```

## Model Pipeline

1. **Data Loading & EDA**
   - Automated data type detection
   - Missing value analysis
   - Class distribution visualization
   - Feature correlations

2. **Feature Selection**
   - Mutual Information Classification
   - Random Forest importance
   - Recursive Feature Elimination

3. **Class Imbalance Handling**
   - Automatic imbalance detection
   - Dynamic resampling strategy selection
   - Performance validation with appropriate metrics

4. **Model Training**
   - Multiple classification algorithms
   - Grid search optimization
   - Cross-validation
   - Ensemble methods

5. **Evaluation**
   - Classification-specific metrics
   - Class-wise performance analysis
   - Prediction distribution analysis

## Branch Information

This branch (`treating_as_classification`) contains modifications specific to classification tasks, including:
- Class imbalance handling
- Classification-specific metrics
- Adapted visualization tools
- Modified model selection criteria

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- pycaret
- imbalanced-learn
- reportlab (for PDF generation)