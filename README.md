# Advanced ML Pipeline

A comprehensive machine learning pipeline with advanced feature selection and model evaluation capabilities.

## Features

- Automated EDA with visualization
- Advanced feature selection using multiple techniques
- Model training with PyCaret
- PDF report generation
- Test data evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python pipeline.py --train_path data/train.csv --test_path data/test.csv
```

## Output Structure

```
.
├── data/
│   ├── processed/
│   ├── train.csv
│   └── test.csv
├── models/
├── viz/
│   ├── eda/
│   ├── feature_selection/
│   └── models/
└── pipeline_results.pdf
```