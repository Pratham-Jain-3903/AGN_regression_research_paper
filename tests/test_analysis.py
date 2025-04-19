import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import (
    load_and_analyze_data, 
    visualize_data, 
    FeatureSelector, 
    ModelTrainer,
    configure_paths
)

@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory structure"""
    (tmp_path / "data").mkdir()
    (tmp_path / "viz" / "eda").mkdir(parents=True)
    (tmp_path / "viz" / "feature_selection").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    return tmp_path

def test_load_and_analyze_data(sample_data, temp_dir):
    """Test data loading and EDA functionality"""
    # Save sample data
    data_path = temp_dir / "data" / "test.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Test loading
    df = load_and_analyze_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_data.shape
    assert all(df.columns == sample_data.columns)

def test_feature_selector(sample_data):
    """Test feature selection pipeline"""
    selector = FeatureSelector(target_col='target')
    
    # Test preprocessing
    X_scaled, y = selector.preprocess(sample_data)
    assert isinstance(X_scaled, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X_scaled.shape[0] == sample_data.shape[0]
    
    # Test feature selection
    selected_features = selector.select_features(X_scaled, y)
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0
    assert all(feat in sample_data.columns for feat in selected_features)

def test_model_trainer(sample_data):
    """Test model training pipeline"""
    trainer = ModelTrainer(target_col='target')
    
    # Test environment setup
    exp = trainer.setup_environment(sample_data)
    assert exp is not None
    
    # Test model training
    best_models = trainer.train_models()
    assert isinstance(best_models, list)
    assert len(best_models) > 0

def test_data_visualization(sample_data, temp_dir):
    """Test visualization functions"""
    try:
        visualize_data(sample_data, target_col='target')
        
        # Check if visualization files were created
        viz_path = Path(temp_dir) / "viz" / "eda"
        assert (viz_path / "feature_distributions.png").exists()
        assert (viz_path / "filtered_correlation_matrix.png").exists()
        assert (viz_path / "top_features_pairplot.png").exists()
    except Exception as e:
        pytest.fail(f"Visualization failed: {str(e)}")

def test_end_to_end_pipeline(sample_data, temp_dir):
    """Test complete pipeline execution"""
    # Save train and test data
    train_data = sample_data.iloc[:80]
    test_data = sample_data.iloc[80:]
    
    train_path = temp_dir / "data" / "train.csv"
    test_path = temp_dir / "data" / "test.csv"
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    try:
        # Configure paths
        configure_paths()
        
        # Load and analyze data
        train_df = load_and_analyze_data(train_path)
        visualize_data(train_df)
        
        # Feature selection
        selector = FeatureSelector()
        X_processed, y = selector.preprocess(train_df)
        selected_features = selector.select_features(X_processed, y)
        
        # Model training
        trainer = ModelTrainer()
        trainer.setup_environment(train_df, selected_features)
        best_models = trainer.train_models()
        
        assert len(best_models) > 0
        assert Path("models").exists()
        assert Path("viz/eda").exists()
        assert Path("viz/feature_selection").exists()
        
    except Exception as e:
        pytest.fail(f"Pipeline failed: {str(e)}")

@pytest.mark.parametrize("invalid_data", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({'feature_1': [], 'target': []}),  # Empty columns
    None  # None input
])
def test_error_handling(invalid_data):
    """Test error handling for invalid inputs"""
    with pytest.raises(Exception):
        if invalid_data is not None:
            load_and_analyze_data(invalid_data)
            visualize_data(invalid_data)