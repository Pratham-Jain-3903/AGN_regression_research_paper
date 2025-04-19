import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from pycaret.regression import *

def load_and_analyze_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    print("Dataset Shape:", df.shape)
    
    # Basic EDA
    print("\nBasic Information:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())
    
    return df

def visualize_data(df):
    # Create viz directory if it doesn't exist
    os.makedirs('viz', exist_ok=True)
    
    # Get numerical columns (excluding 'id' and 'target')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'target']]
    
    # Calculate grid dimensions
    n_cols = len(numeric_cols)
    n_rows = (n_cols - 1) // 3 + 1  # 3 columns per row
    n_cols_per_row = 3
    
    # Create subplots for feature distributions
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols_per_row, i + 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('viz/feature_distributions.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('viz/correlation_heatmap.png')
    plt.close()
    
    # Save correlation matrix to CSV
    correlation_matrix.to_csv('viz/correlation_matrix.csv')

def calculate_mutual_info(X, y):
    # Calculate mutual information scores for regression
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    
    # Plot mutual information scores
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mi_df, x='MI_Score', y='Feature')
    plt.title('Mutual Information Scores')
    plt.show()
    
    return mi_df

def recursive_feature_selection(X, y, mi_threshold=0.1):
    """
    Select features using RFE based on mutual information scores
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features
    y : pandas Series
        Target variable
    mi_threshold : float
        Minimum mutual information score threshold (default: 0.1)
    """
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    # Determine number of features to select based on MI scores
    n_features_to_select = sum(mi_df['MI_Score'] >= mi_threshold)
    n_features_to_select = max(3, min(n_features_to_select, 50))  # Keep between 3 and 20 features
    
    # Perform RFE with selected number of features
    rfe = RFE(
        estimator=RandomForestRegressor(random_state=42),
        n_features_to_select=n_features_to_select
    )
    rfe.fit(X, y)
    
    # Create DataFrame with selection results
    selected_features = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    # Save selection results
    os.makedirs('viz', exist_ok=True)
    selected_features.to_csv('viz/feature_selection_results.csv', index=False)
    
    # Plot feature selection results
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=selected_features, x='MI_Score', y='Ranking', 
                   hue='Selected', size='Selected')
    plt.title('Feature Selection Results')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('RFE Ranking (1 = Selected)')
    plt.savefig('viz/feature_selection_plot.png')
    plt.close()
    
    print(f"\nSelected {n_features_to_select} features based on MI threshold {mi_threshold}")
    return selected_features[selected_features['Selected']]['Feature'].tolist()

def train_models_pycaret(df, target_column, selected_features=None):
    # Initialize PyCaret regression setup
    if selected_features:
        data = df[selected_features + [target_column]]
    else:
        data = df
        
    reg = setup(data=data, target=target_column, silent=True, use_gpu=True, session_id=42)
    
    # Compare regression models
    best_models = compare_models(n_select=3, sort='R2')  # Using R2 as metric
    
    # Create viz directory if it doesn't exist
    os.makedirs('viz', exist_ok=True)
    
    # Save model performance plots
    for i, model in enumerate(best_models, 1):
        # Plot and save feature importance
        plot_model(model, plot='feature', save=True)
        plt.savefig(f'viz/feature_importance_model_{i}.png')
        
        # Plot and save residuals
        plot_model(model, plot='residuals', save=True)
        plt.savefig(f'viz/residuals_model_{i}.png')
        
        # Plot and save prediction error
        plot_model(model, plot='error', save=True)
        plt.savefig(f'viz/prediction_error_model_{i}.png')
        
        # Save the model
        save_model(model, f'viz/model_{i}')
    
    return best_models

def evaluate_models_on_test(best_models, test_data_path, selected_features=None):
    """Evaluate trained models on test data"""
    try:
        # Load and preprocess test data
        test_data = pd.read_csv(test_data_path)
        test_data = test_data.fillna(test_data.mean())
        
        # Create viz/test_results directory
        os.makedirs('viz/test_results', exist_ok=True)
        
        test_results = []
        for i, model in enumerate(best_models, 1):
            # Make predictions
            predictions = predict_model(model, data=test_data)
            
            # Save predictions
            predictions.to_csv(f'viz/test_results/test_predictions_model_{i}.csv', index=False)
            
            # Create prediction visualizations
            plt.figure(figsize=(10, 6))
            plt.hist(predictions['prediction_label'], bins=50)
            plt.title(f'Distribution of Predictions - Model {i}')
            plt.xlabel('Predicted Values')
            plt.ylabel('Frequency')
            plt.savefig(f'viz/test_results/predictions_dist_model_{i}.png')
            plt.close()
            
            # Store results
            result = {
                'Model_Number': i,
                'Model_Name': str(model),
                'Predictions_Mean': predictions['prediction_label'].mean(),
                'Predictions_Std': predictions['prediction_label'].std(),
                'Predictions_Min': predictions['prediction_label'].min(),
                'Predictions_Max': predictions['prediction_label'].max()
            }
            test_results.append(result)
            
            print(f"\nModel {i} Test Predictions Summary:")
            print(predictions['prediction_label'].describe())
        
        # Save test results summary
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv('viz/test_results/test_evaluation_summary.csv', index=False)
        
        return test_results_df
        
    except Exception as e:
        print(f"Error evaluating models on test data: {str(e)}")
        return None

def main():
    # Load and analyze data
    file_path = 'D:\\College\\agn\\agn\\train.csv'  # Updated path to your training data
    df = load_and_analyze_data(file_path)
    
    # Visualize data
    visualize_data(df)
    
    # Prepare features and target
    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean())  # Added missing value handling
    
    # Calculate mutual information
    mi_scores = calculate_mutual_info(X, y)
    print("\nMutual Information Scores:")
    print(mi_scores)
    
    # Select features recursively
    selected_features = recursive_feature_selection(X, y, mi_threshold=0.1)
    print("\nSelected Features:")
    print(selected_features)
    
    # Train models using PyCaret regression
    best_models = train_models_pycaret(df, target_column, selected_features)
    
    # Create evaluation results DataFrame for training
    results = []
    for i, model in enumerate(best_models, 1):
        model_results = pull()  # Get model metrics
        model_results['Model_Number'] = i
        model_results['Model_Name'] = str(model)
        results.append(model_results)
    
    # Save training evaluation results
    results_df = pd.concat(results)
    results_df.to_csv('viz/model_evaluation_results.csv', index=False)
    
    # Evaluate models on test data
    test_data_path = 'D:\\College\\agn\\agn\\test.csv'
    test_results = evaluate_models_on_test(best_models, test_data_path, selected_features)
    
    if test_results is not None:
        # Create and save comparison plot for test predictions
        plt.figure(figsize=(12, 6))
        sns.barplot(data=test_results, x='Model_Name', y='Predictions_Mean')
        plt.xticks(rotation=45)
        plt.title('Model Comparison - Test Predictions Mean')
        plt.tight_layout()
        plt.savefig('viz/test_results/test_predictions_comparison.png')
    
    print("\nModel evaluation completed. Results saved in 'viz' directory.")

if __name__ == "__main__":
    main()