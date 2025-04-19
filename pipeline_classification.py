"""
Enhanced Predictive Modeling Pipeline with Advanced Feature Selection and Model Evaluation
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from pycaret.classification import *  # Change to classification
import click  # For CLI integration
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import glob
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.calibration import calibration_curve

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_paths():
    """Configure directory paths for outputs"""
    Path("viz/eda").mkdir(parents=True, exist_ok=True)
    Path("viz/feature_selection").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_and_analyze_data(file_path):
    """Load and perform comprehensive EDA"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"\nDataset Shape: {df.shape}")
        
        # Enhanced EDA
        eda_results = {
            'dtypes': pd.DataFrame(df.dtypes, columns=['Data Type']),
            'missing_values': pd.DataFrame(df.isnull().sum(), columns=['Missing Values']),
            'duplicates': pd.DataFrame({'Duplicates': [df.duplicated().sum()]}),
            'cardinality': pd.DataFrame(df.select_dtypes(include='object').nunique(), columns=['Unique Values']),
            'statistics': df.describe(percentiles=[.25, .5, .75, .95, .99])
        }
        
        # Save EDA results as separate CSV files
        for name, data in eda_results.items():
            output_path = f'viz/eda/{name}_report.csv'
            data.to_csv(output_path)
            logging.info(f"Saved {name} report to {output_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def visualize_data(df, target_col='target'):
    """Generate enhanced visualizations for classification"""
    try:
        # Verify target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
            
        # Configure style
        sns.set(style="whitegrid", palette="muted")
        plt.rcParams['figure.dpi'] = 300
        
        # Numeric features analysis
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Enhanced distribution plots with outlier detection
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4)
        
        # Target distribution for classification
        ax1 = fig.add_subplot(gs[0, :])
        sns.countplot(data=df, x=target_col, ax=ax1)
        ax1.set_title(f'Target Class Distribution')
        
        # Add class balance ratio
        class_counts = df[target_col].value_counts()
        balance_ratio = class_counts.min() / class_counts.max()
        ax1.text(0.95, 0.95, f'Class Balance Ratio: {balance_ratio:.2f}', 
                transform=ax1.transAxes, ha='right', va='top')
        
        # Feature distributions with outlier indicators
        for i, col in enumerate(numeric_cols[:4]):
            ax = fig.add_subplot(gs[1, i])
            sns.boxenplot(x=df[col], ax=ax)
            ax.set_title(f'{col}\n(Outliers: {len(df[df[col] > df[col].quantile(0.99)])})')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('viz/eda/feature_distributions.png', bbox_inches='tight')
        plt.close()
        
        # Interactive correlation matrix including target
        corr_cols = numeric_cols + [target_col]
        corr_matrix = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(20, 16))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                    annot=False, square=True, linewidths=.5)
        plt.title('Feature Correlation Matrix (Filtered)')
        plt.savefig('viz/eda/filtered_correlation_matrix.png', bbox_inches='tight')
        plt.close()
        
        # Pairplot for top correlated features
        top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:6]
        sns.pairplot(df[top_features.tolist() + [target_col]])
        plt.savefig('viz/eda/top_features_pairplot.png')
        plt.close()
        
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        raise

class FeatureSelector:
    """Advanced feature selection pipeline for classification"""
    def __init__(self, target_col='target'):
        self.target_col = target_col
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = RobustScaler()
        
    def preprocess(self, df):
        """Handle missing values and scale data"""
        try:
            # Separate features and target
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            # Impute missing values
            X_imputed = pd.DataFrame(self.imputer.fit_transform(X),
                                     columns=X.columns)
            
            # Scale features
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imputed),
                                   columns=X.columns)
            
            return X_scaled, y
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def select_features(self, X, y):
        """Hybrid feature selection strategy for classification"""
        try:
            # Stage 1: Mutual Information Filter
            mi_scores = mutual_info_classif(X, y)
            mi_threshold = np.median(mi_scores) * 1.5
            mi_mask = mi_scores > mi_threshold
            
            # Stage 2: Embedded Method (LogisticRegressionCV)
            logistic = make_pipeline(RobustScaler(), 
                                   LogisticRegressionCV(cv=5, max_iter=1000))
            logistic.fit(X, y)
            logistic_mask = logistic.named_steps['logisticregressioncv'].coef_[0] != 0
            
            # Stage 3: Recursive Feature Elimination
            combined_mask = mi_mask & logistic_mask
            X_filtered = X.loc[:, combined_mask]
            
            n_features_to_select = max(1, int(X_filtered.shape[1] * 0.5))
            
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                n_features_to_select=n_features_to_select,
                step=0.1
            )
            
            # Initialize RFE results for all features
            rfe_support = np.zeros(X.shape[1], dtype=bool)
            rfe_ranking = np.ones(X.shape[1], dtype=int) * -1
            
            # Fit RFE only on filtered features and map results back
            if X_filtered.shape[1] > 0:
                rfe.fit(X_filtered, y)
                filtered_indices = X.columns.get_indexer(X_filtered.columns)
                rfe_support[filtered_indices] = rfe.support_
                rfe_ranking[filtered_indices] = rfe.ranking_
            
            # Compile results with aligned lengths
            selection_report = pd.DataFrame({
                'Feature': X.columns,
                'MI_Score': mi_scores,
                'Logistic_Selected': logistic_mask,
                'RFE_Selected': rfe_support,
                'RFE_Ranking': rfe_ranking
            }).sort_values('MI_Score', ascending=False)
            
            selected_features = selection_report[selection_report['RFE_Selected']]
            self.save_selection_report(selection_report)
            
            return selected_features['Feature'].tolist()
            
        except Exception as e:
            logging.error(f"Feature selection failed: {str(e)}")
            raise
    
    def save_selection_report(self, report):
        """Save feature selection results with visualizations"""
        try:
            # Save detailed report
            report.to_csv('viz/feature_selection/feature_selection_report.csv', index=False)
            
            # Visualize selection process
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=report, x='MI_Score', y='RFE_Ranking',
                           hue='RFE_Selected', size='Logistic_Selected',
                           palette='viridis', sizes=(50, 150))
            plt.title('Feature Selection Landscape')
            plt.xlabel('Mutual Information Score')
            plt.ylabel('RFE Ranking')
            plt.savefig('viz/feature_selection/feature_selection_landscape.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Failed to save selection report: {str(e)}")
            raise

class ModelTrainer:
    """Enhanced model training with PyCaret for classification and imbalance handling"""
    def __init__(self, target_col='target'):
        self.target_col = target_col
        self.best_models = []
        
    def analyze_class_imbalance(self, data):
        """Analyze class distribution and recommend resampling strategy"""
        try:
            class_dist = data[self.target_col].value_counts()
            imbalance_ratio = class_dist.min() / class_dist.max()
            
            # Create class distribution plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=class_dist.index, y=class_dist.values)
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.savefig('viz/eda/class_distribution.png')
            plt.close()
            
            # Log imbalance statistics
            logging.info(f"Class distribution: {class_dist.to_dict()}")
            logging.info(f"Imbalance ratio: {imbalance_ratio:.3f}")
            
            # Recommend sampling strategy
            if imbalance_ratio < 0.2:
                return 'severe_imbalance'
            elif imbalance_ratio < 0.5:
                return 'moderate_imbalance'
            else:
                return 'mild_imbalance'
                
        except Exception as e:
            logging.error(f"Failed to analyze class imbalance: {str(e)}")
            raise
    
    def setup_environment(self, data, selected_features=None):
        """Configure PyCaret with imbalance handling"""
        try:
            if selected_features:
                data = data[selected_features + [self.target_col]]
            
            # Analyze imbalance
            imbalance_severity = self.analyze_class_imbalance(data)
            
            # Configure fix_imbalance and fix_imbalance_method based on severity
            fix_imbalance = False
            fix_imbalance_method = None
            
            if imbalance_severity == 'severe_imbalance':
                fix_imbalance = True
                # Use SMOTEENN for severe imbalance (combines SMOTE and ENN)
                fix_imbalance_method = 'smoteenn'
                logging.info("Using SMOTEENN for severe class imbalance")
            elif imbalance_severity == 'moderate_imbalance':
                fix_imbalance = True
                # Use SMOTE for moderate imbalance
                fix_imbalance_method = 'smote'
                logging.info("Using SMOTE for moderate class imbalance")
            else:
                # Use class weights for mild imbalance
                logging.info("Using class weights for mild class imbalance")
            
            exp = setup(
                data=data, 
                target=self.target_col,
                session_id=42,
                normalize=True,
                remove_outliers=True,
                outliers_threshold=0.05,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.9,
                fix_imbalance=fix_imbalance,
                fix_imbalance_method=fix_imbalance_method,
                log_experiment=False,
                experiment_name='agn_classification',
                use_gpu=False
            )
            
            return exp
            
        except Exception as e:
            logging.error(f"PyCaret setup failed: {str(e)}")
            raise
    
    def train_models(self):
        """Train and optimize models with imbalance consideration"""
        try:
            # Compare base models using appropriate metrics for imbalanced data
            top_models = compare_models(
                n_select=3,
                sort='AUC',  # AUC is more robust for imbalanced datasets
                exclude=['catboost'],
                fold=5,
                cross_validation=True
            )
            
            # Model tuning with focus on minority class
            tuned_models = []
            for model in top_models:
                tuned_model = tune_model(
                    model,
                    optimize='AUC',  # Can be changed to 'F1' or 'Kappa' for severe imbalance
                    search_algorithm='optuna',
                    early_stopping='AUC',
                    customize_scoring={
                        'F1': 'weighted',  # Use weighted F1 for imbalanced cases
                        'Precision': 'weighted',
                        'Recall': 'weighted'
                    }
                )
                tuned_models.append(tuned_model)
            
            # Ensemble methods often handle imbalance better
            blended = blend_models(
                tuned_models,
                method='soft',  # Use soft voting for probability calibration
                weights=[1, 1, 1]  # Can be adjusted based on individual model performance
            )
            
            stacked = stack_models(
                tuned_models,
                meta_model='lightgbm',  # LightGBM handles imbalance well
                restack=True
            )
            
            self.best_models = [blended, stacked] + tuned_models
            
            # Generate model insights with focus on minority class performance
            for i, model in enumerate(self.best_models):
                self.save_model_artifacts(model, i+1)
            
            return self.best_models
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise
    
    def save_model_artifacts(self, model, model_id):
        """Save model artifacts with focus on imbalance metrics"""
        try:
            save_model(model, f'models/model_{model_id}')
            
            Path("viz/models").mkdir(parents=True, exist_ok=True)
            
            # Classification plots with emphasis on class imbalance
            plot_types = [
                'feature',
                'confusion_matrix',
                'auc',
                'pr',  # Precision-Recall curve is crucial for imbalanced data
                'class_report',  # Detailed per-class metrics
                'boundary',  # Decision boundary visualization
                'learning',  # Learning curve to check for bias
                'calibration'  # Probability calibration curve
            ]
            
            for plot_type in plot_types:
                try:
                    plot_model(
                        model,
                        plot=plot_type,
                        save=True,
                        plot_kwargs={
                            'display_format': None,
                            'scale': 1.2,
                            'plot_kwargs': {
                                'figsize': (10, 6)
                            }
                        }
                    )
                    plt.savefig(
                        f'viz/models/model_{model_id}_{plot_type}.png',
                        bbox_inches='tight',
                        dpi=300
                    )
                    plt.close()
                except Exception as e:
                    logging.warning(f"Could not generate {plot_type} plot: {str(e)}")
        
        except Exception as e:
            logging.error(f"Failed to save model artifacts: {str(e)}")
            raise

def compile_results_pdf():
    """Compile all results into a single PDF report"""
    try:
        logging.info("Compiling results into PDF...")
        doc = SimpleDocTemplate(
            "viz/pipeline_results.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Collect all content
        story = []
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        
        # Add title
        story.append(Paragraph("Pipeline Execution Results", title_style))
        story.append(Spacer(1, 12))
        
        # Add EDA results
        story.append(Paragraph("1. Exploratory Data Analysis", styles['Heading2']))
        for img_path in glob.glob('viz/eda/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
        # Add Feature Selection results
        story.append(Paragraph("2. Feature Selection", styles['Heading2']))
        for img_path in glob.glob('viz/feature_selection/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
        # Add Model results
        story.append(Paragraph("3. Model Evaluation", styles['Heading2']))
        for img_path in glob.glob('viz/models/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        logging.info("PDF report generated successfully at viz/pipeline_results.pdf")
        
    except Exception as e:
        logging.error(f"Failed to compile PDF report: {str(e)}")
        raise

@click.command()
@click.option('--train_path', default='data/train.csv', help='Path to training data')
@click.option('--test_path', default='data/test.csv', help='Path to test data')
def main(train_path, test_path):
    """Main pipeline execution"""
    try:
        configure_paths()
        
        # Data loading and preparation
        logging.info("Loading and analyzing data...")
        train_df = load_and_analyze_data(train_path)
        visualize_data(train_df)
        
        # Feature selection
        logging.info("Performing feature selection...")
        selector = FeatureSelector()
        X_processed, y = selector.preprocess(train_df)
        selected_features = selector.select_features(X_processed, y)
        logging.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Model training
        logging.info("Training models...")
        trainer = ModelTrainer()
        trainer.setup_environment(train_df, selected_features)
        best_models = trainer.train_models()
        
        # Model evaluation
        if Path(test_path).exists():
            logging.info("Evaluating on test data...")
            test_df = pd.read_csv(test_path)
            
            # Check if test data has the required features
            missing_features = set(selected_features) - set(test_df.columns)
            if missing_features:
                logging.warning(f"Missing features in test data: {missing_features}")
                logging.info("Using only available features for prediction")
                
            # Use intersection of selected features and available features
            test_features = list(set(selected_features) & set(test_df.columns))
            
            if not test_features:
                logging.error("No selected features found in test data")
                logging.info("Available features in test data: " + ", ".join(test_df.columns))
                logging.info("Required features: " + ", ".join(selected_features))
            else:
                test_df = test_df[test_features]
                
                # Create evaluation metrics DataFrame
                eval_metrics = pd.DataFrame()
                
                for i, model in enumerate(best_models, 1):
                    predictions = predict_model(model, data=test_df)
                    
                    # Save predictions
                    predictions.to_csv(f'predictions/test_predictions_model_{i}.csv')
                    
                    # Calculate and store metrics
                    metrics = {
                        'Model': f'model_{i}',
                        'Accuracy': accuracy_score(predictions['y_true'], predictions['y_pred']),
                        'AUC': roc_auc_score(predictions['y_true'], predictions['y_pred']),
                        'F1_weighted': f1_score(predictions['y_true'], predictions['y_pred'], average='weighted'),
                        'F1_macro': f1_score(predictions['y_true'], predictions['y_pred'], average='macro'),
                        'Precision_weighted': precision_score(predictions['y_true'], predictions['y_pred'], average='weighted'),
                        'Recall_weighted': recall_score(predictions['y_true'], predictions['y_pred'], average='weighted'),
                        'Geometric_Mean': geometric_mean_score(predictions['y_true'], predictions['y_pred']),
                        'Matthews_Corr': matthews_corrcoef(predictions['y_true'], predictions['y_pred'])
                    }
                    eval_metrics = eval_metrics.append(metrics, ignore_index=True)
                
                # Save evaluation metrics
                eval_metrics.to_csv('predictions/model_evaluation_metrics.csv', index=False)
                logging.info("Model evaluation metrics saved")
        
        # Compile results into PDF
        compile_results_pdf()
        
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()