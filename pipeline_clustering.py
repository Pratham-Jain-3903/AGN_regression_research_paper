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
from sklearn.preprocessing import RobustScaler
# Updated PyCaret imports for clustering
from pycaret.clustering import (
    setup, 
    create_model,
    assign_model,
    tune_model,
    evaluate_model,
    save_model,
    plot_model
)
import click
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_paths():
    """Configure directory paths for outputs"""
    try:
        # Create base directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("viz").mkdir(parents=True, exist_ok=True)
        
        # Create clustering-specific directories
        directories = [
            "viz/clustering/eda",
            "viz/clustering/feature_selection",
            "viz/clustering/models",
            "data/processed/clustering",
            "models/clustering",
            "predictions/clustering"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
            
    except Exception as e:
        logging.error(f"Failed to create directories: {str(e)}")
        raise

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
            output_path = f'viz/clustering/eda/{name}_report.csv'
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
        plt.savefig('viz/clustering/eda/feature_distributions.png', bbox_inches='tight')
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
        plt.savefig('viz/clustering/eda/filtered_correlation_matrix.png', bbox_inches='tight')
        plt.close()
        
        # Pairplot for top correlated features
        top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:6]
        sns.pairplot(df[top_features.tolist() + [target_col]])
        plt.savefig('viz/clustering/eda/top_features_pairplot.png')
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
            report.to_csv('viz/clustering/feature_selection/feature_selection_report.csv', index=False)
            
            # Visualize selection process
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=report, x='MI_Score', y='RFE_Ranking',
                           hue='RFE_Selected', size='Logistic_Selected',
                           palette='viridis', sizes=(50, 150))
            plt.title('Feature Selection Landscape')
            plt.xlabel('Mutual Information Score')
            plt.ylabel('RFE Ranking')
            plt.savefig('viz/clustering/feature_selection/feature_selection_landscape.png')
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
            plt.savefig('viz/clustering/eda/class_distribution.png')
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
                sort='AUC',
                exclude=['catboost', 'qda'],  # Exclude problematic models
                fold=5,
                cross_validation=True
            )
            
            # Model tuning with focus on minority class
            tuned_models = []
            for model in top_models:
                # Get model name to determine appropriate parameters
                model_name = model.__class__.__name__.lower()
                
                # Define parameter grid based on model type
                if 'randomforest' in model_name:
                    custom_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                elif 'xgboost' in model_name:
                    custom_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                elif 'lightgbm' in model_name:
                    custom_grid = {
                        'n_estimators': [100, 200, 300],
                        'num_leaves': [31, 50, 100],
                        'learning_rate': [0.01, 0.1],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                else:
                    # Default grid for other models
                    custom_grid = None
                
                # Tune model with appropriate parameters
                try:
                    tuned_model = tune_model(
                        model,
                        optimize='AUC',
                        search_algorithm='grid' if custom_grid else 'random',
                        n_iter=10,
                        custom_grid=custom_grid,
                        fold=5
                    )
                    tuned_models.append(tuned_model)
                    logging.info(f"Successfully tuned {model_name}")
                except Exception as e:
                    logging.warning(f"Could not tune {model_name}: {str(e)}")
                    tuned_models.append(model)  # Use untuned model as fallback
            
            # Ensemble methods
            try:
                blended = blend_models(
                    tuned_models,
                    method='soft',
                    weights=[1] * len(tuned_models)
                )
                
                stacked = stack_models(
                    tuned_models,
                    meta_model='lr',  # Use logistic regression as meta-model
                    restack=True
                )
                
                self.best_models = [blended, stacked] + tuned_models
            except Exception as e:
                logging.warning(f"Could not create ensemble models: {str(e)}")
                self.best_models = tuned_models
            
            # Generate model insights
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
            
            Path("viz/clustering/models").mkdir(parents=True, exist_ok=True)
            
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
                        f'viz/clustering/models/model_{model_id}_{plot_type}.png',
                        bbox_inches='tight',
                        dpi=300
                    )
                    plt.close()
                except Exception as e:
                    logging.warning(f"Could not generate {plot_type} plot: {str(e)}")
        
        except Exception as e:
            logging.error(f"Failed to save model artifacts: {str(e)}")
            raise

class ClusterTrainer:
    """Enhanced clustering with PyCaret"""
    def __init__(self):
        self.best_models = []
        
    def setup_environment(self, data):
        """Configure PyCaret for clustering"""
        try:
            exp = setup(
                data=data,
                normalize=True,
                transformation=True,  # Changed from transform to transformation
                ignore_features=None,
                session_id=42,
                # silent=True,
                html=True,  # Disable HTML output
                preprocess=True
            )
            return exp
            
        except Exception as e:
            logging.error(f"PyCaret setup failed: {str(e)}")
            raise
    
    def train_models(self):
        """Train multiple clustering models"""
        try:
            # Create list of models to try with their configurations
            clustering_configs = [
                ('kmeans', {
                    'n_clusters': [2, 3, 4, 5, 6],
                    'init': ['k-means++', 'random']
                }),
                ('ap', {
                    'damping': [0.7, 0.8, 0.9],
                    'preference': [-50, -30, -10]
                }),
                ('meanshift', {
                    'bandwidth': [None, 'auto'],
                    'bin_seeding': [True, False]
                }),
                ('sc', {
                    'n_clusters': [2, 3, 4, 5],
                    'eigen_solver': ['arpack', 'lobpcg']
                }),
                ('hclust', {
                    'n_clusters': [2, 3, 4, 5],
                    'linkage': ['ward', 'complete', 'average']
                }),
                ('dbscan', {
                    'eps': [0.1, 0.3, 0.5],
                    'min_samples': [5, 10, 15]
                }),
                ('optics', {
                    'min_samples': [5, 10, 15],
                    'xi': [0.05, 0.1, 0.15]
                }),
                ('birch', {
                    'n_clusters': [2, 3, 4, 5],
                    'threshold': [0.3, 0.5, 0.7]
                }),
                ('kmodes', {
                    'n_clusters': [2, 3, 4, 5],
                    'init': ['Huang', 'Cao']
                })
            ]
            
            tuned_models = []
            for model_name, param_grid in clustering_configs:
                try:
                    # Create base model
                    model = create_model(model_name)
                    logging.info(f"Created {model_name} model")
                    
                    # Tune model based on algorithm type
                    if model_name in ['kmeans', 'hclust', 'birch', 'kmodes']:
                        # Models that require n_clusters
                        tuned_model = tune_model(
                            model,
                            optimize='silhouette',
                            search_algorithm='grid',
                            custom_grid=param_grid,
                            n_iter=50
                        )
                    elif model_name in ['dbscan', 'optics']:
                        # Density-based models
                        tuned_model = tune_model(
                            model,
                            optimize='calinski_harabasz',
                            search_algorithm='grid',
                            custom_grid=param_grid
                        )
                    elif model_name == 'ap':
                        # Affinity Propagation specific
                        tuned_model = tune_model(
                            model,
                            optimize='silhouette',
                            search_algorithm='grid',
                            custom_grid=param_grid
                        )
                    else:
                        # Other algorithms
                        tuned_model = tune_model(
                            model,
                            optimize='silhouette',
                            search_algorithm='grid',
                            custom_grid=param_grid
                        )
                    
                    # Evaluate and store model
                    metrics = evaluate_model(tuned_model)
                    logging.info(f"Successfully tuned and evaluated {model_name}")
                    
                    # Save model specific metrics
                    metrics_df = pd.DataFrame([metrics])
                    metrics_df.to_csv(f'models/clustering/{model_name}_metrics.csv')
                    
                    tuned_models.append((tuned_model, metrics))
                    
                except Exception as e:
                    logging.warning(f"Could not process {model_name}: {str(e)}")
                    continue
            
            # Sort models by silhouette score if available
            tuned_models.sort(key=lambda x: x[1].get('Silhouette', 0), reverse=True)
            self.best_models = [model for model, _ in tuned_models]
            
            # Generate model insights
            for i, (model, _) in enumerate(tuned_models, 1):
                self.save_model_artifacts(model, i+1)
            
            # Create comparison visualization
            self.visualize_model_comparison(tuned_models)
            
            return self.best_models
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def visualize_model_comparison(self, tuned_models):
        """Create comparison visualization for all clustering models"""
        try:
            # Extract metrics for comparison
            comparison_data = {
                'Model': [],
                'Silhouette': [],
                'Calinski': [],
                'Davies': []
            }
            
            for model, metrics in tuned_models:
                model_name = model.__class__.__name__
                comparison_data['Model'].append(model_name)
                comparison_data['Silhouette'].append(metrics.get('Silhouette', 0))
                comparison_data['Calinski'].append(metrics.get('Calinski', 0))
                comparison_data['Davies'].append(metrics.get('Davies', 0))
            
            # Create comparison plot
            df_comparison = pd.DataFrame(comparison_data)
            plt.figure(figsize=(12, 6))
            
            # Plot metrics as grouped bars
            x = np.arange(len(df_comparison['Model']))
            width = 0.25
            
            plt.bar(x - width, df_comparison['Silhouette'], width, label='Silhouette')
        
        except Exception as e:
            logging.error(f"Failed to create comparison visualization: {str(e)}")
            raise

def compile_results_pdf():
    """Compile all results into a single PDF report"""
    try:
        logging.info("Compiling results into PDF...")
        doc = SimpleDocTemplate(
            "viz/clustering/pipeline_results.pdf",
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
        for img_path in glob.glob('viz/clustering/eda/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
        # Add Feature Selection results
        story.append(Paragraph("2. Feature Selection", styles['Heading2']))
        for img_path in glob.glob('viz/clustering/feature_selection/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
        # Add Model results
        story.append(Paragraph("3. Model Evaluation", styles['Heading2']))
        for img_path in glob.glob('viz/clustering/models/*.png'):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        logging.info("PDF report generated successfully at viz/clustering/pipeline_results.pdf")
        
    except Exception as e:
        logging.error(f"Failed to compile PDF report: {str(e)}")
        raise

@click.command()
@click.option('--train_path', default='data/data.csv', help='Path to data')
def main(train_path):
    """Main clustering pipeline execution"""
    try:
        configure_paths()
        
        # Data loading and preparation
        logging.info("Loading and analyzing data...")
        df = load_and_analyze_data(train_path)
        visualize_data(df)
        
        # Clustering
        logging.info("Training clustering models...")
        trainer = ClusterTrainer()
        trainer.setup_environment(df)
        best_models = trainer.train_models()
        
        # Get cluster assignments
        for i, model in enumerate(best_models, 1):
            try:
                assignments = assign_model(model)
                assignments.to_csv(f'data/processed/clustering/cluster_assignments_model_{i}.csv')
                logging.info(f"Saved cluster assignments for model {i}")
            except Exception as e:
                logging.warning(f"Could not get cluster assignments for model {i}: {str(e)}")
        
        # Compile results into PDF
        compile_results_pdf()
        
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()