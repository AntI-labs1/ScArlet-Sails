"""
ML Training Pipeline - Week 2 Phase 3
=====================================

Complete end-to-end pipeline for ML model training:
1. Feature extraction from all modules
2. Data preprocessing (normalization, missing values, encoding)
3. Train/test split + cross-validation
4. Hyperparameter tuning (grid search)
5. Model training (XGBoost)
6. Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
7. Feature importance ranking
8. Backtest on historical data
9. Production-ready inference

Philosophy: "From raw data to production model in one script"

Author: Scarlet Sails Team
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MLPipelineConfig:
    """Configuration for ML training pipeline."""
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Training
    use_cross_validation: bool = True
    cv_folds: int = 5
    use_grid_search: bool = False  # Set True for hyperparameter tuning

    # Evaluation
    compute_feature_importance: bool = True
    save_predictions: bool = True
    save_model: bool = True

    # Paths
    output_dir: str = "output/ml_models"
    model_name: str = "crisis_classifier"


class MLTrainingPipeline:
    """
    End-to-end ML training pipeline.

    Usage:
        pipeline = MLTrainingPipeline(config)
        results = pipeline.run(X, y)
        predictions = pipeline.predict(X_new)
    """

    def __init__(self, config: MLPipelineConfig = None):
        """Initialize pipeline with configuration."""
        self.config = config or MLPipelineConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.training_results: Dict = {}

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def preprocess_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess features:
        - Handle missing values
        - Normalize to 0-1 range (already done by FeatureEngine, but can re-scale)
        - One-hot encoding (if categorical features exist)

        Args:
            X: Feature matrix
            fit: If True, fit scaler. If False, use existing scaler.

        Returns:
            Preprocessed feature matrix
        """
        X_processed = X.copy()

        # Handle missing values (fill with median)
        X_processed = X_processed.fillna(X_processed.median())

        # Optional: Additional normalization with StandardScaler
        # (Features already normalized by FeatureEngine, but this adds robustness)
        if fit:
            self.scaler = StandardScaler()
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        elif self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )

        return X_processed

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/val/test sets.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y  # Maintain class balance
        )

        # Second split: train vs val
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.config.random_state,
            stratify=y_trainval
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Trained model
        """
        self.feature_names = list(X_train.columns)

        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective='multi:softprob',
            num_class=len(np.unique(y_train)),
            eval_metric='mlogloss',
            random_state=self.config.random_state,
            use_label_encoder=False
        )

        # Train with optional validation set
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        return self.model

    def grid_search_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Best hyperparameters
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'min_child_weight': [1, 3, 5],
        }

        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(np.unique(y_train)),
                random_state=self.config.random_state,
                use_label_encoder=False
            ),
            param_grid,
            cv=self.config.cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_params_

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Cross-validation scores
        """
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.config.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )

        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

    def evaluate_model(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            X: Features
            y_true: True labels
            dataset_name: Name of dataset (train/val/test)

        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        # Basic metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

        # ROC-AUC (multi-class: one-vs-rest)
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_pred_proba,
                multi_class='ovr',
                average='macro'
            )
        except ValueError:
            # roc_auc_score raises ValueError when only one class is present
            metrics['roc_auc_ovr'] = None

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['per_class'] = report

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filename: Optional[str] = None):
        """
        Save trained model to disk.

        Args:
            filename: Optional custom filename
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        if filename is None:
            filename = f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = Path(self.config.output_dir) / filename
        self.model.save_model(str(filepath))
        print(f"Model saved to: {filepath}")

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """
        Save training results to JSON.

        Args:
            results: Results dictionary
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{self.config.model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = Path(self.config.output_dir) / filename

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_serializable = json.loads(json.dumps(results, default=convert_numpy))

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to: {filepath}")

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Run complete training pipeline.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("ML Training Pipeline")
        print("=" * 60)

        results = {}

        # Step 1: Preprocess features
        print("\n1. Preprocessing features...")
        X_processed = self.preprocess_features(X, fit=True)
        print(f"   Features: {X_processed.shape[1]}")
        print(f"   Samples: {len(X_processed)}")

        # Step 2: Split data
        print("\n2. Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_processed, y)
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")

        results['data_split'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
        }

        # Step 3: Hyperparameter tuning (optional)
        if self.config.use_grid_search:
            print("\n3. Grid search for hyperparameters...")
            best_params = self.grid_search_hyperparameters(X_train, y_train)
            print(f"   Best params: {best_params}")
            results['best_params'] = best_params

            # Update config with best params
            for key, value in best_params.items():
                setattr(self.config, key, value)

        # Step 4: Train model
        print(f"\n{3 if not self.config.use_grid_search else 4}. Training model...")
        self.train_model(X_train, y_train, X_val, y_val)
        print(f"   Model: XGBoost Classifier")
        print(f"   n_estimators: {self.config.n_estimators}")
        print(f"   max_depth: {self.config.max_depth}")
        print(f"   learning_rate: {self.config.learning_rate}")

        # Step 5: Cross-validation
        if self.config.use_cross_validation:
            print(f"\n{4 if not self.config.use_grid_search else 5}. Cross-validation...")
            cv_results = self.cross_validate(X_train, y_train)
            print(f"   CV Mean Accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
            results['cross_validation'] = cv_results

        # Step 6: Evaluate on train/val/test
        step_num = 5 if not self.config.use_grid_search else 6
        if not self.config.use_cross_validation:
            step_num -= 1

        print(f"\n{step_num}. Evaluating model...")

        train_metrics = self.evaluate_model(X_train, y_train, "train")
        val_metrics = self.evaluate_model(X_val, y_val, "validation")
        test_metrics = self.evaluate_model(X_test, y_test, "test")

        print(f"\n   Train Accuracy:      {train_metrics['accuracy']:.4f}")
        print(f"   Val Accuracy:        {val_metrics['accuracy']:.4f}")
        print(f"   Test Accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"\n   Test Precision:      {test_metrics['precision_macro']:.4f}")
        print(f"   Test Recall:         {test_metrics['recall_macro']:.4f}")
        print(f"   Test F1:             {test_metrics['f1_macro']:.4f}")
        print(f"   Test ROC-AUC (OvR):  {test_metrics.get('roc_auc_ovr', 'N/A')}")

        results['metrics'] = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
        }

        # Step 7: Feature importance
        if self.config.compute_feature_importance:
            step_num += 1
            print(f"\n{step_num}. Feature importance (Top 10)...")
            importance = self.get_feature_importance(top_n=10)
            for i, row in importance.iterrows():
                print(f"   {row['feature']:30s}: {row['importance']:.4f}")

            results['feature_importance'] = importance.to_dict('records')

        # Step 8: Save model and results
        if self.config.save_model:
            step_num += 1
            print(f"\n{step_num}. Saving model...")
            self.save_model()

        if self.config.save_predictions:
            self.save_results(results)

        print("\n" + "=" * 60)
        print("✅ Training pipeline complete!")
        print("=" * 60)

        self.training_results = results
        return results

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # Preprocess features
        X_processed = self.preprocess_features(X, fit=False)

        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        return predictions, probabilities


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    # Import crisis classifier data generation
    from crisis_classifier import generate_crisis_training_data

    print("=" * 60)
    print("ML Training Pipeline Demo")
    print("=" * 60)

    # Generate training data
    print("\nGenerating training data...")
    X, y = generate_crisis_training_data(n_samples=1000)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")

    # Configure pipeline
    config = MLPipelineConfig(
        test_size=0.2,
        val_size=0.1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_cross_validation=True,
        cv_folds=5,
        use_grid_search=False,  # Set True to enable hyperparameter tuning (slower)
        compute_feature_importance=True,
        save_model=True,
        save_predictions=True,
        model_name="crisis_classifier_demo"
    )

    # Run pipeline
    pipeline = MLTrainingPipeline(config)
    results = pipeline.run(X, y)

    # Test inference on new data
    print("\n" + "=" * 60)
    print("Testing Inference on New Data")
    print("=" * 60)

    # Generate new test samples
    X_new, y_new = generate_crisis_training_data(n_samples=10)
    predictions, probabilities = pipeline.predict(X_new)

    print(f"\nPredictions on {len(X_new)} new samples:")
    for i, (pred, true, prob) in enumerate(zip(predictions, y_new, probabilities)):
        crisis_names = ['CRASH', 'MANIPULATION', 'GLITCH', 'OPPORTUNITY']
        print(f"  Sample {i+1}: Predicted={crisis_names[pred]} (confidence={prob[pred]:.2%}), "
              f"True={crisis_names[true]}")

    print("\n✅ ML Training Pipeline demo complete!")
