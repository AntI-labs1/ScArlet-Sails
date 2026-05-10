"""
Out-of-Distribution (OOD) Detector.

Detects when input features are significantly different from training data.
Uses Mahalanobis distance to measure how "unusual" an input is.

Mathematical basis:
    D_M = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    
Where:
    x = input feature vector
    μ = mean of training features
    Σ = covariance matrix of training features
    
OOD penalty:
    R_ood = κ · max(0, D_M - threshold)
    
Usage:
    - D_M < threshold: Normal input, no penalty
    - D_M > threshold: OOD input, reduce confidence
"""
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timedelta
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class OODState:
    """Result of OOD detection."""
    mahalanobis_distance: float
    is_ood: bool
    ood_penalty: float
    percentile: float  # Where this sample falls in training distribution
    confidence_multiplier: float  # 1.0 = normal, < 1.0 = OOD
    safe_mode_active: bool = False  # Safe Mode state (with hysteresis)
    threshold_used: float = 0.0  # Actual threshold used for decision


class OODDetector:
    """
    Detects out-of-distribution inputs using Mahalanobis distance.
    
    Training statistics must be computed once and saved.
    At inference, computes distance to training distribution.
    """
    
    def __init__(
        self,
        threshold_sigma: float = 3.0,
        kappa: float = 0.1,
        min_confidence: float = 0.3,
        hysteresis_enter: float = 1.0,  # Enter Safe Mode at threshold * 1.0
        hysteresis_exit: float = 0.9,    # Exit Safe Mode at threshold * 0.9
        cooldown_hours: float = 4.0,     # Minimum time in Safe Mode (hours)
    ):
        """
        Args:
            threshold_sigma: Number of standard deviations for OOD threshold
            kappa: Penalty scaling factor
            min_confidence: Minimum confidence multiplier for extreme OOD
            hysteresis_enter: Multiplier for entering Safe Mode (1.0 = exact threshold)
            hysteresis_exit: Multiplier for exiting Safe Mode (0.9 = 10% buffer)
            cooldown_hours: Minimum hours to stay in Safe Mode before allowing exit
        """
        self.threshold_sigma = threshold_sigma
        self.kappa = kappa
        self.min_confidence = min_confidence
        self.hysteresis_enter = hysteresis_enter
        self.hysteresis_exit = hysteresis_exit
        self.cooldown_hours = cooldown_hours
        
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.n_features: int = 0
        self._fitted = False
        
        # For percentile estimation
        self._training_distances: Optional[np.ndarray] = None
        
        # FIX: Safe Mode Flickering - Stateful hysteresis and cooldown
        self.is_safe_mode: bool = False
        self.safe_mode_start_time: Optional[datetime] = None
    
    def fit(self, X: np.ndarray, store_distances: bool = True) -> 'OODDetector':
        """
        Compute training statistics.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            store_distances: Store training distances for percentile estimation
            
        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Compute mean
        self.mean = np.mean(X, axis=0)
        
        # Compute covariance with regularization
        cov = np.cov(X, rowvar=False)
        
        # Add small regularization for numerical stability
        reg = 1e-6 * np.eye(n_features)
        cov_reg = cov + reg
        
        # Compute inverse
        try:
            self.cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix inversion failed, using pseudo-inverse")
            self.cov_inv = np.linalg.pinv(cov_reg)
        
        # Compute threshold based on chi-squared distribution
        # For n_features degrees of freedom, 3-sigma ≈ 99.7 percentile
        from scipy import stats
        self.threshold = stats.chi2.ppf(0.997, df=n_features)
        
        # Store training distances for percentile estimation
        if store_distances:
            distances = []
            for i in range(min(n_samples, 10000)):  # Sample for efficiency
                d = self._compute_distance(X[i])
                distances.append(d)
            self._training_distances = np.array(distances)
        
        self._fitted = True
        
        # Reset Safe Mode state on fit
        self.is_safe_mode = False
        self.safe_mode_start_time = None
        
        return self
    
    def _compute_distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance for single sample."""
        diff = x - self.mean
        return np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff))
    
    def detect(
        self,
        x: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> OODState:
        """
        Detect if input is out-of-distribution with hysteresis and cooldown.
        
        FIX: Safe Mode Flickering - Implements stateful hysteresis and cooldown:
        - Enter Safe Mode: distance > threshold * hysteresis_enter (1.0)
        - Exit Safe Mode: distance < threshold * hysteresis_exit (0.9) AND cooldown expired
        
        Args:
            x: Input features, shape (n_features,) or (1, n_features)
            timestamp: Current timestamp for cooldown calculation (defaults to now)
            
        Returns:
            OODState with detection results including safe_mode_active
        """
        if not self._fitted:
            return OODState(
                mahalanobis_distance=0.0,
                is_ood=False,
                ood_penalty=0.0,
                percentile=50.0,
                confidence_multiplier=1.0,
                safe_mode_active=False,
                threshold_used=0.0,
            )
        
        # Flatten if needed
        x = np.asarray(x).flatten()
        
        if len(x) != self.n_features:
            warnings.warn(f"Feature count mismatch: {len(x)} vs {self.n_features}")
            return OODState(
                mahalanobis_distance=0.0,
                is_ood=False,
                ood_penalty=0.0,
                percentile=50.0,
                confidence_multiplier=1.0,
                safe_mode_active=False,
                threshold_used=0.0,
            )
        
        # Get current time for cooldown
        if timestamp is None:
            timestamp = datetime.now()
        
        # Compute distance
        distance = self._compute_distance(x)
        
        # HYSTERESIS LOGIC: Prevent flickering
        enter_threshold = self.threshold * self.hysteresis_enter  # 1.0 * threshold
        exit_threshold = self.threshold * self.hysteresis_exit    # 0.9 * threshold
        
        # Determine Safe Mode state
        if not self.is_safe_mode:
            # Not in Safe Mode: Check if we should enter
            if distance > enter_threshold:
                # Enter Safe Mode
                self.is_safe_mode = True
                self.safe_mode_start_time = timestamp
                logger.warning(
                    f"OOD: Entering Safe Mode (distance={distance:.2f} > "
                    f"enter_threshold={enter_threshold:.2f})"
                )
        else:
            # Already in Safe Mode: Check if we can exit
            cooldown_expired = False
            if self.safe_mode_start_time is not None:
                elapsed = timestamp - self.safe_mode_start_time
                cooldown_expired = elapsed >= timedelta(hours=self.cooldown_hours)
            
            # Exit conditions: distance below exit threshold AND cooldown expired
            if distance < exit_threshold and cooldown_expired:
                # Exit Safe Mode
                self.is_safe_mode = False
                elapsed_str = str(elapsed).split('.')[0] if cooldown_expired else "N/A"
                logger.info(
                    f"OOD: Exiting Safe Mode (distance={distance:.2f} < "
                    f"exit_threshold={exit_threshold:.2f}, cooldown={elapsed_str})"
                )
            elif distance < exit_threshold and not cooldown_expired:
                # Still in cooldown - maintain Safe Mode
                remaining = timedelta(hours=self.cooldown_hours) - elapsed
                logger.debug(
                    f"OOD: Safe Mode maintained (cooldown remaining: {remaining})"
                )
        
        # Check if OOD (for backward compatibility)
        is_ood = distance > self.threshold
        
        # Compute penalty
        ood_penalty = self.kappa * max(0, distance - self.threshold)
        
        # Compute percentile
        if self._training_distances is not None:
            percentile = (np.sum(self._training_distances <= distance) / 
                         len(self._training_distances) * 100)
        else:
            percentile = 50.0
        
        # Compute confidence multiplier
        if is_ood:
            # Reduce confidence for OOD samples
            # More OOD = lower confidence
            excess = distance / self.threshold  # > 1 for OOD
            confidence_multiplier = max(
                self.min_confidence,
                1.0 / excess
            )
        else:
            confidence_multiplier = 1.0
        
        # Determine threshold used for decision
        threshold_used = enter_threshold if not self.is_safe_mode else exit_threshold
        
        return OODState(
            mahalanobis_distance=distance,
            is_ood=is_ood,
            ood_penalty=ood_penalty,
            percentile=percentile,
            confidence_multiplier=confidence_multiplier,
            safe_mode_active=self.is_safe_mode,
            threshold_used=threshold_used,
        )
    
    def save(self, path: str):
        """Save detector state to file."""
        path = Path(path)
        
        data = {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'cov_inv': self.cov_inv.tolist() if self.cov_inv is not None else None,
            'threshold': self.threshold,
            'n_features': self.n_features,
            'threshold_sigma': self.threshold_sigma,
            'kappa': self.kappa,
            'min_confidence': self.min_confidence,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> 'OODDetector':
        """Load detector state from file."""
        path = Path(path)
        
        with open(path) as f:
            data = json.load(f)
        
        self.mean = np.array(data['mean']) if data['mean'] else None
        self.cov_inv = np.array(data['cov_inv']) if data['cov_inv'] else None
        self.threshold = data['threshold']
        self.n_features = data['n_features']
        self.threshold_sigma = data.get('threshold_sigma', 3.0)
        self.kappa = data.get('kappa', 0.1)
        self.min_confidence = data.get('min_confidence', 0.3)
        self._fitted = self.mean is not None
        
        return self


def compute_ood_stats_from_features(
    features_path: str,
    output_path: str,
    feature_names: Optional[list] = None,
) -> OODDetector:
    """
    Compute and save OOD statistics from training features.
    
    Args:
        features_path: Path to features parquet file
        output_path: Path to save detector state
        feature_names: Optional list of feature names to use
        
    Returns:
        Fitted OODDetector
    """
    import pandas as pd
    
    df = pd.read_parquet(features_path)
    
    # Use specified features or all numeric columns
    if feature_names:
        X = df[feature_names].values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude OHLCV columns
        exclude = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in numeric_cols if c not in exclude]
        X = df[feature_cols].values
    
    # Remove NaN rows
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X = X[valid_mask]
    
    print(f"Computing OOD statistics on {X.shape[0]} samples, {X.shape[1]} features")
    
    detector = OODDetector()
    detector.fit(X)
    detector.save(output_path)
    
    print(f"Saved to {output_path}")
    print(f"Threshold: {detector.threshold:.2f}")
    
    return detector