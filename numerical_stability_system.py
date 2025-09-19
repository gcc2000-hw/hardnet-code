#!/usr/bin/env python3
"""
Numerical Stability System for Beta SPRING Training
===================================================

Provides numerical stability monitoring and fixes for Beta distribution training.
Quick implementation to resolve import error in beta_training_framework.py.

Author: Professor Davies
Date: 2025-08-16
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumericalStabilityMonitor:
    """Monitor and fix numerical stability issues during training"""
    
    def __init__(self):
        self.eps = 1e-8
        self.alpha_min = 0.1
        self.alpha_max = 100.0
        self.beta_min = 0.1
        self.beta_max = 100.0
        
    def check_tensor_health(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Check tensor for numerical issues"""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        return {
            'name': name,
            'has_nan': has_nan.item(),
            'has_inf': has_inf.item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item()
        }
    
    def clamp_parameters(self, alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clamp Beta parameters to stable ranges"""
        alpha_clamped = torch.clamp(alpha, self.alpha_min, self.alpha_max)
        beta_clamped = torch.clamp(beta, self.beta_min, self.beta_max)
        return alpha_clamped, beta_clamped
    
    def apply_stability_corrections(self, alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply stability corrections to Beta parameters"""
        return self.clamp_parameters(alpha, beta)

class GradientClipper:
    """Gradient clipping utilities"""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """Clip gradients and return the norm"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class NaNDetector:
    """Detect and handle NaN values"""
    
    def __init__(self):
        self.nan_count = 0
    
    def detect_and_fix(self, tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """Detect NaN values and replace with fill_value"""
        if torch.isnan(tensor).any():
            self.nan_count += 1
            logger.warning(f"NaN detected (count: {self.nan_count}), replacing with {fill_value}")
            return torch.where(torch.isnan(tensor), torch.tensor(fill_value), tensor)
        return tensor

class TrainingStabilityController:
    """Comprehensive stability controller for training"""
    
    def __init__(self):
        self.monitor = NumericalStabilityMonitor()
        self.clipper = GradientClipper()
        self.detector = NaNDetector()
        self.metrics = StabilityMetrics()
        
    def check_and_fix(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Check and fix all tensors for numerical issues"""
        fixed_tensors = {}
        for name, tensor in tensors.items():
            fixed_tensor = self.detector.detect_and_fix(tensor)
            fixed_tensors[name] = fixed_tensor
            self.metrics.update(self.monitor.check_tensor_health(fixed_tensor, name))
        return fixed_tensors
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current stability metrics"""
        return self.metrics.get_summary()

class StabilityMetrics:
    """Track stability metrics over time"""
    
    def __init__(self):
        self.metrics_history = []
        
    def update(self, metrics: Dict[str, Any]):
        """Update metrics with new measurements"""
        self.metrics_history.append(metrics)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1] if self.metrics_history else {}
        return {
            'latest': latest,
            'total_checks': len(self.metrics_history),
            'nan_count': sum(1 for m in self.metrics_history if m.get('has_nan', False)),
            'inf_count': sum(1 for m in self.metrics_history if m.get('has_inf', False))
        }

def create_stability_controller() -> TrainingStabilityController:
    """Factory function to create stability controller"""
    return TrainingStabilityController()

# Export the required classes
__all__ = [
    'NumericalStabilityMonitor', 
    'GradientClipper', 
    'NaNDetector',
    'TrainingStabilityController',
    'StabilityMetrics',
    'create_stability_controller'
]

if __name__ == "__main__":
    print("Numerical Stability System for Beta SPRING Training")
    print("Quick implementation to resolve import dependencies")
    
    # Test basic functionality
    monitor = NumericalStabilityMonitor()
    clipper = GradientClipper()
    detector = NaNDetector()
    
    print("✅ All numerical stability components initialized successfully")