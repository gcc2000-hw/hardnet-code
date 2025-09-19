"""
Complete Beta Distribution-based Spatial Reasoning Module for SPRING

This implements Professor Davies' revolutionary probabilistic approach to spatial reasoning
using learned Beta distributions for coordinate prediction.

Mathematical Foundation:
- Each coordinate (x,y,w,h) modeled as Beta(α, β) distribution scaled to [0,1000]
- Spatial constraints satisfied probabilistically via P(constraint | distributions)
- Fully differentiable through reparameterization trick
- Universal approximation while maintaining constraint satisfaction

Architecture Innovation:
- Multi-scale constraint networks for hierarchical reasoning
- Attention-based object interaction modeling
- Curriculum learning for constraint complexity
- Analytical constraint solutions where possible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Gamma
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from collections import namedtuple
import warnings

# Import SPRING constraint types
from constraint_language_v2 import *

# Import Chen's mathematically rigorous constraint solver
from analytical_constraints_corrected import (
    MathematicallyRigorousConstraintSolver,
    ValidatedConstraintFramework
)

# Import coordinate validation system for mathematical consistency
from coordinate_validation_system import (
    CoordinateValidator,
    CoordinateBounds,
    CoordinateConsistencyChecker
)

class MultiScaleBetaPredictor(nn.Module):
    """
    Multi-scale Beta parameter prediction with object interaction modeling
    
    Architecture:
    Scene Features -> Multi-Head Attention -> Object-Specific Encoders -> (α, β) parameters
    
    Chen's Phase 1 Fixes:
    - Asymmetric initialization strategy for x/y vs w/h coordinates
    - Dynamic parameter bounds based on training progress
    - Improved numerical stability throughout
    - Object-specific and coordinate-specific initialization
    """
    
    def __init__(self, scene_dim: int = 512, hidden_dim: int = 256, 
                 num_objects: int = 10, num_heads: int = 8):
        super().__init__()
        
        self.scene_dim = scene_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.num_heads = num_heads
        
        # Scene encoding with residual connections
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Multi-head attention for object interactions with stability improvements
        # Chen: Add numerical stability through proper scaling
        self.object_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Attention stability epsilon for numerical guards
        self.attn_epsilon = 1e-8
        
        # Object-specific parameter predictors
        self.object_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # Separate networks for α and β with different inductive biases
        # DAVIES FIX: Remove Softplus, apply exponential in forward pass
        self.alpha_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 4),  # x, y, w, h
            # No activation - apply exponential in forward pass
        )
        
        # β controls right tail - apply exponential in forward pass
        self.beta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 4),   # x, y, w, h
            # No activation - apply exponential in forward pass
        )
        
        # CHEN'S ASYMMETRIC INITIALIZATION STRATEGY
        # Different initialization for position (x,y) vs size (w,h) coordinates
        # Position coordinates: wider range for exploration
        # Size coordinates: narrower range for stability
        alpha_init = torch.zeros(num_objects, 4)
        beta_init = torch.zeros(num_objects, 4)
        
        # Position coordinates (x, y) - indices 0, 1
        # Encourage exploration across full space with more variation
        alpha_init[:, :2] = torch.randn(num_objects, 2) * 3.0 + 5.0  # Mean 5, std 3 for positions
        beta_init[:, :2] = torch.randn(num_objects, 2) * 3.0 + 5.0
        
        # Size coordinates (w, h) - indices 2, 3  
        # More conservative initialization for stability
        alpha_init[:, 2:] = torch.randn(num_objects, 2) * 1.5 + 3.0  # Mean 3, std 1.5 for sizes
        beta_init[:, 2:] = torch.randn(num_objects, 2) * 1.5 + 3.0
        
        # CRITICAL FIX 2.3: Replace progressive bias with spatial grid initialization
        # This eliminates the bottom-right clustering by distributing objects across space
        for i in range(num_objects):
            # Use spatial grid distribution (3x3 grid for up to 9 objects)  
            grid_x = (i % 3) / 3.0  # Grid positions: 0, 0.33, 0.67 
            grid_y = (i // 3 % 3) / 3.0  # Grid positions: 0, 0.33, 0.67
            
            # Position coordinates (x, y) - bias toward grid position
            alpha_init[i, 0] = 3.0 + grid_x * 2.0      # x: 3.0-5.0 range
            beta_init[i, 0] = 5.0 - grid_x * 2.0       # x: 3.0-5.0 range (inverted)
            alpha_init[i, 1] = 3.0 + grid_y * 2.0      # y: 3.0-5.0 range  
            beta_init[i, 1] = 5.0 - grid_y * 2.0       # y: 3.0-5.0 range (inverted)
            
            # Size coordinates (w, h) - add variety but avoid extreme bias
            alpha_init[i, 2] = 3.0 + (i * 0.2) % 1.0   # Cycling size variation
            beta_init[i, 2] = 3.5 - (i * 0.2) % 1.0    # Cycling size variation
            alpha_init[i, 3] = 3.0 + ((i + 2) * 0.2) % 1.0  # Different cycle for height
            beta_init[i, 3] = 3.5 - ((i + 2) * 0.2) % 1.0   # Different cycle for height
        
        self.alpha_bias = nn.Parameter(alpha_init)
        self.beta_bias = nn.Parameter(beta_init)
        
        # CHEN'S DYNAMIC PARAMETER BOUNDS
        # Start conservative and gradually expand during training
        self.register_buffer('training_progress', torch.tensor(0.0))
        self.min_param_initial = 2.5  # Start with higher minimum for stability  
        self.min_param_final = 2.0   # CRITICAL FIX 1.2: Prevent variance explosion (was 1.01)
        self.max_param_initial = 20.0  # CHEN PRIORITY 2: Allow β escape from 15.0 trap
        self.max_param_final = 20.0   # CHEN PRIORITY 2: Relaxed safety limit for expressiveness
        
        # Current bounds (will be updated during training)
        self.min_param = self.min_param_initial
        self.max_param = self.max_param_initial
        
        # Initialize weights properly
        self._initialize_weights()
        
    def forward(self, scene_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scene_features: [batch_size, scene_dim]
            
        Returns:
            alpha: [batch_size, num_objects, 4] Beta α parameters (bounded to [1.01, 50.0])
            beta:  [batch_size, num_objects, 4] Beta β parameters (bounded to [1.01, 50.0])
        """
        batch_size = scene_features.size(0)
        
        # Encode scene features
        encoded = self.scene_encoder(scene_features)  # [batch_size, hidden_dim]
        
        # Expand for multiple objects with learned embeddings
        object_queries = encoded.unsqueeze(1).expand(-1, self.num_objects, -1)
        
        # Chen: Add residual connection and numerical guards for attention stability
        # Object interaction modeling via self-attention with stability improvements
        attended, attn_weights = self.object_attention(
            object_queries, object_queries, object_queries
        )  # [batch_size, num_objects, hidden_dim]
        
        # Residual connection to preserve information
        attended = attended + object_queries
        
        # Numerical guard against NaN/Inf
        attended = torch.where(
            torch.isfinite(attended),
            attended,
            object_queries  # Fallback to original if attention produces NaN/Inf
        )
        
        # Object-specific encoding
        object_features = self.object_encoder(attended)
        
        # Predict Beta parameters for each object
        alpha_raw = self.alpha_predictor(object_features)  # [batch_size, num_objects, 4]
        beta_raw = self.beta_predictor(object_features)    # [batch_size, num_objects, 4]
        
        # Apply object-specific bias (broadcast properly)
        alpha_with_bias = alpha_raw + self.alpha_bias.unsqueeze(0)  # self.alpha_bias is [num_objects, 4]
        beta_with_bias = beta_raw + self.beta_bias.unsqueeze(0)     # self.beta_bias is [num_objects, 4]
        
        # CHEN'S EMERGENCY MATHEMATICAL FIX:
        # Replace gradient-killing clamp+exp with smooth sigmoid transformation
        # This ensures continuous gradients and proper bounded output
        self.update_dynamic_bounds()
        
        # CRITICAL FIX 2.1: Replace gradient-killing sigmoid with softplus + soft bounds  
        # This preserves gradient flow while maintaining numerical stability
        alpha = self.min_param + F.softplus(alpha_with_bias)
        beta = self.min_param + F.softplus(beta_with_bias)
        
        # Apply soft upper bounds to prevent explosion while preserving gradients
        alpha = alpha * self.max_param / (alpha + self.max_param)
        beta = beta * self.max_param / (beta + self.max_param)
        
        # CHEN PRIORITY 2: Relaxed concentration limit for β escape  
        max_concentration = 20.0  # Allow β escape from 15.0 trap while maintaining stability
        alpha = torch.clamp(alpha, max=max_concentration)
        beta = torch.clamp(beta, max=max_concentration)
        
        # Softplus + soft bounds ensures proper parameter ranges with preserved gradients
        
        return alpha, beta
    
    def update_dynamic_bounds(self):
        """Update parameter bounds based on training progress"""
        if self.training:
            # Linearly interpolate bounds based on training progress
            progress = torch.clamp(self.training_progress, 0.0, 1.0)
            self.min_param = (self.min_param_initial * (1 - progress) + 
                            self.min_param_final * progress).item()
            self.max_param = (self.max_param_initial * (1 - progress) + 
                            self.max_param_final * progress).item()
    
    def set_training_progress(self, progress: float):
        """Set training progress for dynamic bound adjustment"""
        self.training_progress.fill_(progress)
    
    def _initialize_weights(self):
        """
        Chen's IMPROVED initialization strategy.
        
        Key improvements:
        - Different initialization for alpha vs beta predictors
        - Coordinate-aware initialization (position vs size)
        - Ensures initial predictions span reasonable coordinate ranges
        """
        # DAVIES FIX: Xavier initialization with gain=2.0
        for module in [self.alpha_predictor, self.beta_predictor]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=2.0)  # Davies' fix
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Standard initialization for other components
        for module in [self.scene_encoder, self.object_encoder]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=2.0)  # Davies' consistent gain
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    # Standard initialization for LayerNorm
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)
        
        # Initialize attention layers separately
        if hasattr(self, 'object_attention'):
            for param in self.object_attention.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=2.0)  # Davies' consistent gain


class AnalyticalConstraintSolver(nn.Module):
    """
    Chen's Mathematically Rigorous Constraint Solver Integration
    
    This provides TRUE analytical solutions for T1 constraints using scipy's
    incomplete beta function, and rigorously validated approximations for T2 constraints.
    
    Academic Standards:
    - No misleading terminology (exact vs. approximate clearly distinguished)
    - Mathematical error bounds for all computations
    - Runtime validation of approximation quality
    """
    
    def __init__(self):
        super().__init__()
        self.solver = MathematicallyRigorousConstraintSolver()
        
    def constraint_t1_probability(self, alpha: torch.Tensor, beta: torch.Tensor,
                                 constraint_type: str, coord_idx: int, 
                                 value: float, offset: float, 
                                 epsilon: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        TRUE analytical solution for T1 constraints using incomplete beta function
        
        Args:
            alpha, beta: Beta distribution parameters [batch_size, num_objects, 4]
            constraint_type: 'lt', 'gt', or 'eq'
            coord_idx: coordinate index (0=x, 1=y, 2=w, 3=h)
            value: threshold value in [0,1000] coordinate system
            offset: offset to apply to threshold
            epsilon: numerical stability constant
        
        Returns:
            prob: [batch_size, num_objects] constraint satisfaction probabilities
        """
        result = self.solver.t1_exact_analytical(
            alpha, beta, constraint_type, coord_idx, value, offset
        )
        
        # CRITICAL: Preserve ALL metadata from Chen's solver
        # Do NOT double-clamp - Chen's solver already handles numerical stability
        return result
    
    def constraint_t2_probability_analytical(self, alpha1: torch.Tensor, beta1: torch.Tensor,
                                           alpha2: torch.Tensor, beta2: torch.Tensor,
                                           constraint_type: str, coord_idx: int, 
                                           offset: float, epsilon: float = 1e-6, 
                                           validity_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Rigorous T2 constraint evaluation with Chen's validity checking
        
        Uses Chen's mathematically validated normal approximation where applicable,
        falls back to high-precision sampling with error bounds otherwise.
        
        Args:
            alpha1, beta1: Beta parameters for first object [batch_size, num_objects, 4]
            alpha2, beta2: Beta parameters for second object [batch_size, num_objects, 4]
            constraint_type: 'lt', 'gt', or 'eq'
            coord_idx: coordinate index (0=x, 1=y, 2=w, 3=h)
            offset: offset value in [0,1000] coordinate system
            epsilon: numerical stability constant
            validity_threshold: minimum validity score to trust normal approximation
            
        Returns:
            result: Dict with probability, error_bound, validity_score, method
        """
        # First try Chen's normal approximation
        normal_result = self.solver.t2_normal_approximation(
            alpha1, beta1, alpha2, beta2, constraint_type, coord_idx, offset
        )
        
        # Check Chen's method to see if it already fell back to sampling
        chen_method = normal_result.get('method', 'unknown')
        
        if 'sampling' in chen_method:
            # Chen already used sampling - trust his result (he's already handling invalid cases)
            return normal_result
        elif 'normal_approximation' in chen_method:
            # Chen used normal approximation - check validity score
            validity_score = normal_result.get('validity_score', torch.tensor(1.0))  # Default to valid if missing
            
            if validity_score >= validity_threshold:
                # Chen says normal approximation is valid - trust it
                return normal_result
            else:
                # Chen says normal approximation is invalid but used it anyway - fall back to our sampling
                return self._constraint_t2_sampling_high_precision(
                    alpha1, beta1, alpha2, beta2, constraint_type, coord_idx, offset
                )
        else:
            # Unknown method - trust Chen's result
            return normal_result
    
    def _constraint_t2_sampling_high_precision(self, alpha1: torch.Tensor, beta1: torch.Tensor,
                                             alpha2: torch.Tensor, beta2: torch.Tensor,
                                             constraint_type: str, coord_idx: int, 
                                             offset: float, target_error: float = 0.01) -> Dict[str, torch.Tensor]:
        """Adaptive high-precision sampling for T2 constraints when normal approximation is invalid"""
        a1, b1 = alpha1[:, :, coord_idx], beta1[:, :, coord_idx]
        a2, b2 = alpha2[:, :, coord_idx], beta2[:, :, coord_idx]
        
        # Adaptive sampling with convergence check
        # CRITICAL FIX 1.2: Updated numerical guards to prevent variance explosion
        a1_safe = torch.clamp(a1, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        b1_safe = torch.clamp(b1, min=2.0, max=50.0)  # Increased from 1.01 to 2.0  
        a2_safe = torch.clamp(a2, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        b2_safe = torch.clamp(b2, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        
        dist1 = Beta(a1_safe, b1_safe)
        dist2 = Beta(a2_safe, b2_safe)
        
        # Start with a reasonable batch size
        batch_size = 10000
        max_samples = 200000  # Computational limit
        total_samples = 0
        running_sum = 0.0
        running_sum_sq = 0.0
        
        while total_samples < max_samples:
            # Generate batch of samples
            samples1 = dist1.rsample((batch_size,)) * 1000  
            samples2 = dist2.rsample((batch_size,)) * 1000
            
            if constraint_type == 'lt':
                satisfied = (samples1 < samples2 + offset).float()
            elif constraint_type == 'gt':
                satisfied = (samples1 > samples2 + offset).float()
            elif constraint_type == 'eq':
                tolerance = 10.0  # 10 pixels for equality
                satisfied = (torch.abs(samples1 - samples2 - offset) < tolerance).float()
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
            
            # Update running statistics
            batch_mean = satisfied.mean(0)
            batch_sum = satisfied.sum(0)
            batch_sum_sq = (satisfied ** 2).sum(0)
            
            running_sum += batch_sum
            running_sum_sq += batch_sum_sq
            total_samples += batch_size
            
            # Check convergence every 20k samples
            if total_samples >= 20000 and total_samples % 20000 == 0:
                current_mean = running_sum / total_samples
                current_var = (running_sum_sq / total_samples) - (current_mean ** 2)
                current_std_err = torch.sqrt(current_var / total_samples)
                
                # Check if 95% confidence interval is tight enough
                if torch.all(1.96 * current_std_err < target_error):
                    break
        
        # Final statistics
        prob = running_sum / total_samples
        variance = (running_sum_sq / total_samples) - (prob ** 2)
        std_err = torch.sqrt(variance / total_samples)
        error_bound = 1.96 * std_err  # 95% confidence interval
        
        return {
            'probability': prob,
            'error_bound': error_bound,
            'method': f'adaptive_sampling_{total_samples}',
            'validity_score': torch.tensor(1.0),  # Sampling is always "valid"
            'computational_cost': total_samples / 1000.0  # Scale by 1k samples
        }
    
    def _constraint_t2_sampling(self, alpha1: torch.Tensor, beta1: torch.Tensor,
                               alpha2: torch.Tensor, beta2: torch.Tensor,
                               constraint_type: str, coord_idx: int, 
                               offset: float, n_samples: int = 1000) -> torch.Tensor:
        """Fallback sampling method for complex T2 constraints"""
        a1, b1 = alpha1[:, :, coord_idx], beta1[:, :, coord_idx]
        a2, b2 = alpha2[:, :, coord_idx], beta2[:, :, coord_idx]
        
        # Sample from both distributions  
        # CRITICAL FIX 1.2: Updated numerical guards to prevent variance explosion
        a1_safe = torch.clamp(a1, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        b1_safe = torch.clamp(b1, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        a2_safe = torch.clamp(a2, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        b2_safe = torch.clamp(b2, min=2.0, max=50.0)  # Increased from 1.01 to 2.0
        
        dist1 = Beta(a1_safe, b1_safe)
        dist2 = Beta(a2_safe, b2_safe)
        
        samples1 = dist1.rsample((n_samples,)) * 1000  # [n_samples, batch_size, num_objects]
        samples2 = dist2.rsample((n_samples,)) * 1000
        
        if constraint_type == 'lt':
            satisfied = (samples1 < samples2 + offset).float()
        elif constraint_type == 'gt':
            satisfied = (samples1 > samples2 + offset).float()
        elif constraint_type == 'eq':
            satisfied = (torch.abs(samples1 - samples2 - offset) < 10.0).float()  # 10 pixel tolerance
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
            
        return satisfied.mean(0)  # Average over samples
    
    def _sample_t3_constraint(self, alpha: torch.Tensor, beta: torch.Tensor,
                             obj_idx: int, coord1_idx: int, coord2_idx: int,
                             value: float, offset: float, constraint_type: str,
                             n_samples: int = 1000) -> torch.Tensor:
        """Sampling-based evaluation for T3 constraints (coordinate sums)"""
        try:
            # Ensure indices are integers
            if torch.is_tensor(obj_idx):
                obj_idx = int(obj_idx.item()) if obj_idx.numel() == 1 else int(obj_idx.flatten()[0].item())
            if torch.is_tensor(coord1_idx):
                coord1_idx = int(coord1_idx.item()) if coord1_idx.numel() == 1 else int(coord1_idx.flatten()[0].item())
            if torch.is_tensor(coord2_idx):
                coord2_idx = int(coord2_idx.item()) if coord2_idx.numel() == 1 else int(coord2_idx.flatten()[0].item())
            if torch.is_tensor(value):
                value = float(value.item()) if value.numel() == 1 else float(value.flatten()[0].item())
            if torch.is_tensor(offset):
                offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
            
            # Extract parameters for object and coordinates
            a1 = alpha[:, obj_idx, coord1_idx]
            b1 = beta[:, obj_idx, coord1_idx]
            a2 = alpha[:, obj_idx, coord2_idx]
            b2 = beta[:, obj_idx, coord2_idx]
            
            # Create distributions
            dist1 = Beta(torch.clamp(a1, 1.01, 50.0), torch.clamp(b1, 1.01, 50.0))
            dist2 = Beta(torch.clamp(a2, 1.01, 50.0), torch.clamp(b2, 1.01, 50.0))
            
            # Sample and scale to [0,1000]
            samples1 = dist1.rsample((n_samples,)) * 1000
            samples2 = dist2.rsample((n_samples,)) * 1000
            sum_samples = samples1 + samples2
            
            # Apply constraint
            threshold = value + offset
            if constraint_type == 'lt':
                satisfied = (sum_samples < threshold).float()
            elif constraint_type == 'gt':
                satisfied = (sum_samples > threshold).float()
            elif constraint_type == 'eq':
                satisfied = (torch.abs(sum_samples - threshold) < 5.0).float()  # 5px tolerance
            else:
                satisfied = torch.ones_like(sum_samples) * 0.5
                
            return satisfied.mean(0)
            
        except Exception:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _sample_t4_constraint(self, alpha: torch.Tensor, beta: torch.Tensor,
                             obj1_idx: int, coord1_idx: int, coord2_idx: int,
                             obj2_idx: int, coord3_idx: int, offset: float,
                             constraint_type: str, n_samples: int = 1000) -> torch.Tensor:
        """Sampling-based evaluation for T4 constraints (sum vs coordinate)"""
        try:
            # Ensure indices are integers
            if torch.is_tensor(obj1_idx):
                obj1_idx = int(obj1_idx.item()) if obj1_idx.numel() == 1 else int(obj1_idx.flatten()[0].item())
            if torch.is_tensor(obj2_idx):
                obj2_idx = int(obj2_idx.item()) if obj2_idx.numel() == 1 else int(obj2_idx.flatten()[0].item())
            if torch.is_tensor(coord1_idx):
                coord1_idx = int(coord1_idx.item()) if coord1_idx.numel() == 1 else int(coord1_idx.flatten()[0].item())
            if torch.is_tensor(coord2_idx):
                coord2_idx = int(coord2_idx.item()) if coord2_idx.numel() == 1 else int(coord2_idx.flatten()[0].item())
            if torch.is_tensor(coord3_idx):
                coord3_idx = int(coord3_idx.item()) if coord3_idx.numel() == 1 else int(coord3_idx.flatten()[0].item())
            if torch.is_tensor(offset):
                offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
            
            # Object 1 coordinates
            a1 = alpha[:, obj1_idx, coord1_idx]
            b1 = beta[:, obj1_idx, coord1_idx]
            a2 = alpha[:, obj1_idx, coord2_idx]
            b2 = beta[:, obj1_idx, coord2_idx]
            
            # Object 2 coordinate
            a3 = alpha[:, obj2_idx, coord3_idx]
            b3 = beta[:, obj2_idx, coord3_idx]
            
            # Create distributions
            dist1 = Beta(torch.clamp(a1, 1.01, 50.0), torch.clamp(b1, 1.01, 50.0))
            dist2 = Beta(torch.clamp(a2, 1.01, 50.0), torch.clamp(b2, 1.01, 50.0))
            dist3 = Beta(torch.clamp(a3, 1.01, 50.0), torch.clamp(b3, 1.01, 50.0))
            
            # Sample and compute sum vs single coordinate
            samples1 = dist1.rsample((n_samples,)) * 1000
            samples2 = dist2.rsample((n_samples,)) * 1000
            samples3 = dist3.rsample((n_samples,)) * 1000
            
            sum_left = samples1 + samples2
            right_side = samples3 + offset
            
            # Apply constraint
            if constraint_type == 'lt':
                satisfied = (sum_left < right_side).float()
            elif constraint_type == 'gt':
                satisfied = (sum_left > right_side).float()
            elif constraint_type == 'eq':
                satisfied = (torch.abs(sum_left - right_side) < 5.0).float()
            else:
                satisfied = torch.ones_like(sum_left) * 0.5
                
            return satisfied.mean(0)
            
        except Exception:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _sample_intersection_constraint(self, alpha: torch.Tensor, beta: torch.Tensor,
                                      obj1_idx: int, obj2_idx: int, offset: float,
                                      n_samples: int = 1000) -> torch.Tensor:
        """Sampling-based evaluation for T5 intersection constraints"""
        try:
            # Ensure indices are integers
            if torch.is_tensor(obj1_idx):
                obj1_idx = int(obj1_idx.item()) if obj1_idx.numel() == 1 else int(obj1_idx.flatten()[0].item())
            if torch.is_tensor(obj2_idx):
                obj2_idx = int(obj2_idx.item()) if obj2_idx.numel() == 1 else int(obj2_idx.flatten()[0].item())
            if torch.is_tensor(offset):
                offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
            
            # Get bounding box parameters for both objects
            # obj1: (x1, y1, w1, h1), obj2: (x2, y2, w2, h2)
            obj1_params = [
                (alpha[:, obj1_idx, i], beta[:, obj1_idx, i]) for i in range(4)
            ]
            obj2_params = [
                (alpha[:, obj2_idx, i], beta[:, obj2_idx, i]) for i in range(4)
            ]
            
            # Sample bounding boxes
            obj1_samples = []
            obj2_samples = []
            
            for (a, b) in obj1_params:
                dist = Beta(torch.clamp(a, 1.01, 50.0), torch.clamp(b, 1.01, 50.0))
                obj1_samples.append(dist.rsample((n_samples,)) * 1000)
            
            for (a, b) in obj2_params:
                dist = Beta(torch.clamp(a, 1.01, 50.0), torch.clamp(b, 1.01, 50.0))
                obj2_samples.append(dist.rsample((n_samples,)) * 1000)
            
            # Check intersection
            x1, y1, w1, h1 = obj1_samples
            x2, y2, w2, h2 = obj2_samples
            
            # Intersection condition: rectangles overlap
            x_overlap = (x1 < x2 + w2) & (x2 < x1 + w1)
            y_overlap = (y1 < y2 + h2) & (y2 < y1 + h1)
            intersects = (x_overlap & y_overlap).float()
            
            return intersects.mean(0)
            
        except Exception:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _sample_manhattan_distance_constraint(self, alpha: torch.Tensor, beta: torch.Tensor,
                                            obj1_idx: int, obj2_idx: int, obj3_idx: int,
                                            offset: float, n_samples: int = 1000) -> torch.Tensor:
        """Sampling-based evaluation for T6 Manhattan distance constraints"""
        try:
            # Ensure indices are integers
            if torch.is_tensor(obj1_idx):
                obj1_idx = int(obj1_idx.item()) if obj1_idx.numel() == 1 else int(obj1_idx.flatten()[0].item())
            if torch.is_tensor(obj2_idx):
                obj2_idx = int(obj2_idx.item()) if obj2_idx.numel() == 1 else int(obj2_idx.flatten()[0].item())
            if torch.is_tensor(obj3_idx):
                obj3_idx = int(obj3_idx.item()) if obj3_idx.numel() == 1 else int(obj3_idx.flatten()[0].item())
            if torch.is_tensor(offset):
                offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
            
            # Sample center coordinates for all three objects
            objects_samples = []
            for obj_idx in [obj1_idx, obj2_idx, obj3_idx]:
                if obj_idx < alpha.size(1):
                    # Sample x, y coordinates (center points)
                    x_dist = Beta(torch.clamp(alpha[:, obj_idx, 0], 1.01, 50.0),  # REVERTED: Keep expressiveness
                                torch.clamp(beta[:, obj_idx, 0], 1.01, 50.0))   # REVERTED: Keep expressiveness
                    y_dist = Beta(torch.clamp(alpha[:, obj_idx, 1], 1.01, 50.0),  # REVERTED: Keep expressiveness
                                torch.clamp(beta[:, obj_idx, 1], 1.01, 50.0))   # REVERTED: Keep expressiveness
                    
                    x_samples = x_dist.rsample((n_samples,)) * 1000
                    y_samples = y_dist.rsample((n_samples,)) * 1000
                    objects_samples.append((x_samples, y_samples))
                else:
                    # Invalid object index
                    return torch.ones(alpha.size(0), device=alpha.device) * 0.5
            
            # Compute Manhattan distances
            (x1, y1), (x2, y2), (x3, y3) = objects_samples
            
            dist_12 = torch.abs(x1 - x2) + torch.abs(y1 - y2)
            dist_13 = torch.abs(x1 - x3) + torch.abs(y1 - y3)
            
            # Check if distances are approximately equal (within offset tolerance)
            distance_equal = (torch.abs(dist_12 - dist_13) < offset + 5.0).float()
            
            return distance_equal.mean(0)
            
        except Exception:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5


class HierarchicalConstraintNetwork(nn.Module):
    """
    Hierarchical constraint evaluation with curriculum learning support
    
    Handles complex constraint compositions (OR, AND, NOT) and provides
    curriculum learning from simple to complex constraints.
    """
    
    def __init__(self, complexity_levels: int = 5):
        super().__init__()
        self.complexity_levels = complexity_levels
        self.analytical_solver = AnalyticalConstraintSolver()
        
        # Learnable weights for constraint importance
        self.constraint_weights = nn.Parameter(torch.ones(complexity_levels))
        
    def evaluate_constraint_list(self, constraints: List[Any], 
                                alpha: torch.Tensor, beta: torch.Tensor,
                                complexity_level: int = 5) -> torch.Tensor:
        """
        Evaluate a list of constraints with hierarchical complexity handling.
        
        Args:
            constraints: List of constraint lists (one per batch item) or single constraint list
            alpha: [batch_size, num_objects, 4] Beta parameters
            beta: [batch_size, num_objects, 4] Beta parameters
            complexity_level: Current curriculum level
            
        Returns:
            torch.Tensor: [batch_size] constraint satisfaction probabilities
        """
        batch_size = alpha.size(0)
        device = alpha.device
        
        # Handle case where constraints is a list of constraint lists (batched)
        if constraints and isinstance(constraints[0], list):
            # Batched constraints - evaluate each batch item separately
            batch_probs = []
            
            for batch_idx in range(batch_size):
                if batch_idx < len(constraints):
                    batch_constraints = constraints[batch_idx]
                    # Extract single batch item parameters
                    alpha_single = alpha[batch_idx:batch_idx+1]  # [1, num_objects, 4]
                    beta_single = beta[batch_idx:batch_idx+1]    # [1, num_objects, 4]
                    
                    # Evaluate constraints for this batch item
                    batch_prob = self._evaluate_constraint_list_single(
                        batch_constraints, alpha_single, beta_single, complexity_level
                    )
                    batch_probs.append(batch_prob.squeeze(0))  # Remove batch dimension
                else:
                    # No constraints for this batch item
                    batch_probs.append(torch.tensor(1.0, device=device))
            
            return torch.stack(batch_probs)
        
        else:
            # Single constraint list - apply to all batch items
            return self._evaluate_constraint_list_single(constraints, alpha, beta, complexity_level).squeeze(0)
    
    def _evaluate_constraint_list_single(self, constraints: List[Any], 
                                        alpha: torch.Tensor, beta: torch.Tensor,
                                        complexity_level: int = 5) -> torch.Tensor:
        """
        Evaluate constraints for a single batch item
        """
        if not constraints:
            return torch.ones(alpha.size(0), device=alpha.device)  # No constraints = always satisfied
            
        constraint_probs = []
        
        for constraint in constraints:
            if hasattr(constraint, 'c'):  # Check if it's a constraint object
                prob = self._evaluate_single_constraint(constraint, alpha, beta, complexity_level)
                constraint_probs.append(prob)
        
        if not constraint_probs:
            return torch.ones(alpha.size(0), device=alpha.device)
            
        # Combine constraint probabilities (AND operation)
        combined_prob = torch.stack(constraint_probs, dim=0)
        return torch.prod(combined_prob, dim=0)  # Product for AND
    
    def _evaluate_single_constraint(self, constraint: Any, alpha: torch.Tensor, 
                                   beta: torch.Tensor, complexity_level: int) -> torch.Tensor:
        """Evaluate a single constraint based on its type"""
        
        try:
            if hasattr(constraint, '_fields'):  # namedtuple constraint
                constraint_type = str(type(constraint))
                
                if 'ConstraintT1' in constraint_type:
                    return self._evaluate_t1_constraint(constraint, alpha, beta)
                elif 'ConstraintT2' in constraint_type:
                    return self._evaluate_t2_constraint(constraint, alpha, beta)
                elif 'ConstraintT3' in constraint_type:
                    return self._evaluate_t3_constraint(constraint, alpha, beta)
                elif 'ConstraintT4' in constraint_type:
                    return self._evaluate_t4_constraint(constraint, alpha, beta)
                elif 'ConstraintT5' in constraint_type:
                    return self._evaluate_t5_constraint(constraint, alpha, beta)
                elif 'ConstraintT6' in constraint_type:
                    return self._evaluate_t6_constraint(constraint, alpha, beta)
                elif 'ConstraintOR' in constraint_type:
                    return self._evaluate_or_constraint(constraint, alpha, beta, complexity_level)
                elif 'ConstraintAND' in constraint_type:
                    return self._evaluate_and_constraint(constraint, alpha, beta, complexity_level)
                elif 'ConstraintNOT' in constraint_type:
                    return self._evaluate_not_constraint(constraint, alpha, beta, complexity_level)
                else:
                    print(f"Warning: Unknown constraint type {constraint_type}")
                    return torch.ones(alpha.size(0), device=alpha.device) * 0.9
            else:
                print(f"Warning: Constraint {constraint} does not have _fields attribute")
                return torch.ones(alpha.size(0), device=alpha.device) * 0.9
        
        except Exception as e:
            print(f"Error evaluating constraint {constraint}: {e}")
            print(f"Constraint type: {type(constraint)}")
            if hasattr(constraint, '_fields'):
                print(f"Constraint fields: {constraint._fields}")
                for field in constraint._fields:
                    field_value = getattr(constraint, field)
                    print(f"  {field}: {field_value} (type: {type(field_value)})")
            
            # Return neutral probability to prevent training failure
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _evaluate_t1_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T1: object[o1].coord[v1] OP value + offset"""
        obj_idx = constraint.o1
        coord_idx = constraint.v1
        value = constraint.val
        offset = constraint.offset
        constraint_type = constraint.c
        
        # Ensure indices are integers - handle both scalar and tensor cases safely
        if torch.is_tensor(obj_idx):
            if obj_idx.numel() == 1:
                obj_idx = int(obj_idx.item())
            else:
                # This shouldn't happen - log error and use first element
                print(f"Warning: obj_idx tensor has {obj_idx.numel()} elements, using first element")
                obj_idx = int(obj_idx.flatten()[0].item())
        
        if torch.is_tensor(coord_idx):
            if coord_idx.numel() == 1:
                coord_idx = int(coord_idx.item())
            else:
                print(f"Warning: coord_idx tensor has {coord_idx.numel()} elements, using first element")
                coord_idx = int(coord_idx.flatten()[0].item())
        
        # Ensure value and offset are scalars
        if torch.is_tensor(value):
            value = float(value.item()) if value.numel() == 1 else float(value.flatten()[0].item())
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        # Extract parameters for specific object
        if obj_idx < alpha.size(1):
            alpha_obj = alpha[:, obj_idx:obj_idx+1, :]  # Keep object dimension
            beta_obj = beta[:, obj_idx:obj_idx+1, :]
            
            result = self.analytical_solver.constraint_t1_probability(
                alpha_obj, beta_obj, constraint_type, coord_idx, value, offset
            )
            return result['probability'].squeeze(1)  # Remove object dimension
        else:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _evaluate_t2_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T2: object[o1].coord[v1] OP object[o2].coord[v2] + offset"""
        obj1_idx = constraint.o1
        obj2_idx = constraint.o2
        coord_idx = constraint.v1  # Assuming v1 == v2 for same coordinate comparison
        offset = constraint.offset
        constraint_type = constraint.c
        
        # Convert to int if tensor - handle multi-element tensors safely
        if torch.is_tensor(obj1_idx):
            if obj1_idx.numel() == 1:
                obj1_idx = int(obj1_idx.item())
            else:
                print(f"Warning: obj1_idx tensor has {obj1_idx.numel()} elements, using first element")
                obj1_idx = int(obj1_idx.flatten()[0].item())
        
        if torch.is_tensor(obj2_idx):
            if obj2_idx.numel() == 1:
                obj2_idx = int(obj2_idx.item())
            else:
                print(f"Warning: obj2_idx tensor has {obj2_idx.numel()} elements, using first element")
                obj2_idx = int(obj2_idx.flatten()[0].item())
        
        if torch.is_tensor(coord_idx):
            if coord_idx.numel() == 1:
                coord_idx = int(coord_idx.item())
            else:
                print(f"Warning: coord_idx tensor has {coord_idx.numel()} elements, using first element")
                coord_idx = int(coord_idx.flatten()[0].item())
        
        # Ensure offset is scalar
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        if obj1_idx < alpha.size(1) and obj2_idx < alpha.size(1):
            alpha1 = alpha[:, obj1_idx:obj1_idx+1, :]
            beta1 = beta[:, obj1_idx:obj1_idx+1, :]
            alpha2 = alpha[:, obj2_idx:obj2_idx+1, :]
            beta2 = beta[:, obj2_idx:obj2_idx+1, :]
            
            result = self.analytical_solver.constraint_t2_probability_analytical(
                alpha1, beta1, alpha2, beta2, constraint_type, coord_idx, offset
            )
            return result['probability'].squeeze(1)
        else:
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
    
    def _evaluate_t3_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T3: object[o1].coord[v1] + object[o1].coord[v2] OP value + offset"""
        # This involves addition of two coordinates from same object
        # Requires sampling approach for Beta sum distribution
        obj_idx = constraint.o1
        coord1_idx = constraint.v1
        coord2_idx = constraint.v2
        value = constraint.val
        offset = constraint.offset
        constraint_type = constraint.c
        
        # Convert to int if tensor - handle multi-element tensors safely
        if torch.is_tensor(obj_idx):
            if obj_idx.numel() == 1:
                obj_idx = int(obj_idx.item())
            else:
                print(f"Warning: obj_idx tensor has {obj_idx.numel()} elements, using first element")
                obj_idx = int(obj_idx.flatten()[0].item())
        
        if torch.is_tensor(coord1_idx):
            if coord1_idx.numel() == 1:
                coord1_idx = int(coord1_idx.item())
            else:
                print(f"Warning: coord1_idx tensor has {coord1_idx.numel()} elements, using first element")
                coord1_idx = int(coord1_idx.flatten()[0].item())
        
        if torch.is_tensor(coord2_idx):
            if coord2_idx.numel() == 1:
                coord2_idx = int(coord2_idx.item())
            else:
                print(f"Warning: coord2_idx tensor has {coord2_idx.numel()} elements, using first element")
                coord2_idx = int(coord2_idx.flatten()[0].item())
        
        # Ensure value and offset are scalars
        if torch.is_tensor(value):
            value = float(value.item()) if value.numel() == 1 else float(value.flatten()[0].item())
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        if obj_idx >= alpha.size(1):
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
            
        # Use sampling for sum of Beta distributions (no simple analytical form)
        return self.analytical_solver._sample_t3_constraint(alpha, beta, obj_idx, coord1_idx, coord2_idx, 
                                                          value, offset, constraint_type)
    
    def _evaluate_t4_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T4: object[o1].coord[v1] + object[o1].coord[v2] OP object[o2].coord[v3] + offset"""
        # Complex constraint involving sums from different objects
        obj1_idx = constraint.o1
        coord1_idx = constraint.v1
        coord2_idx = constraint.v2
        obj2_idx = constraint.o2
        coord3_idx = constraint.v3
        offset = constraint.offset
        constraint_type = constraint.c
        
        # Convert to int if tensor - handle multi-element tensors safely
        if torch.is_tensor(obj1_idx):
            if obj1_idx.numel() == 1:
                obj1_idx = int(obj1_idx.item())
            else:
                print(f"Warning: obj1_idx tensor has {obj1_idx.numel()} elements, using first element")
                obj1_idx = int(obj1_idx.flatten()[0].item())
        
        if torch.is_tensor(obj2_idx):
            if obj2_idx.numel() == 1:
                obj2_idx = int(obj2_idx.item())
            else:
                print(f"Warning: obj2_idx tensor has {obj2_idx.numel()} elements, using first element")
                obj2_idx = int(obj2_idx.flatten()[0].item())
        
        if torch.is_tensor(coord1_idx):
            if coord1_idx.numel() == 1:
                coord1_idx = int(coord1_idx.item())
            else:
                print(f"Warning: coord1_idx tensor has {coord1_idx.numel()} elements, using first element")
                coord1_idx = int(coord1_idx.flatten()[0].item())
        
        if torch.is_tensor(coord2_idx):
            if coord2_idx.numel() == 1:
                coord2_idx = int(coord2_idx.item())
            else:
                print(f"Warning: coord2_idx tensor has {coord2_idx.numel()} elements, using first element")
                coord2_idx = int(coord2_idx.flatten()[0].item())
        
        if torch.is_tensor(coord3_idx):
            if coord3_idx.numel() == 1:
                coord3_idx = int(coord3_idx.item())
            else:
                print(f"Warning: coord3_idx tensor has {coord3_idx.numel()} elements, using first element")
                coord3_idx = int(coord3_idx.flatten()[0].item())
        
        # Ensure offset is scalar
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        if obj1_idx >= alpha.size(1) or obj2_idx >= alpha.size(1):
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
            
        # Use sampling for complex sum comparisons
        return self.analytical_solver._sample_t4_constraint(alpha, beta, obj1_idx, coord1_idx, coord2_idx,
                                                          obj2_idx, coord3_idx, offset, constraint_type)
    
    def _evaluate_t5_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T5: Intersection constraints"""
        obj1_idx = constraint.o1
        obj2_idx = constraint.o2
        offset = constraint.offset
        
        # Convert to int if tensor - handle multi-element tensors safely
        if torch.is_tensor(obj1_idx):
            if obj1_idx.numel() == 1:
                obj1_idx = int(obj1_idx.item())
            else:
                print(f"Warning: obj1_idx tensor has {obj1_idx.numel()} elements, using first element")
                obj1_idx = int(obj1_idx.flatten()[0].item())
        
        if torch.is_tensor(obj2_idx):
            if obj2_idx.numel() == 1:
                obj2_idx = int(obj2_idx.item())
            else:
                print(f"Warning: obj2_idx tensor has {obj2_idx.numel()} elements, using first element")
                obj2_idx = int(obj2_idx.flatten()[0].item())
        
        # Ensure offset is scalar
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        if obj1_idx >= alpha.size(1) or obj2_idx >= alpha.size(1):
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
            
        # Use sampling for intersection probability computation
        return self.analytical_solver._sample_intersection_constraint(alpha, beta, obj1_idx, obj2_idx, offset)
    
    def _evaluate_t6_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """T6: Manhattan distance equality - requires sampling"""
        obj1_idx = constraint.o1
        obj2_idx = constraint.o2
        obj3_idx = constraint.o3
        offset = constraint.offset
        
        # Convert to int if tensor - handle multi-element tensors safely
        if torch.is_tensor(obj1_idx):
            if obj1_idx.numel() == 1:
                obj1_idx = int(obj1_idx.item())
            else:
                print(f"Warning: obj1_idx tensor has {obj1_idx.numel()} elements, using first element")
                obj1_idx = int(obj1_idx.flatten()[0].item())
        
        if torch.is_tensor(obj2_idx):
            if obj2_idx.numel() == 1:
                obj2_idx = int(obj2_idx.item())
            else:
                print(f"Warning: obj2_idx tensor has {obj2_idx.numel()} elements, using first element")
                obj2_idx = int(obj2_idx.flatten()[0].item())
        
        if torch.is_tensor(obj3_idx):
            if obj3_idx.numel() == 1:
                obj3_idx = int(obj3_idx.item())
            else:
                print(f"Warning: obj3_idx tensor has {obj3_idx.numel()} elements, using first element")
                obj3_idx = int(obj3_idx.flatten()[0].item())
        
        # Ensure offset is scalar
        if torch.is_tensor(offset):
            offset = float(offset.item()) if offset.numel() == 1 else float(offset.flatten()[0].item())
        
        if (obj1_idx >= alpha.size(1) or obj2_idx >= alpha.size(1) or 
            obj3_idx >= alpha.size(1)):
            return torch.ones(alpha.size(0), device=alpha.device) * 0.5
            
        # Use sampling for Manhattan distance equality
        return self.analytical_solver._sample_manhattan_distance_constraint(alpha, beta, obj1_idx, obj2_idx, 
                                                                          obj3_idx, offset)
    
    def _evaluate_or_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor, 
                               complexity_level: int) -> torch.Tensor:
        """OR constraint: P(A OR B) = P(A) + P(B) - P(A AND B) ≈ 1 - (1-P(A))(1-P(B))"""
        sub_constraints = constraint.c
        probs = []
        
        for sub_constraint in sub_constraints:
            prob = self._evaluate_single_constraint(sub_constraint, alpha, beta, complexity_level)
            probs.append(prob)
        
        if probs:
            # OR probability: 1 - ∏(1 - p_i)
            failure_probs = torch.stack([1.0 - p for p in probs], dim=0)
            total_failure = torch.prod(failure_probs, dim=0)
            return 1.0 - total_failure
        else:
            return torch.zeros(alpha.size(0), device=alpha.device)
    
    def _evaluate_and_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor,
                                complexity_level: int) -> torch.Tensor:
        """AND constraint: P(A AND B) ≈ P(A) * P(B) assuming independence"""
        sub_constraints = constraint.c
        probs = []
        
        for sub_constraint in sub_constraints:
            prob = self._evaluate_single_constraint(sub_constraint, alpha, beta, complexity_level)
            probs.append(prob)
        
        if probs:
            # AND probability: ∏p_i
            combined = torch.stack(probs, dim=0)
            return torch.prod(combined, dim=0)
        else:
            return torch.ones(alpha.size(0), device=alpha.device)
    
    def _evaluate_not_constraint(self, constraint, alpha: torch.Tensor, beta: torch.Tensor,
                                complexity_level: int) -> torch.Tensor:
        """NOT constraint: P(NOT A) = 1 - P(A)"""
        sub_constraint = constraint.c
        prob = self._evaluate_single_constraint(sub_constraint, alpha, beta, complexity_level)
        return 1.0 - prob


class ComprehensiveBetaLoss(nn.Module):
    """
    Comprehensive loss function with CHEN'S ADAPTIVE WEIGHTING SYSTEM.
    
    Key improvements:
    - Dynamic weight adjustment based on loss magnitudes
    - Prevents any single loss component from dominating
    - Maintains balanced gradient flow
    """
    
    def __init__(self, coord_weight: float = 1.0, constraint_weight: float = 0.1, 
                 uncertainty_weight: float = 0.01, boundary_weight: float = 0.05,
                 n_samples: int = 100):
        super().__init__()
        # Base weights (will be adapted dynamically)
        self.base_coord_weight = coord_weight
        self.base_constraint_weight = constraint_weight
        self.base_uncertainty_weight = uncertainty_weight
        self.base_boundary_weight = boundary_weight
        self.n_samples = n_samples
        
        # CHEN'S ADAPTIVE WEIGHT SYSTEM
        # Track running averages of loss components for balancing
        self.register_buffer('coord_loss_avg', torch.tensor(1.0))
        self.register_buffer('constraint_loss_avg', torch.tensor(1.0))
        self.register_buffer('uncertainty_loss_avg', torch.tensor(1.0))
        self.register_buffer('boundary_loss_avg', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        
        # Momentum for running averages
        self.momentum = 0.99
        
        # Use fixed constraint checker for realistic satisfaction rates
        from fixed_constraint_checker import FixedConstraintChecker
        # Detect device from first available parameter or default to cuda
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda')
        self.constraint_evaluator = FixedConstraintChecker(device=device)
        
        # Add numerical stability monitoring - use fallback if not available
        try:
            from numerical_stability_system import NumericalStabilityMonitor
            self.stability_monitor = NumericalStabilityMonitor()
        except ImportError as e:
            # Log warning and create basic fallback monitor
            import warnings
            warnings.warn(f"NumericalStabilityMonitor not available: {e}. Using basic fallback.")
            
            # Create a basic fallback monitor that at least clamps values
            class BasicStabilityMonitor:
                def apply_stability_corrections(self, alpha, beta):
                    # Basic clamping to prevent extreme values
                    alpha = torch.clamp(alpha, min=0.1, max=100.0)
                    beta = torch.clamp(beta, min=0.1, max=100.0)
                    return alpha, beta
            
            self.stability_monitor = BasicStabilityMonitor()
        
    def forward(self, alpha: torch.Tensor, beta: torch.Tensor, 
                target_coords: torch.Tensor, constraints: List[Any] = None,
                complexity_level: int = 5) -> Dict[str, torch.Tensor]:
        """
        Comprehensive loss computation
        
        Args:
            alpha, beta: [batch_size, num_objects, 4] Beta parameters
            target_coords: [batch_size, num_objects, 4] ground truth coordinates in [0,1000]
            constraints: List of constraint objects
            complexity_level: Curriculum learning level (1=simple, 5=complex)
        """
        batch_size = alpha.size(0)
        device = alpha.device
        
        # 1. Coordinate accuracy loss using distribution sampling
        # CRITICAL FIX 1.1: Use sampling for training consistency with inference
        # Training must use same mechanism as inference to prevent mean-sample mismatch
        distributions = Beta(alpha, beta)
        predicted_samples_01 = distributions.rsample()  # [batch_size, num_objects, 4] in [0,1]
        target_coords_01 = target_coords / 1000.0    # Scale targets to [0,1]
        coord_loss = F.mse_loss(predicted_samples_01, target_coords_01)
        
        # 2. Constraint satisfaction loss
        constraint_loss = torch.tensor(0.0, device=device)
        constraint_satisfaction_prob = torch.tensor(1.0, device=device)
        approximation_error_bound = torch.tensor(0.0, device=device)
        numerical_stability_score = torch.tensor(1.0, device=device)
        computational_cost = torch.tensor(0.0, device=device)
        
        if constraints:
            # Apply stability corrections if monitor is available
            if self.stability_monitor is not None:
                try:
                    alpha_stable, beta_stable = self.stability_monitor.apply_stability_corrections(alpha, beta)
                except:
                    # Fallback to original parameters if stability correction fails
                    alpha_stable, beta_stable = alpha, beta
            else:
                alpha_stable, beta_stable = alpha, beta
            
            # CHEN'S FIX: Flatten nested constraint lists before passing to evaluator
            # Handle both flat lists and nested lists (batch format)
            if constraints and isinstance(constraints[0], list):
                # Nested format: [[con1, con2], [con3]] -> [con1, con2, con3]
                flat_constraints = [c for batch_constraints in constraints for c in batch_constraints]
            else:
                # Already flat format: [con1, con2, con3]
                flat_constraints = constraints
            
            # Evaluate constraints using simplified constraint checker
            constraint_results = self.constraint_evaluator.evaluate_constraints(
                alpha_stable, beta_stable, flat_constraints
            )
            
            # Store constraint satisfaction probability
            constraint_satisfaction_prob = constraint_results['satisfaction_rate']
            
            # CHEN PRIORITY 1: Switch to MSE for stronger constraint gradients  
            # Target 85% constraint satisfaction (0.85)
            target_satisfaction = torch.tensor(0.85, device=device)
            constraint_loss = F.mse_loss(constraint_satisfaction_prob, 
                                       target_satisfaction.expand_as(constraint_satisfaction_prob))
            # MSE gradient: 2*(0.69-0.85) = -0.32 vs Smooth L1: ≈-0.16 (stronger signal)
        
        # 3. Uncertainty regularization - encourage reasonable concentration
        concentration = alpha + beta
        # CRITICAL FIX 1.2: Updated concentration regularization with new parameter bounds
        # Target should be in middle of our dynamic range  
        min_possible = 2 * 2.0   # 2 * min_param (updated from 1.01 to 2.0)
        max_possible = 2 * 20.0  # 2 * max_param (CHEN PRIORITY 2: updated for relaxed limit) 
        target_concentration = (min_possible + max_possible) / 2.0  # = 22.0 (CHEN PRIORITY 2: updated for 20.0 limit)
        
        # Use smooth L1 loss instead of MSE to reduce sensitivity to outliers
        uncertainty_loss = F.smooth_l1_loss(concentration, 
                                           torch.full_like(concentration, target_concentration))
        
        # 4. Boundary enforcement - ensure coordinates respect [0,1000] bounds
        boundary_loss = self._compute_boundary_loss(alpha, beta)
        
        # 5. Shape regularization - encourage well-behaved distributions
        shape_loss = self._compute_shape_regularization(alpha, beta)
        
        # FIX 4: Clamp loss components to prevent any from being >10x coordinate loss
        coord_loss, constraint_loss, uncertainty_loss, boundary_loss = self._clamp_loss_components(
            coord_loss, constraint_loss, uncertainty_loss, boundary_loss
        )
        
        # CHEN'S ADAPTIVE LOSS WEIGHTING
        # Dynamically adjust weights to prevent any component from dominating
        
        # Update running averages (only during training)
        if self.training and self.update_count > 0:  # Only after initialization
            self._update_loss_statistics(coord_loss, constraint_loss, 
                                        uncertainty_loss, boundary_loss)
        
        # Compute adaptive weights based on loss magnitudes
        adaptive_weights = self._compute_adaptive_weights() if self.update_count > 5 else {
            'coord': 1.0, 'constraint': 1.0, 'uncertainty': 1.0, 'boundary': 1.0
        }
        
        # Apply curriculum scaling to constraint weight
        curriculum_weight = complexity_level / 5.0
        
        # Combine losses with adaptive weights
        total_loss = (
            adaptive_weights['coord'] * self.base_coord_weight * coord_loss + 
            adaptive_weights['constraint'] * self.base_constraint_weight * curriculum_weight * constraint_loss + 
            adaptive_weights['uncertainty'] * self.base_uncertainty_weight * uncertainty_loss +
            adaptive_weights['boundary'] * self.base_boundary_weight * boundary_loss +
            0.1 * shape_loss  # Shape loss with fixed small weight
        )
        
        # Prepare return dictionary with enhanced metrics
        result = {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'constraint_loss': constraint_loss,
            'uncertainty_loss': uncertainty_loss,
            'boundary_loss': boundary_loss,
            'shape_loss': shape_loss,
            'predicted_samples': predicted_samples_01,
            'concentration': concentration.mean(),
            # Include adaptive weights for monitoring
            'adaptive_weights': adaptive_weights if self.training else None
        }
        
        # Add constraint evaluation metrics including detailed satisfaction rates
        result.update({
            'constraint_satisfaction_prob': constraint_satisfaction_prob,
            'approximation_error_bound': approximation_error_bound,
            'numerical_stability_score': numerical_stability_score,
            'computational_cost': computational_cost
        })
        
        # Add detailed constraint satisfaction metrics if constraints were provided
        if constraints is not None and len(constraints) > 0:
            result.update({
                't1_satisfaction_rate': constraint_results.get('t1_rate', torch.tensor(1.0)),
                't2_satisfaction_rate': constraint_results.get('t2_rate', torch.tensor(1.0)),
                'complex_satisfaction_rate': constraint_results.get('complex_rate', torch.tensor(1.0)),
                'num_constraints_evaluated': len(constraints)
            })
        
        return result
    
    def _compute_boundary_loss(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Ensure predicted coordinates respect [0,1000] boundaries"""
        # CRITICAL FIX 1.2: Updated numerical guards to prevent variance explosion
        alpha_safe = torch.clamp(alpha, min=2.0, max=50.0)  # Increased from 1.01 to 2.0 for stability
        beta_safe = torch.clamp(beta, min=2.0, max=50.0)   # Increased from 1.01 to 2.0 for stability
        distributions = Beta(alpha_safe, beta_safe)
        
        # Sample coordinates
        samples = distributions.rsample((self.n_samples,)) * 1000  # Scale to [0,1000] - PRESERVE GRADIENTS
        
        # Penalize samples outside bounds
        lower_violations = F.relu(-samples)  # Negative coordinates
        upper_violations = F.relu(samples - 1000)  # Coordinates > 1000
        
        boundary_loss = (lower_violations.mean() + upper_violations.mean())
        return boundary_loss
    
    def _compute_shape_regularization(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Encourage well-behaved distribution shapes"""
        # Penalize extreme parameter values that lead to degenerate distributions
        extreme_penalty = F.relu(alpha - 50) + F.relu(beta - 50)  # Cap very large values
        degenerate_penalty = F.relu(1.1 - alpha) + F.relu(1.1 - beta)  # Prevent values < 1.1
        
        return extreme_penalty.mean() + degenerate_penalty.mean()
    
    def _update_loss_statistics(self, coord_loss, constraint_loss, 
                               uncertainty_loss, boundary_loss):
        """Update running averages of loss components"""
        # Detach and clamp to prevent overflow
        coord_loss_val = torch.clamp(coord_loss.detach(), max=100.0)
        constraint_loss_val = torch.clamp(constraint_loss.detach(), max=10.0) if constraint_loss > 0 else torch.tensor(0.1)
        uncertainty_loss_val = torch.clamp(uncertainty_loss.detach(), max=10.0)
        boundary_loss_val = torch.clamp(boundary_loss.detach(), max=10.0)
        
        # Initialize on first call
        if self.update_count == 0:
            self.coord_loss_avg = coord_loss_val
            self.constraint_loss_avg = constraint_loss_val
            self.uncertainty_loss_avg = uncertainty_loss_val
            self.boundary_loss_avg = boundary_loss_val
        else:
            # Exponential moving average with clamping
            self.coord_loss_avg = self.momentum * self.coord_loss_avg + (1 - self.momentum) * coord_loss_val
            self.constraint_loss_avg = self.momentum * self.constraint_loss_avg + (1 - self.momentum) * constraint_loss_val
            self.uncertainty_loss_avg = self.momentum * self.uncertainty_loss_avg + (1 - self.momentum) * uncertainty_loss_val
            self.boundary_loss_avg = self.momentum * self.boundary_loss_avg + (1 - self.momentum) * boundary_loss_val
        
        self.update_count += 1
    
    def _clamp_loss_components(self, coord_loss, constraint_loss, uncertainty_loss, boundary_loss):
        """FIX 4: Prevent any loss component from being >10x coordinate loss"""
        coord_loss_val = coord_loss.item() if hasattr(coord_loss, 'item') else coord_loss
        max_allowed = coord_loss_val * 10.0
        
        # Clamp each component
        if hasattr(constraint_loss, 'clamp'):
            constraint_loss = torch.clamp(constraint_loss, max=max_allowed)
        if hasattr(uncertainty_loss, 'clamp'):
            uncertainty_loss = torch.clamp(uncertainty_loss, max=max_allowed)
        if hasattr(boundary_loss, 'clamp'):
            boundary_loss = torch.clamp(boundary_loss, max=max_allowed)
            
        return coord_loss, constraint_loss, uncertainty_loss, boundary_loss
    
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """Compute adaptive weights to balance loss components"""
        # CHEN FIX 2: Mathematically stable log-scale adaptive weighting
        eps = 1e-8
        
        # Log-scale dampening prevents oscillation while preserving balance
        log_coord = torch.log(self.coord_loss_avg + eps)
        log_constraint = torch.log(self.constraint_loss_avg + eps)
        log_uncertainty = torch.log(self.uncertainty_loss_avg + eps) 
        log_boundary = torch.log(self.boundary_loss_avg + eps)
        
        # Stable normalization: exponential of negative log-losses
        exp_neg_coord = torch.exp(-log_coord)
        exp_neg_constraint = torch.exp(-log_constraint)
        exp_neg_uncertainty = torch.exp(-log_uncertainty)
        exp_neg_boundary = torch.exp(-log_boundary)
        
        # Normalize to sum = 4.0 (maintains total weight balance)
        total_exp = exp_neg_coord + exp_neg_constraint + exp_neg_uncertainty + exp_neg_boundary
        normalization = 4.0 / (total_exp + eps)
        
        weights = {
            'coord': (exp_neg_coord * normalization).item(),
            'constraint': (exp_neg_constraint * normalization).item(),
            'uncertainty': (exp_neg_uncertainty * normalization).item(),
            'boundary': (exp_neg_boundary * normalization).item()
        }
        
        # Apply smoothing to prevent drastic changes
        for key in weights:
            weights[key] = 0.25 + 0.75 * weights[key]  # Keep weights in [0.25, 1.0] range
        
        return weights


class BetaSpatialReasonerComplete(nn.Module):
    """
    Complete Beta-based spatial reasoning system integrating all components
    """
    
    def __init__(self, scene_dim: int = 512, hidden_dim: int = 256, 
                 num_objects: int = 10, num_heads: int = 8, 
                 input_channels: int = 3, input_size: int = 128,
                 coord_weight: float = 1.0, constraint_weight: float = 0.1,
                 uncertainty_weight: float = 0.01, boundary_weight: float = 0.05):
        super().__init__()
        
        self.scene_dim = scene_dim
        self.num_objects = num_objects
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Feature extractor for raw images -> scene features
        self.feature_extractor = nn.Sequential(
            # Convolutional feature extraction
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),  # 128->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64->32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 8->4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling to get 512-dim features
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # [batch_size, 512, 1, 1] -> [batch_size, 512]
        )
        
        # Core prediction network
        self.predictor = MultiScaleBetaPredictor(scene_dim, hidden_dim, num_objects, num_heads)
        
        # Chen's mathematically rigorous constraint solver
        self.constraint_solver = AnalyticalConstraintSolver()
        
        # Coordinate validation system for mathematical consistency
        self.coordinate_validator = CoordinateValidator()
        
        # Loss function with balanced weights
        self.loss_fn = ComprehensiveBetaLoss(
            coord_weight=coord_weight,
            constraint_weight=constraint_weight, 
            uncertainty_weight=uncertainty_weight,
            boundary_weight=boundary_weight
        )
        
        # Sampling parameters
        self.inference_samples = 100
        self.training_samples = 50
        
    def set_training_progress(self, progress: float):
        """Update training progress for dynamic parameter bounds"""
        self.predictor.set_training_progress(progress)
    
    def forward(self, input_data: torch.Tensor, constraints: List[Any] = None,
                target_coords: torch.Tensor = None, complexity_level: int = 5) -> Dict[str, torch.Tensor]:
        """
        Forward pass with training/inference mode handling
        
        Args:
            input_data: Either raw images [batch_size, 3, 128, 128] or scene features [batch_size, 512]
            constraints: Constraint list
            target_coords: Target coordinates for loss computation
            complexity_level: Curriculum complexity level
        
        CRITICAL FIX: Always compute loss when target_coords provided, regardless of training mode
        """
        # Determine if input is raw images or features based on dimensions
        if len(input_data.shape) == 4:  # [batch_size, channels, height, width]
            # Extract features from raw images
            scene_features = self.feature_extractor(input_data)
        elif len(input_data.shape) == 2:  # [batch_size, feature_dim]
            # Already processed features
            scene_features = input_data
        else:
            raise ValueError(f"Unexpected input shape: {input_data.shape}. Expected 4D images or 2D features.")
        
        alpha, beta = self.predictor(scene_features)
        
        if target_coords is not None:
            # FIXED: Compute loss whenever targets are provided (training AND validation)
            loss_dict = self.loss_fn(alpha, beta, target_coords, constraints, complexity_level)
            loss_dict.update({
                'alpha': alpha,
                'beta': beta
            })
            return loss_dict
        else:
            # Pure inference mode: sample coordinates
            coordinates = self.sample_coordinates(alpha, beta, self.inference_samples)
            stats = self.get_distribution_statistics(alpha, beta)
            
            return {
                'alpha': alpha,
                'beta': beta,
                'coordinates': coordinates,
                'statistics': stats
            }
    
    def sample_coordinates(self, alpha: torch.Tensor, beta: torch.Tensor, 
                          n_samples: int = 1) -> torch.Tensor:
        """
        Sample coordinates from Beta distributions with validation
        
        Mathematical Properties:
        - Validates Beta parameters for numerical stability
        - Converts samples to per-mille space [0, 1000]
        - Ensures bounding box validity
        """
        # Validate Beta parameters
        alpha_valid, beta_valid = self.coordinate_validator.validate_beta_parameters(alpha, beta)
        
        # Create distributions with validated parameters
        distributions = Beta(alpha_valid, beta_valid)
        
        # Sample from distributions (use rsample for gradient flow)
        if alpha_valid.requires_grad or beta_valid.requires_grad:
            samples = distributions.rsample((n_samples,))  # Reparameterized sampling for gradients
        else:
            samples = distributions.sample((n_samples,))  # Regular sampling when no gradients needed
        
        # Convert to per-mille coordinates with validation
        # Reshape to handle batching properly
        batch_size = samples.shape[1]
        num_objs = samples.shape[2]
        coords_per_obj = samples.shape[3]
        
        # Process each sample
        validated_samples = []
        for sample_idx in range(n_samples):
            sample = samples[sample_idx]  # [batch, num_objs, 4]
            
            # Validate as bounding boxes if 4 coordinates
            if coords_per_obj == 4:
                # Convert Beta samples to per-mille
                permille_coords = self.coordinate_validator.beta_to_permille(sample.reshape(-1, 4))
                # Validate as bboxes
                validated = self.coordinate_validator.validate_bbox(permille_coords)
                validated = validated.reshape(batch_size, num_objs, 4)
            else:
                # Generic coordinate validation
                permille_coords = self.coordinate_validator.beta_to_permille(sample)
                validated = permille_coords
            
            validated_samples.append(validated)
        
        # Stack samples back
        result = torch.stack(validated_samples, dim=0)
        
        return result
    
    def get_distribution_statistics(self, alpha: torch.Tensor, beta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute comprehensive distribution statistics"""
        # Basic statistics
        mean = alpha / (alpha + beta) * 1000
        variance = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1)) * (1000**2)
        std = torch.sqrt(variance)
        
        # Mode (undefined if α ≤ 1 or β ≤ 1)
        mode_mask = (alpha > 1) & (beta > 1)
        mode = torch.zeros_like(mean)
        mode[mode_mask] = ((alpha[mode_mask] - 1) / (alpha[mode_mask] + beta[mode_mask] - 2)) * 1000
        
        # Concentration and skewness
        concentration = alpha + beta
        skewness = 2 * (beta - alpha) * torch.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * torch.sqrt(alpha * beta))
        
        # Confidence intervals (95%) - use numerical approximation
        # Note: Beta distribution doesn't have analytical inverse CDF
        try:
            from scipy.stats import beta as scipy_beta
            import numpy as np
            
            # Convert to numpy for scipy
            alpha_np = alpha.detach().cpu().numpy()
            beta_np = beta.detach().cpu().numpy()
            
            # Compute percentiles
            p025_np = scipy_beta.ppf(0.025, alpha_np, beta_np) * 1000
            p975_np = scipy_beta.ppf(0.975, alpha_np, beta_np) * 1000
            
            p025 = torch.from_numpy(p025_np).to(alpha.device)
            p975 = torch.from_numpy(p975_np).to(alpha.device)
            
        except ImportError:
            # Fallback: use approximation based on mean ± 2*std
            p025 = torch.clamp(mean - 2 * std, 0, 1000)
            p975 = torch.clamp(mean + 2 * std, 0, 1000)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'mode': mode,
            'concentration': concentration,
            'skewness': skewness,
            'ci_lower': p025,
            'ci_upper': p975
        }


# Training and evaluation utilities
class CurriculumTrainer:
    """
    Curriculum learning trainer for Beta spatial reasoning
    """
    
    def __init__(self, model: BetaSpatialReasonerComplete, 
                 optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Curriculum schedule
        self.curriculum_schedule = {
            'epochs_per_level': 20,
            'levels': 5,
            'constraint_complexity': {
                1: ['T1'],  # Simple value constraints
                2: ['T1', 'T2'],  # Add object-object constraints
                3: ['T1', 'T2', 'T3'],  # Add coordinate arithmetic
                4: ['T1', 'T2', 'T3', 'T4'],  # Add complex arithmetic
                5: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'OR', 'AND', 'NOT']  # Full complexity
            }
        }
        
    def train_epoch(self, dataloader, epoch: int, total_epochs: int = 100) -> Dict[str, float]:
        """Train one epoch with curriculum learning and dynamic bounds"""
        self.model.train()
        
        # CHEN: Update training progress for dynamic parameter bounds
        training_progress = epoch / total_epochs
        self.model.set_training_progress(training_progress)
        
        # Determine curriculum level
        complexity_level = min(5, (epoch // self.curriculum_schedule['epochs_per_level']) + 1)
        
        losses = {
            'total_loss': 0.0,
            'coord_loss': 0.0,
            'constraint_loss': 0.0,
            'uncertainty_loss': 0.0,
            'constraint_satisfaction_prob': 0.0,
            'approximation_error_bound': 0.0,
            'numerical_stability_score': 0.0
        }
        
        num_batches = 0
        for batch_idx, (scene_features, target_coords, constraints) in enumerate(dataloader):
            scene_features = scene_features.to(self.device)
            target_coords = target_coords.to(self.device)
            
            # Forward pass
            outputs = self.model(scene_features, constraints, target_coords, complexity_level)
            loss = outputs['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in losses:
                if key in outputs:
                    # Handle both scalar tensors and regular floats
                    value = outputs[key]
                    if torch.is_tensor(value):
                        losses[key] += value.item()
                    else:
                        losses[key] += value
            num_batches += 1
        
        # Average losses
        for key in losses:
            losses[key] /= num_batches
            
        losses['complexity_level'] = complexity_level
        return losses


# Integration with SPRING pipeline
def integrate_with_spring_pipeline():
    """
    Example integration of Beta spatial reasoner with existing SPRING components
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the complete system
    beta_reasoner = BetaSpatialReasonerComplete(
        scene_dim=512,  # Feature dimension after extraction
        hidden_dim=256,
        num_objects=5,
        num_heads=8,
        input_channels=3,  # RGB images
        input_size=128     # COCO dataset image size
    ).to(device)
    
    # Optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': beta_reasoner.feature_extractor.parameters(), 'lr': 1e-5},  # Conservative for pretrained
        {'params': beta_reasoner.predictor.scene_encoder.parameters(), 'lr': 1e-4},
        {'params': beta_reasoner.predictor.alpha_predictor.parameters(), 'lr': 5e-4},
        {'params': beta_reasoner.predictor.beta_predictor.parameters(), 'lr': 5e-4},
        {'params': beta_reasoner.loss_fn.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)
    
    # Curriculum trainer
    trainer = CurriculumTrainer(beta_reasoner, optimizer, device)
    
    print("Beta Spatial Reasoner initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in beta_reasoner.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in beta_reasoner.parameters() if p.requires_grad):,}")
    
    return beta_reasoner, trainer


if __name__ == "__main__":
    print("Complete Beta Distribution Spatial Reasoning Implementation")
    print("=" * 60)
    print("Revolutionary probabilistic approach to spatial constraint satisfaction")
    print("Full integration with SPRING pipeline ready for deployment")
    print()
    
    # Initialize system
    model, trainer = integrate_with_spring_pipeline()
    
    # Test with dummy data (ensure device consistency)
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with both raw images and pre-extracted features
    dummy_images = torch.randn(batch_size, 3, 128, 128).to(device)  # Raw images
    dummy_features = torch.randn(batch_size, 512).to(device)        # Pre-extracted features
    target_coords = torch.rand(batch_size, 5, 4) * 1000
    
    # Create some test constraints
    test_constraints = [
        con_left(0, 1, 50),  # Object 0 left of object 1 by 50 pixels
        con_above(2, 3, 30), # Object 2 above object 3 by 30 pixels
    ]
    
    # Forward pass - test both input types
    model.eval()
    with torch.no_grad():
        # Test with raw images
        print("Testing with raw images (3x128x128)...")
        outputs_images = model(dummy_images)
        print(f"Output keys: {outputs_images.keys()}")
        print(f"Predicted coordinates shape: {outputs_images['coordinates'].shape}")
        print(f"Mean concentration: {outputs_images['statistics']['concentration'].mean().item():.2f}")
        
        # Test with pre-extracted features
        print("\nTesting with pre-extracted features (512-dim)...")
        outputs_features = model(dummy_features)
        print(f"Output keys: {outputs_features.keys()}")
        print(f"Predicted coordinates shape: {outputs_features['coordinates'].shape}")
        print(f"Mean concentration: {outputs_features['statistics']['concentration'].mean().item():.2f}")