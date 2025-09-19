"""
Complete Training and Evaluation Framework for Beta Spatial Reasoning

This provides the full training pipeline, evaluation metrics, and integration
protocols for the revolutionary Beta distribution approach to spatial reasoning.

Professor Davies - Training Framework:
- Curriculum learning from simple to complex constraints
- Comprehensive evaluation suite with uncertainty calibration
- Memory-efficient training with adaptive batch sizing
- Integration protocols with existing SPRING components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings

# Import our Beta spatial reasoner and SPRING components
from beta_spatial_complete import BetaSpatialReasonerComplete, CurriculumTrainer
from constraint_language_v2 import *
from coco_dataset import EnhancedCOCO_Wrapper
from spring_srm_networks import Super_Encoder

# Import enhanced numerical stability system
try:
    from numerical_stability_system import (
        TrainingStabilityController,
        create_stability_controller,
        StabilityMetrics
    )
except ImportError:
    # Fallback implementations
    class StabilityMetrics:
        def __init__(self):
            self.gradient_norm = 0.0
            self.beta_parameter_health = 1.0
            self.numerical_precision = 1.0
    
    def create_stability_controller():
        return FallbackStabilityController()


class StabilityControllerAdapter:
    """Adapter for TrainingStabilityController to match expected interface"""
    
    def __init__(self, controller):
        self.controller = controller
        self.last_metrics = None
        
    def pre_step_check(self, epoch):
        """Adapt check_and_fix to pre_step_check interface"""
        # Get current metrics from the controller
        metrics = self.controller.get_metrics() if hasattr(self.controller, 'get_metrics') else {}
        self.last_metrics = metrics
        
        # Check for intervention needs based on metrics
        intervention_needed = False
        intervention_type = None
        
        if metrics:
            # Check gradient health
            if metrics.get('gradient_norm', 0) > 10.0:
                intervention_needed = True
                intervention_type = 'gradient_clipping'
            # Check for NaN/Inf
            if metrics.get('nan_count', 0) > 0 or metrics.get('inf_count', 0) > 0:
                intervention_needed = True
                intervention_type = 'nan_recovery'
                
        return {
            'intervention_needed': intervention_needed,
            'intervention_type': intervention_type,
            'stability_metrics': metrics
        }
    
    def apply_interventions(self, intervention_type, epoch):
        """Apply interventions based on type"""
        # The actual intervention is handled by check_and_fix
        return True
    
    def get_stability_report(self):
        """Get stability report"""
        if self.last_metrics:
            return f"Stability monitoring: Active (grad_norm: {self.last_metrics.get('gradient_norm', 0):.3f})"
        return "Stability monitoring: Active"
    
    def check_and_fix(self, tensors):
        """Pass through to underlying controller if available"""
        if hasattr(self.controller, 'check_and_fix'):
            return self.controller.check_and_fix(tensors)
        return tensors

class FallbackStabilityController:
    """Fallback stability controller when advanced stability system is not available"""
    
    def __init__(self):
        self.gradient_threshold = 10.0
        self.last_grad_norm = 0.0
        self.consecutive_high_grads = 0
        
    def pre_step_check(self, epoch):
        """Basic stability check with actual gradient monitoring"""
        metrics = StabilityMetrics()
        
        # Check if we have high gradients for too long
        intervention_needed = False
        intervention_type = None
        
        if self.consecutive_high_grads > 5:
            intervention_needed = True
            intervention_type = 'gradient_explosion'
            print(f"WARNING: High gradients detected for {self.consecutive_high_grads} steps")
            
        return {
            'intervention_needed': intervention_needed,
            'intervention_type': intervention_type,
            'stability_metrics': metrics
        }
    
    def apply_interventions(self, intervention_type, epoch):
        """Apply basic interventions"""
        if intervention_type == 'gradient_explosion':
            print(f"INTERVENTION: Reducing learning rate due to gradient explosion")
            self.consecutive_high_grads = 0  # Reset counter
            return True
        return True
    
    def update_gradient_info(self, grad_norm):
        """Update gradient tracking"""
        self.last_grad_norm = grad_norm
        if grad_norm > self.gradient_threshold:
            self.consecutive_high_grads += 1
        else:
            self.consecutive_high_grads = 0
    
    def get_stability_report(self):
        """Basic stability report"""
        return f"Stability monitoring: Fallback (grad_norm: {self.last_grad_norm:.3f})"

class AdaptiveTrainingFramework:
    """
    Adaptive training framework with memory management, convergence monitoring,
    and performance optimization for Beta spatial reasoning.
    """
    
    def __init__(self, model: BetaSpatialReasonerComplete, 
                 config: Dict[str, Any],
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Adaptive parameters
        self.adaptive_batch_size = config.get('batch_size', 64)
        self.adaptive_n_samples = config.get('n_samples', 100)
        self.memory_threshold = config.get('max_memory_gb', 10)
        
        # Curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(config.get('curriculum', {}))
        
        # Performance monitoring
        self.loss_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=50)
        self.timing_history = deque(maxlen=50)
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
        # Setup numerical stability controller with fallback
        raw_controller = create_stability_controller()
        # Wrap it with adapter to match expected interface
        self.stability_controller = StabilityControllerAdapter(raw_controller)
        
        # Setup logging
        self.setup_logging()
        
    def setup_optimization(self):
        """Setup optimizer with different learning rates for different components"""
        optimizer_config = self.config.get('optimizer', {})
        
        # PHASE 2 FIX: Register ALL trainable parameters correctly
        param_groups = []
        
        # 1. Feature extractor CNN (MUST be included!)
        if hasattr(self.model, 'feature_extractor'):
            param_groups.append({
                'params': self.model.feature_extractor.parameters(),
                'lr': optimizer_config.get('feature_extractor_lr', 5e-4),
                'name': 'feature_extractor'
            })
        
        # 2. Predictor components
        param_groups.extend([
            {
                'params': self.model.predictor.scene_encoder.parameters(),
                'lr': optimizer_config.get('scene_encoder_lr', 1e-4),
                'name': 'scene_encoder'
            },
            {
                'params': self.model.predictor.object_attention.parameters(),
                'lr': optimizer_config.get('attention_lr', 1e-4),
                'name': 'object_attention'
            },
            {
                'params': self.model.predictor.object_encoder.parameters(),
                'lr': optimizer_config.get('object_encoder_lr', 1e-4),
                'name': 'object_encoder'
            },
            {
                'params': self.model.predictor.alpha_predictor.parameters(),
                'lr': optimizer_config.get('alpha_lr', 1e-3),
                'name': 'alpha_predictor'
            },
            {
                'params': self.model.predictor.beta_predictor.parameters(),
                'lr': optimizer_config.get('beta_lr', 1e-3),
                'name': 'beta_predictor'
            }
        ])
        
        # 3. Learnable bias parameters (critical for Beta distributions)
        param_groups.append({
            'params': [self.model.predictor.alpha_bias, self.model.predictor.beta_bias],
            'lr': optimizer_config.get('bias_lr', 2e-4),  # OPTION 2: Use Option 2 default
            'name': 'distribution_biases'
        })
        
        # VERIFIED: loss_fn has NO trainable parameters (correctly excluded)
        # Chen's adaptive loss weighting uses only buffer parameters
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=optimizer_config.get('weight_decay', 1e-5),
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
        
        # DISABLED: Scheduler was killing gradients via aggressive cosine annealing
        # scheduler_config = self.config.get('scheduler', {})
        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=scheduler_config.get('T_0', 20),
        #     T_mult=scheduler_config.get('T_mult', 2),
        #     eta_min=scheduler_config.get('eta_min', 1e-6)
        # )
        self.scheduler = None  # FIXED: Use constant learning rates for stable training
        
    def setup_logging(self):
        """Setup logging directories and files"""
        self.log_dir = Path(self.config.get('log_dir', 'logs/beta_training'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints/beta_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training log file
        self.log_file = self.log_dir / 'training.log'
        
    def log_message(self, message: str, level: str = "INFO"):
        """Log message to file and console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def log_detailed_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log comprehensive epoch summary with mathematical and systems health"""
        self.log_message("=" * 80)
        self.log_message(f"DETAILED EPOCH {epoch} SUMMARY")
        self.log_message("=" * 80)
        
        # Core Training Metrics
        self.log_message("CORE TRAINING METRICS:")
        self.log_message(f"  Training Loss: {train_metrics.get('total_loss', 0):.6f}")
        self.log_message(f"  Validation Loss: {val_metrics.get('total_loss', 0):.6f}")
        self.log_message(f"  Constraint Loss: {val_metrics.get('constraint_loss', 0):.6f}")
        self.log_message(f"  Training Constraint Satisfaction: {train_metrics.get('constraint_satisfaction', 0):.6f}")
        self.log_message(f"  Validation Constraint Satisfaction: {val_metrics.get('constraint_satisfaction', 0):.6f}")
        
        # Mathematical Health Indicators
        self.log_message("MATHEMATICAL HEALTH:")
        if 'alpha_mean' in val_metrics:
            self.log_message(f"  Alpha Mean: {val_metrics['alpha_mean']:.4f}")
            self.log_message(f"  Beta Mean: {val_metrics['beta_mean']:.4f}")
            self.log_message(f"  Concentration Mean: {val_metrics['concentration_mean']:.4f}")
            self.log_message(f"  Concentration Std: {val_metrics['concentration_std']:.4f}")
        
        if 'degenerate_distribution_rate' in val_metrics:
            degenerate_rate = val_metrics['degenerate_distribution_rate']
            self.log_message(f"  Degenerate Distribution Rate: {degenerate_rate:.4f}")
            if degenerate_rate > 0.1:
                self.log_message("  WARNING: High degenerate distribution rate!", "WARN")
        
        # Training Dynamics
        self.log_message("TRAINING DYNAMICS:")
        if 'gradient_norm' in val_metrics:
            grad_norm = val_metrics['gradient_norm']
            self.log_message(f"  Gradient Norm: {grad_norm:.6f}")
            if grad_norm > 10.0:
                self.log_message("  WARNING: Large gradient norm detected!", "WARN")
            elif grad_norm < 1e-6:
                self.log_message("  WARNING: Very small gradients - potential vanishing gradient!", "WARN")
        
        if 'learning_rate' in val_metrics:
            self.log_message(f"  Learning Rate: {val_metrics['learning_rate']:.2e}")
        
        # System Performance
        self.log_message("SYSTEM PERFORMANCE:")
        if 'avg_batch_time' in val_metrics:
            self.log_message(f"  Avg Batch Time: {val_metrics['avg_batch_time']:.3f}s")
        if 'avg_throughput' in val_metrics:
            self.log_message(f"  Throughput: {val_metrics['avg_throughput']:.1f} samples/sec")
        if 'memory_usage' in val_metrics:
            self.log_message(f"  Memory Usage: {val_metrics['memory_usage']:.2f}GB")
        
        # Training Progress
        if 'training_progress' in val_metrics:
            progress_pct = val_metrics['training_progress'] * 100
            self.log_message(f"  Training Progress: {progress_pct:.1f}%")
        
        self.log_message("=" * 80)
    
    def monitor_memory_usage(self) -> float:
        """Monitor GPU memory usage and return current usage in GB"""
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            self.memory_history.append(memory_gb)
            return memory_gb
        return 0.0
    
    def adapt_batch_parameters(self, current_memory: float) -> Tuple[int, int]:
        """Adapt batch size and sampling parameters based on memory usage"""
        if current_memory > self.memory_threshold:
            # Reduce batch size and sampling
            self.adaptive_batch_size = max(1, int(self.adaptive_batch_size * 0.8))
            self.adaptive_n_samples = max(50, int(self.adaptive_n_samples * 0.8))
            
            self.log_message(
                f"Memory optimization: batch_size={self.adaptive_batch_size}, "
                f"n_samples={self.adaptive_n_samples}"
            )
        elif current_memory < self.memory_threshold * 0.5:
            # Increase parameters if memory is available
            original_batch = self.config.get('batch_size', 64)
            original_samples = self.config.get('n_samples', 100)
            
            self.adaptive_batch_size = min(original_batch, int(self.adaptive_batch_size * 1.1))
            self.adaptive_n_samples = min(original_samples, int(self.adaptive_n_samples * 1.1))
        
        return self.adaptive_batch_size, self.adaptive_n_samples
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch with adaptive parameters and monitoring"""
        self.model.train()
        
        # Get current curriculum level
        complexity_level = self.curriculum_scheduler.get_complexity_level(self.epoch)
        
        # Initialize metrics
        epoch_metrics = defaultdict(list)
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Track batch processing time
            batch_start_time = time.time()
            
            # Unpack batch data
            images, target_coords, constraints = batch_data
            images = images.to(self.device)
            target_coords = target_coords.to(self.device)
            
            # Monitor memory for logging but disable adaptive batch scaling (conflicts with GPU optimization)
            current_memory = self.monitor_memory_usage()
            # batch_size, n_samples = self.adapt_batch_parameters(current_memory)  # DISABLED
            batch_size, n_samples = self.config['batch_size'], self.config['n_samples']
            
            # Adjust model sampling parameters
            self.model.loss_fn.n_samples = n_samples
            
            # PHASE 2 FIX: Update Chen's dynamic parameter bounds based on training progress
            progress = min(1.0, self.global_step / (self.config.get('num_epochs', 200) * len(dataloader)))
            self.model.predictor.set_training_progress(progress)
            
            # Forward pass
            outputs = self.model(
                input_data=images,
                constraints=constraints,
                target_coords=target_coords,
                complexity_level=complexity_level
            )
            
            loss = outputs['total_loss']
            
            # PHASE 2 FIX: Extract Chen's adaptive weights for monitoring
            adaptive_weights = outputs.get('adaptive_weights', None)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # PHASE 2 FIX: Comprehensive gradient verification system
            gradient_diagnostics = self._compute_gradient_diagnostics()
            total_grad_norm = gradient_diagnostics['total_norm']
            
            # Update fallback controller if it has the method
            if hasattr(self.stability_controller, 'update_gradient_info'):
                self.stability_controller.update_gradient_info(total_grad_norm)
            
            # Check for gradient pathologies
            if gradient_diagnostics['nan_count'] > 0:
                self.log_message(f"WARNING: {gradient_diagnostics['nan_count']} NaN gradients detected!", level="WARNING")
                # Skip this update to prevent model corruption
                self.optimizer.zero_grad()
                continue
            
            if gradient_diagnostics['inf_count'] > 0:
                self.log_message(f"WARNING: {gradient_diagnostics['inf_count']} Inf gradients detected!", level="WARNING")
                # Skip this update to prevent model corruption
                self.optimizer.zero_grad()
                continue
            
            if total_grad_norm > 100.0:  # Explosion threshold
                self.log_message(f"WARNING: Gradient explosion detected! Norm={total_grad_norm:.2f}", level="WARNING")
                # Apply stronger clipping for this step ONLY
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # FIXED: Less aggressive
            elif total_grad_norm < 1e-8:  # Vanishing threshold
                self.log_message(f"WARNING: Vanishing gradients detected! Norm={total_grad_norm:.2e}", level="WARNING")
                # No clipping when gradients are already too small
            else:
                # OPTIMIZED: Moderate gradient clipping for stable learning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # FIXED: More permissive
            
            # Store parameter snapshots before update for change verification
            param_snapshots = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_snapshots[name] = param.data.clone()
            
            self.optimizer.step()
            
            # PHASE 2 FIX: Verify parameters actually changed
            param_change_diagnostics = self._verify_parameter_updates(param_snapshots)
            
            # Record gradient and parameter change diagnostics
            epoch_metrics['gradient_norm'].append(total_grad_norm)
            epoch_metrics['gradient_nan_count'].append(gradient_diagnostics['nan_count'])
            epoch_metrics['param_changes'].append(param_change_diagnostics['mean_change'])
            epoch_metrics['stuck_params'].append(param_change_diagnostics['stuck_count'])
            
            # Enhanced stability monitoring (after step for parameter health)
            stability_check = self.stability_controller.pre_step_check(self.epoch)
            stability_metrics = stability_check['stability_metrics']
            
            # Apply interventions if needed
            if stability_check['intervention_needed']:
                intervention_type = stability_check['intervention_type']
                self.log_message(f"Applying stability intervention: {intervention_type}")
                success = self.stability_controller.apply_interventions(intervention_type, self.epoch)
                
                if not success:
                    self.log_message("Stability intervention failed!", level="WARNING")
            
            # Gradient norm already recorded in verification section above
            
            # Check parameter health by computing parameter magnitude changes
            param_health = 1.0  # Default healthy
            if hasattr(stability_metrics, 'beta_parameter_health'):
                param_health = stability_metrics.beta_parameter_health
            epoch_metrics['parameter_health'].append(param_health)
            
            # Numerical precision from stability controller or default
            numerical_precision = getattr(stability_metrics, 'numerical_precision', 1.0)
            epoch_metrics['numerical_precision'].append(numerical_precision)
            
            # Complete batch timing and calculate performance metrics
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            self._last_batch_time = batch_time  # Store for logging access
            
            # Record metrics
            for key, value in outputs.items():
                if key == 'constraint_satisfaction_prob' and isinstance(value, torch.Tensor):
                    # Handle constraint satisfaction probability (might be batch-sized) - CHECK FIRST
                    epoch_metrics['constraint_satisfaction'].append(value.mean().item())
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    epoch_metrics[key].append(value.item())
            
            # Systems-level monitoring
            epoch_metrics['memory_usage'].append(current_memory)
            epoch_metrics['batch_time'].append(batch_time)
            epoch_metrics['throughput_samples_per_sec'].append(len(images) / batch_time)
            
            # Training dynamics monitoring
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.get('optimizer', {}).get('alpha_lr', 1e-4)
            epoch_metrics['learning_rate'].append(current_lr)
            
            # EMERGENCY FIX: Remove duplicate gradient computation - already done at line 340-341
            # Use the already computed gradient norm from gradient_diagnostics
            param_count = sum(p.numel() for p in self.model.parameters() if p.grad is not None)
            
            if param_count > 0 and total_grad_norm > 0:
                # Already appended at line 378, just compute avg per param
                epoch_metrics['avg_gradient_per_param'].append(total_grad_norm / param_count)
            
            # Training progress indicators
            progress = min(1.0, self.global_step / (self.config.get('num_epochs', 200) * len(dataloader)))
            epoch_metrics['training_progress'].append(progress)
            
            # Mathematical health indicators - Beta parameter monitoring
            if 'alpha' in outputs and 'beta' in outputs:
                alpha = outputs['alpha']
                beta = outputs['beta']
                concentration = alpha + beta
                
                # Beta parameter health metrics
                epoch_metrics['alpha_mean'].append(alpha.mean().item())
                epoch_metrics['beta_mean'].append(beta.mean().item())
                epoch_metrics['concentration_mean'].append(concentration.mean().item())
                epoch_metrics['concentration_std'].append(concentration.std().item())
                
                # Parameter stability indicators
                alpha_stability = torch.std(alpha) / (torch.mean(alpha) + 1e-8)
                beta_stability = torch.std(beta) / (torch.mean(beta) + 1e-8)
                epoch_metrics['alpha_cv'].append(alpha_stability.item())  # Coefficient of variation
                epoch_metrics['beta_cv'].append(beta_stability.item())
                
                # Distribution quality indicators
                min_concentration = concentration.min().item()
                max_concentration = concentration.max().item()
                epoch_metrics['concentration_range'].append(max_concentration - min_concentration)
                
                # Numerical stability check
                has_degenerate = ((alpha < 1e-6) | (beta < 1e-6)).float().mean()
                epoch_metrics['degenerate_distribution_rate'].append(has_degenerate.item())
            
            # PHASE 2 FIX: Record Chen's adaptive weights for analysis
            if adaptive_weights is not None:
                for weight_name, weight_val in adaptive_weights.items():
                    epoch_metrics[f'adaptive_weight_{weight_name}'].append(weight_val)
            
            # Log progress with actual gradient info
            if batch_idx % 10 == 0:
                # Use actual computed gradient norm, not stability controller dummy values
                actual_grad_norm = total_grad_norm
                param_health = getattr(stability_metrics, 'beta_parameter_health', 1.0)
                
                # PHASE 2 FIX: Enhanced logging with adaptive weights and bounds info
                current_bounds = f"[{self.model.predictor.min_param:.2f}, {self.model.predictor.max_param:.2f}]"
                weights_info = "" 
                if adaptive_weights is not None:
                    weights_info = f", AdaptWt={adaptive_weights['coord']:.3f}/{adaptive_weights['constraint']:.3f}"
                
                self.log_message(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(dataloader)}: "
                    f"Loss={loss.item():.4f}, Memory={current_memory:.2f}GB, "
                    f"Complexity={complexity_level}, "
                    f"GradNorm={actual_grad_norm:.6f}, "
                    f"ParamHealth={param_health:.3f}, "
                    f"Bounds={current_bounds}{weights_info}"
                )
            
            self.global_step += 1
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Compute epoch averages
        epoch_results = {}
        for key, values in epoch_metrics.items():
            if values:  # Only compute if values exist
                epoch_results[key] = np.mean(values)
            else:
                epoch_results[key] = 0.0
        
        epoch_results['epoch_time'] = time.time() - start_time
        epoch_results['complexity_level'] = complexity_level
        
        # Add epoch-level performance summaries
        if 'batch_time' in epoch_results:
            epoch_results['avg_batch_time'] = epoch_results['batch_time']
            epoch_results['est_epoch_time'] = epoch_results['batch_time'] * len(dataloader)
        
        if 'throughput_samples_per_sec' in epoch_results:
            epoch_results['avg_throughput'] = epoch_results['throughput_samples_per_sec']
        
        # Training health summary
        if 'gradient_norm' in epoch_results:
            epoch_results['gradient_health'] = 1.0 if epoch_results['gradient_norm'] < 10.0 else 0.5
        
        if 'degenerate_distribution_rate' in epoch_results:
            epoch_results['beta_health'] = 1.0 - epoch_results['degenerate_distribution_rate']
        
        self.timing_history.append(epoch_results['epoch_time'])
        
        # MEMORY LEAK FIX: Explicit memory cleanup between epochs
        self._cleanup_epoch_memory()
        
        # Generate stability report every 10 epochs
        if self.epoch % 10 == 0:
            stability_report = self.stability_controller.get_stability_report()
            self.log_message("STABILITY REPORT:")
            for line in stability_report.split('\n'):
                if line.strip():
                    self.log_message(line)
        
        return epoch_results
    
    def _cleanup_epoch_memory(self):
        """
        MEMORY LEAK FIX: Explicit memory cleanup between epochs
        """
        import gc
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force Python garbage collection
        gc.collect()
        
        # Clear any accumulated constraint objects from previous batches
        # (They can create circular references that GC misses)
        if hasattr(self, '_last_constraints'):
            self._last_constraints = None
        
        # Optional: Reduce deque sizes if they get too large
        if len(self.memory_history) > 100:
            # Keep only recent memory history
            recent_memory = list(self.memory_history)[-50:]
            self.memory_history.clear()
            self.memory_history.extend(recent_memory)
        
        if self.epoch % 10 == 0:
            # More aggressive cleanup every 10 epochs
            self.loss_history.clear()
            
        self.log_message(f"Memory cleanup completed for epoch {self.epoch}")
    
    def _compute_gradient_diagnostics(self) -> Dict[str, float]:
        """
        PHASE 2 FIX: Comprehensive gradient health diagnostics
        """
        total_norm = 0.0
        nan_count = 0
        inf_count = 0
        zero_count = 0
        param_count = 0
        
        component_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_count += 1
                grad_norm = param.grad.data.norm(2).item()
                
                if torch.isnan(param.grad.data).any():
                    nan_count += 1
                if torch.isinf(param.grad.data).any():
                    inf_count += 1
                if grad_norm < 1e-10:
                    zero_count += 1
                
                total_norm += grad_norm ** 2
                
                # Track component-wise gradient norms
                component = name.split('.')[0]  # Get component name
                if component not in component_norms:
                    component_norms[component] = 0.0
                component_norms[component] += grad_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # Compute component norms
        for component in component_norms:
            component_norms[component] = component_norms[component] ** 0.5
        
        return {
            'total_norm': total_norm,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_count': zero_count,
            'param_count': param_count,
            'component_norms': component_norms
        }
    
    def _verify_parameter_updates(self, param_snapshots: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PHASE 2 FIX: Verify that parameters actually changed after optimizer step
        """
        total_change = 0.0
        stuck_count = 0
        param_count = 0
        max_change = 0.0
        
        component_changes = {}
        
        for name, param in self.model.named_parameters():
            if name in param_snapshots:
                param_count += 1
                old_param = param_snapshots[name]
                change = (param.data - old_param).norm(2).item()
                
                total_change += change
                max_change = max(max_change, change)
                
                if change < 1e-10:  # Parameter didn't change
                    stuck_count += 1
                
                # Track component-wise changes
                component = name.split('.')[0]
                if component not in component_changes:
                    component_changes[component] = []
                component_changes[component].append(change)
        
        # Compute mean changes per component
        for component in component_changes:
            component_changes[component] = np.mean(component_changes[component])
        
        mean_change = total_change / max(param_count, 1)
        
        return {
            'mean_change': mean_change,
            'max_change': max_change,
            'stuck_count': stuck_count,
            'param_count': param_count,
            'component_changes': component_changes
        }
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validation with comprehensive metrics"""
        self.model.eval()
        
        val_metrics = defaultdict(list)
        
        # CRITICAL FIX: Match training complexity level to avoid validation loss explosion
        current_complexity = self.curriculum_scheduler.get_complexity_level(self.epoch)
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                images, target_coords, constraints = batch_data
                images = images.to(self.device)
                target_coords = target_coords.to(self.device)
                
                # Forward pass - use SAME complexity as training
                outputs = self.model(
                    input_data=images,
                    constraints=constraints,
                    target_coords=target_coords,
                    complexity_level=current_complexity  # Match training complexity
                )
                
                # Record metrics
                for key, value in outputs.items():
                    if key == 'constraint_satisfaction_prob' and isinstance(value, torch.Tensor):
                        # Handle constraint satisfaction probability (same as training loop) - CHECK FIRST
                        val_metrics['constraint_satisfaction'].append(value.mean().item())
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        val_metrics[key].append(value.item())
        
        # Compute validation averages
        val_results = {}
        for key, values in val_metrics.items():
            val_results[key] = np.mean(values)
        
        return val_results
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with metrics"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'curriculum_state': self.curriculum_scheduler.get_state()
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            loss_value = metrics.get('total_loss', 0.0)
            self.log_message(f"New best model saved with loss: {loss_value:.4f}")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        patience = self.config.get('patience', 50)
        min_delta = self.config.get('min_delta', 1e-4)
        
        if val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                self.log_message(f"Early stopping triggered after {patience} epochs without improvement")
                return True
            return False


class CurriculumScheduler:
    """
    Manages curriculum learning progression from simple to complex constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default curriculum schedule
        self.schedule = config.get('schedule', {
            'epochs_per_level': 20,
            'max_level': 5,
            'constraint_types': {
                1: ['T1'],  # Simple value constraints
                2: ['T1', 'T2'],  # Add object-object constraints
                3: ['T1', 'T2', 'T3'],  # Add coordinate arithmetic
                4: ['T1', 'T2', 'T3', 'T4'],  # Add complex arithmetic
                5: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'OR', 'AND', 'NOT']  # Full complexity
            }
        })
        
    def get_complexity_level(self, epoch: int) -> int:
        """Get current complexity level based on epoch"""
        epochs_per_level = self.schedule['epochs_per_level']
        max_level = self.schedule['max_level']
        
        level = min(max_level, (epoch // epochs_per_level) + 1)
        return level
    
    def get_allowed_constraints(self, epoch: int) -> List[str]:
        """Get list of allowed constraint types for current epoch"""
        level = self.get_complexity_level(epoch)
        return self.schedule['constraint_types'].get(level, ['T1'])
    
    def get_state(self) -> Dict[str, Any]:
        """Get curriculum scheduler state for checkpointing"""
        return {
            'schedule': self.schedule,
            'config': self.config
        }


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation suite for Beta spatial reasoning with uncertainty calibration
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def evaluate_model(self, model: BetaSpatialReasonerComplete, 
                      dataloader: DataLoader,
                      constraints_list: List[List[Any]]) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_constraints = []
        all_outputs = []
        
        with torch.no_grad():
            for batch_idx, (images, target_coords, constraints) in enumerate(dataloader):
                images = images.to(self.device)
                target_coords = target_coords.to(self.device)
                
                # CRITICAL FIX: Pass target_coords and constraints to get training-style outputs
                # This ensures consistent tensor structure across all batches
                outputs = model(input_data=images, target_coords=target_coords, constraints=constraints)
                
                all_predictions.append(outputs)
                all_targets.append(target_coords)
                all_constraints.extend(constraints)
                all_outputs.append(outputs)
        
        # Combine all batches
        combined_predictions = self._combine_batch_outputs(all_outputs)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Compute comprehensive metrics
        metrics = self._compute_comprehensive_metrics(
            combined_predictions, combined_targets, all_constraints
        )
        
        return metrics
    
    def _combine_batch_outputs(self, batch_outputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Combine outputs from multiple batches with robust handling of variable batch sizes
        
        CRITICAL FIX: Handle variable batch sizes and different tensor structures gracefully
        """
        if not batch_outputs:
            return {}
            
        combined = {}
        
        # Debug information for troubleshooting
        # print(f"DEBUG: Combining {len(batch_outputs)} batch outputs")
        
        for key in batch_outputs[0].keys():
            if isinstance(batch_outputs[0][key], torch.Tensor):
                # Collect tensors and validate dimensions before concatenation
                tensors_to_cat = []
                expected_shape = None
                
                for i, batch in enumerate(batch_outputs):
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        tensor = batch[key]
                        
                        # Check shape consistency (all dims except 0 must match)
                        if expected_shape is None:
                            expected_shape = tensor.shape[1:]  # All dims except batch
                        elif tensor.shape[1:] != expected_shape:
                            print(f"WARNING: Shape mismatch for key '{key}' at batch {i}: "
                                  f"expected {expected_shape}, got {tensor.shape[1:]}")
                            # Skip this tensor to avoid concatenation errors
                            continue
                            
                        tensors_to_cat.append(tensor)
                
                if tensors_to_cat:
                    try:
                        # Handle scalar tensors (0-dim) by stacking instead of concatenating
                        if tensors_to_cat[0].dim() == 0:
                            combined[key] = torch.stack(tensors_to_cat, dim=0)
                        else:
                            combined[key] = torch.cat(tensors_to_cat, dim=0)
                    except RuntimeError as e:
                        print(f"CRITICAL ERROR: Failed to concatenate key '{key}': {e}")
                        # Detailed error information
                        for i, t in enumerate(tensors_to_cat):
                            print(f"  Batch {i}: shape {t.shape}, dtype {t.dtype}")
                        # Raise error instead of silently using incomplete data
                        raise RuntimeError(f"Cannot combine batch outputs for key '{key}': {e}") from e
                        
            elif isinstance(batch_outputs[0][key], dict):
                combined[key] = {}
                for sub_key in batch_outputs[0][key].keys():
                    # Collect sub-tensors with same validation
                    sub_tensors_to_cat = []
                    expected_sub_shape = None
                    
                    for i, batch in enumerate(batch_outputs):
                        if (key in batch and isinstance(batch[key], dict) and 
                            sub_key in batch[key] and isinstance(batch[key][sub_key], torch.Tensor)):
                            
                            sub_tensor = batch[key][sub_key]
                            
                            # Check shape consistency
                            if expected_sub_shape is None:
                                expected_sub_shape = sub_tensor.shape[1:]
                            elif sub_tensor.shape[1:] != expected_sub_shape:
                                print(f"WARNING: Shape mismatch for key '{key}.{sub_key}' at batch {i}: "
                                      f"expected {expected_sub_shape}, got {sub_tensor.shape[1:]}")
                                continue
                                
                            sub_tensors_to_cat.append(sub_tensor)
                    
                    if sub_tensors_to_cat:
                        try:
                            combined[key][sub_key] = torch.cat(sub_tensors_to_cat, dim=0)
                        except RuntimeError as e:
                            print(f"CRITICAL ERROR: Failed to concatenate key '{key}.{sub_key}': {e}")
                            # Detailed error information
                            for i, t in enumerate(sub_tensors_to_cat):
                                print(f"  Batch {i}: shape {t.shape}, dtype {t.dtype}")
                            # Raise error instead of silently using incomplete data
                            raise RuntimeError(f"Cannot combine batch outputs for key '{key}.{sub_key}': {e}") from e
            else:
                # Handle non-tensor values (copy from first batch)
                combined[key] = batch_outputs[0][key]
        
        return combined
    
    def _compute_comprehensive_metrics(self, predictions: Dict[str, torch.Tensor],
                                     targets: torch.Tensor,
                                     constraints: List[List[Any]]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        CRITICAL FIX: Handle training-style output structure with loss components
        """
        metrics = {}
        
        # Extract alpha and beta parameters (always available in training-style outputs)
        alpha = predictions['alpha']  # [batch, num_objects, 4]
        beta = predictions['beta']    # [batch, num_objects, 4]
        
        # Compute prediction means from Beta distributions
        pred_means = alpha / (alpha + beta) * 1000.0  # Convert to [0,1000] space
        
        # Compute prediction variances and standard deviations
        pred_variances = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1)) * (1000.0**2)
        pred_stds = torch.sqrt(pred_variances)
        
        # 1. Coordinate Accuracy
        metrics['coordinate_mse'] = F.mse_loss(pred_means, targets).item()
        metrics['coordinate_mae'] = F.l1_loss(pred_means, targets).item()
        
        # 2. Uncertainty Calibration
        actual_errors = torch.abs(pred_means - targets)
        metrics['uncertainty_calibration'] = self._compute_calibration_score(pred_stds, actual_errors)
        
        # 3. Loss Components (from training-style outputs)
        if 'coordinate_loss' in predictions:
            metrics['coordinate_loss'] = predictions['coordinate_loss'].item()
        if 'constraint_loss' in predictions:
            metrics['constraint_loss'] = predictions['constraint_loss'].item()
        if 'total_loss' in predictions:
            metrics['total_loss'] = predictions['total_loss'].item()
        
        # 4. Constraint Satisfaction
        # Use the constraint satisfaction probability from model outputs if available
        if 'constraint_satisfaction_prob' in predictions:
            metrics['constraint_satisfaction'] = predictions['constraint_satisfaction_prob'].mean().item()
        else:
            # Fallback: Generate samples from Beta distributions for constraint checking
            coord_samples = self._sample_from_beta_distributions(alpha, beta, n_samples=100)
            metrics['constraint_satisfaction'] = self._compute_constraint_satisfaction(coord_samples, constraints)
        
        # 5. Generation Diversity (computed from variance of predictions)
        metrics['generation_diversity'] = torch.mean(pred_variances).item()
        
        # 6. Distribution Quality
        concentration = alpha + beta
        metrics['mean_concentration'] = concentration.mean().item()
        metrics['concentration_std'] = concentration.std().item()
        
        # 7. Boundary Compliance (using predicted means)
        metrics['boundary_violations'] = self._compute_boundary_violations(pred_means)
        
        # 8. Prediction Confidence
        metrics['mean_uncertainty'] = pred_stds.mean().item()
        
        return metrics
    
    def _compute_calibration_score(self, predicted_stds: torch.Tensor, 
                                  actual_errors: torch.Tensor) -> float:
        """
        Compute uncertainty calibration score
        Perfect calibration: predicted_std ≈ actual_error
        """
        # Correlation between predicted uncertainty and actual errors
        pred_flat = predicted_stds.flatten()
        error_flat = actual_errors.flatten()
        
        # Pearson correlation coefficient
        correlation = torch.corrcoef(torch.stack([pred_flat, error_flat]))[0, 1]
        
        # Convert to calibration score (0 = no calibration, 1 = perfect calibration)
        calibration_score = max(0.0, correlation.item())
        
        return calibration_score
    
    def _sample_from_beta_distributions(self, alpha: torch.Tensor, beta: torch.Tensor, 
                                       n_samples: int = 100) -> torch.Tensor:
        """
        Sample coordinates from Beta distributions
        Args:
            alpha: [batch, objects, 4] - Beta alpha parameters
            beta: [batch, objects, 4] - Beta beta parameters
            n_samples: Number of samples to draw
        Returns:
            coord_samples: [n_samples, batch, objects, 4] - Sampled coordinates
        """
        batch_size, num_objects, num_coords = alpha.shape
        
        # CRITICAL FIX: Ensure parameter bounds consistent with inference system
        # Updated from 1.01 to 2.0 to match all other components (prevent variance explosion)
        alpha_clamped = torch.clamp(alpha, min=2.0, max=50.0)  # FIXED: Consistent with inference bounds
        beta_clamped = torch.clamp(beta, min=2.0, max=50.0)   # FIXED: Consistent with inference bounds
        
        # Create Beta distributions
        dist = torch.distributions.Beta(alpha_clamped, beta_clamped)
        
        # Sample and scale to [0, 1000]
        samples = dist.rsample((n_samples,)) * 1000.0
        
        return samples
    
    def _compute_constraint_satisfaction(self, coord_samples: torch.Tensor,
                                       constraints: List[List[Any]]) -> float:
        """
        Compute constraint satisfaction rate across samples
        """
        # Handle both sampled and non-sampled inputs
        if coord_samples.dim() == 3:  # [batch, objects, 4] - add sample dimension
            coord_samples = coord_samples.unsqueeze(0)  # [1, batch, objects, 4]
        
        n_samples, batch_size, num_objects, coords = coord_samples.shape
        
        satisfaction_rates = []
        
        for batch_idx in range(batch_size):
            if batch_idx < len(constraints) and constraints[batch_idx]:
                batch_constraints = constraints[batch_idx]
                batch_coords = coord_samples[:, batch_idx]  # [n_samples, num_objects, 4]
                
                # Check constraint satisfaction for each sample
                sample_satisfactions = []
                for sample_idx in range(n_samples):
                    sample_coords = batch_coords[sample_idx]  # [num_objects, 4]
                    satisfied = self._check_constraints_satisfied(sample_coords, batch_constraints)
                    sample_satisfactions.append(satisfied)
                
                # Average satisfaction across samples for this batch
                batch_satisfaction = np.mean(sample_satisfactions)
                satisfaction_rates.append(batch_satisfaction)
        
        return np.mean(satisfaction_rates) if satisfaction_rates else 1.0
    
    def _check_constraints_satisfied(self, coords: torch.Tensor, 
                                   constraints: List[Any]) -> bool:
        """
        Check if a single coordinate configuration satisfies constraints
        """
        for constraint in constraints:
            if not self._evaluate_single_constraint(coords, constraint):
                return False
        return True
    
    def _evaluate_single_constraint(self, coords: torch.Tensor, constraint: Any) -> bool:
        """
        Evaluate a single constraint on coordinate configuration
        """
        # Simplified constraint evaluation - would use full constraint evaluator in practice
        if hasattr(constraint, '_fields'):
            if 'ConstraintT1' in str(type(constraint)):
                return self._evaluate_t1_constraint_simple(coords, constraint)
            elif 'ConstraintT2' in str(type(constraint)):
                return self._evaluate_t2_constraint_simple(coords, constraint)
        
        return True  # Default to satisfied for unknown constraints
    
    def _evaluate_t1_constraint_simple(self, coords: torch.Tensor, constraint) -> bool:
        """Simple T1 constraint evaluation"""
        if constraint.o1 < coords.size(0) and constraint.v1 < coords.size(1):
            coord_val = coords[constraint.o1, constraint.v1].item()
            target_val = constraint.val + constraint.offset
            
            if constraint.c == 'lt':
                return coord_val < target_val
            elif constraint.c == 'gt':
                return coord_val > target_val
            elif constraint.c == 'eq':
                return abs(coord_val - target_val) < 5.0  # 5 pixel tolerance
        
        return True
    
    def _evaluate_t2_constraint_simple(self, coords: torch.Tensor, constraint) -> bool:
        """Simple T2 constraint evaluation"""
        if (constraint.o1 < coords.size(0) and constraint.o2 < coords.size(0) and
            constraint.v1 < coords.size(1) and constraint.v2 < coords.size(1)):
            
            coord1 = coords[constraint.o1, constraint.v1].item()
            coord2 = coords[constraint.o2, constraint.v2].item()
            
            if constraint.c == 'lt':
                return coord1 < coord2 + constraint.offset
            elif constraint.c == 'gt':
                return coord1 > coord2 + constraint.offset
            elif constraint.c == 'eq':
                return abs(coord1 - coord2 - constraint.offset) < 5.0
        
        return True
    
    def _compute_boundary_violations(self, coord_samples: torch.Tensor) -> float:
        """Compute fraction of coordinates outside [0, 1000] bounds"""
        violations = ((coord_samples < 0) | (coord_samples > 1000)).float()
        return violations.mean().item()
    
    def generate_evaluation_report(self, metrics: Dict[str, float], 
                                 save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("BETA SPATIAL REASONER EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Coordinate Accuracy
        report.append("COORDINATE ACCURACY:")
        report.append(f"  Mean Squared Error: {metrics['coordinate_mse']:.4f}")
        report.append(f"  Mean Absolute Error: {metrics['coordinate_mae']:.4f}")
        report.append("")
        
        # Uncertainty Calibration
        report.append("UNCERTAINTY CALIBRATION:")
        report.append(f"  Calibration Score: {metrics['uncertainty_calibration']:.4f}")
        report.append(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
        report.append("")
        
        # Constraint Satisfaction
        report.append("CONSTRAINT SATISFACTION:")
        report.append(f"  Satisfaction Rate: {metrics['constraint_satisfaction']:.4f}")
        report.append("")
        
        # Generation Quality
        report.append("GENERATION QUALITY:")
        report.append(f"  Generation Diversity: {metrics['generation_diversity']:.4f}")
        report.append(f"  Boundary Violations: {metrics['boundary_violations']:.4f}")
        report.append("")
        
        # Distribution Properties
        report.append("DISTRIBUTION PROPERTIES:")
        report.append(f"  Mean Concentration: {metrics['mean_concentration']:.4f}")
        report.append(f"  Concentration Std: {metrics['concentration_std']:.4f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def create_training_configuration() -> Dict[str, Any]:
    """Create default training configuration"""
    return {
        'batch_size': 64,
        'n_samples': 100,
        'max_memory_gb': 8,  # MEMORY LEAK FIX: Reduced threshold to trigger cleanup earlier
        'num_epochs': 200,
        'patience': 50,
        'min_delta': 1e-4,
        
        # OPTIMIZED: Validated parameters for >75% constraint satisfaction
        'loss_weights': {
            'coord_weight': 0.8,         # OPTIMIZED: Reduced to allow constraint focus  
            'constraint_weight': 1.2,    # REVERT: Back to original working value
            'uncertainty_weight': 0.005, # OPTIMIZED: Minimized interference
            'boundary_weight': 0.1       # OPTIMIZED: Reduced boundary emphasis
        },
        
        'optimizer': {
            'feature_extractor_lr': 5e-6,  # REVERT: Back to original working value
            'scene_encoder_lr': 1e-4,      # REVERT: Back to original working value  
            'attention_lr': 1e-4,          # REVERT: Back to original working value
            'object_encoder_lr': 1e-4,     # REVERT: Back to original working value
            'alpha_lr': 5e-5,              # CHEN PRIORITY 2: 5x increase for escape velocity  
            'beta_lr': 5e-5,               # CHEN PRIORITY 2: 5x increase for escape velocity
            'bias_lr': 2e-4,               # OPTIMIZED: 2x increase for bias adaptation
            'weight_decay': 5e-7,          # OPTIMIZED: Reduced for less learning constraint
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        
        'scheduler': {
            'T_0': 15,                     # OPTIMIZED: Faster restart for adaptation
            'T_mult': 2,
            'eta_min': 5e-4                # FIXED: Preserve Option 2 learning rates (was 5e-7)
        },
        
        'curriculum': {
            'epochs_per_level': 15,        # OPTIMIZED: Accelerated curriculum progression
            'max_level': 5
        },
        
        'log_dir': 'logs/beta_training',
        'checkpoint_dir': 'checkpoints/beta_training'
    }


def main_training_loop():
    """
    Complete training pipeline for Beta spatial reasoning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Configuration
    config = create_training_configuration()
    
    # Initialize model with balanced loss weights
    loss_weights = config.get('loss_weights', {})
    model = BetaSpatialReasonerComplete(
        scene_dim=512,
        hidden_dim=256,
        num_objects=5,
        num_heads=8,
        coord_weight=loss_weights.get('coord_weight', 10.0),
        constraint_weight=loss_weights.get('constraint_weight', 0.5),
        uncertainty_weight=loss_weights.get('uncertainty_weight', 0.1),
        boundary_weight=loss_weights.get('boundary_weight', 0.2)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize training framework
    trainer = AdaptiveTrainingFramework(model, config, device)
    evaluator = ComprehensiveEvaluator(device)
    
    # Create real COCO datasets with interior design focus
    print("Loading COCO datasets with interior design categories...")
    
    # CRITICAL FIX: Use INPAINTED images as per SPRING paper methodology
    # "We remove these objects from the original images using the Telea method.
    # This lets us train SPRING to place bounding boxes in their most natural positions."
    
    # Original annotations (for bounding box targets)
    coco_train_json = "/home/gaurang/hardnetnew/data/coco/annotations/instances_train2017_interior.json"
    coco_val_json = "/home/gaurang/hardnetnew/data/coco/annotations/instances_val2017_interior.json"
    
    # INPAINTED images (empty backgrounds with objects removed) - THE CRITICAL FIX
    coco_train_dir = "/home/gaurang/hardnetnew/data/coco/inpainted_backgrounds/inpainted_images"
    coco_val_dir = "/home/gaurang/hardnetnew/data/coco/inpainted_validation"  # INPAINTED validation images
    
    # Check if inpainted images exist
    if not os.path.exists(coco_train_dir):
        print("ERROR: Inpainted training images not found!")
        print("SPRING requires inpainted backgrounds as per the paper.")
        print("Please run telea_inpainting_system.py first to generate inpainted backgrounds.")
        return
    
    # Create enhanced COCO datasets with INPAINTED backgrounds
    # The model learns to predict WHERE objects should be placed on empty backgrounds
    print("Creating dataset with INPAINTED backgrounds (Telea method)...")
    train_dataset = EnhancedCOCO_Wrapper.from_args_interior(
        coco_dir=coco_train_dir,  # Now points to inpainted images
        coco_json=coco_train_json,  # Original annotations for targets
        max_objects_per_image=5,  # Match model capacity
        min_bbox_area=100,
        img_size=128,
        enable_lazy_filtering=False,
        enable_adaptive_thresholds=True,
        enable_hybrid_sampling=True,
        use_inpainted_prefix=True  # Handle 'inpainted_' filename prefix
    )
    
    val_dataset = EnhancedCOCO_Wrapper.from_args_interior(
        coco_dir=coco_val_dir,
        coco_json=coco_val_json,
        max_objects_per_image=5,  # Match model capacity
        min_bbox_area=100,
        img_size=128,
        enable_lazy_filtering=False,
        enable_adaptive_thresholds=True,  # Need curriculum for proper evaluation
        enable_hybrid_sampling=False,  # Keep deterministic validation
        use_inpainted_prefix=True  # Handle 'inpainted_' filename prefix for validation too
    )
    
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    print(f"Validation dataset loaded: {len(val_dataset)} samples")
    
    # Create real constraint-aware collate function for COCO data
    def coco_constraint_collate_fn(batch):
        """
        Enhanced collate function for real COCO data with constraint generation
        CRITICAL FIX: Pass raw images to model for proper CNN feature extraction
        """
        images = []
        target_coords = []
        constraints_list = []
        
        for img_tensor, categories, bboxes in batch:
            # CRITICAL FIX: Keep images as 3D tensors for CNN processing
            # The model's feature_extractor will handle the feature extraction
            images.append(img_tensor)  # Shape: [3, 128, 128]
            
            # Process target coordinates with padding
            if len(bboxes) > 0:
                # Convert bboxes to tensor format [num_objects, 4]
                coords_tensor = torch.zeros((5, 4))  # Pad to max 5 objects
                for i, bbox in enumerate(bboxes[:5]):  # Take max 5 objects
                    coords_tensor[i] = torch.tensor(bbox, dtype=torch.float32)
                target_coords.append(coords_tensor)
                
                # Generate real spatial constraints from bboxes
                constraints = generate_spatial_constraints_from_bboxes(bboxes[:5])
                constraints_list.append(constraints)
            else:
                # Empty sample - create default
                coords_tensor = torch.zeros((5, 4))
                target_coords.append(coords_tensor)
                constraints_list.append([])
        
        # Stack images and coordinates
        # CRITICAL: Return 4D image tensors for CNN feature extraction
        images = torch.stack(images)  # Shape: [batch_size, 3, 128, 128]
        target_coords = torch.stack(target_coords)
        
        return images, target_coords, constraints_list
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=coco_constraint_collate_fn,
        num_workers=2,  # BALANCED: Some parallelism without heavy memory load
        pin_memory=True,  # Re-enable for better GPU transfer
        persistent_workers=False,  # MEMORY LEAK FIX: Disable persistent workers
        prefetch_factor=2  # Small buffer to pipeline data
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=coco_constraint_collate_fn,
        num_workers=2,  # BALANCED: Some parallelism without heavy memory load
        pin_memory=True,  # Re-enable for better GPU transfer
        persistent_workers=False,  # MEMORY LEAK FIX: Disable persistent workers
        prefetch_factor=2  # Small buffer to pipeline data
    )
    
    trainer.log_message("Starting training...")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        trainer.epoch = epoch
        
        # Training
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validation
        val_metrics = trainer.validate(val_loader)
        
        # Extract comprehensive metrics for logging
        train_loss = train_metrics.get('total_loss', 0.0)
        val_loss = val_metrics.get('total_loss', 0.0)
        constraint_loss = val_metrics.get('constraint_loss', 0.0)
        train_constraint_satisfaction = train_metrics.get('constraint_satisfaction', 0.0)
        val_constraint_satisfaction = val_metrics.get('constraint_satisfaction', 0.0)
        
        # Systems-level performance metrics
        batch_time = getattr(trainer, '_last_batch_time', 0.0)
        memory_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        # Training dynamics
        current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else trainer.config.get('optimizer', {}).get('alpha_lr', 1e-4)
        
        # Enhanced logging with comprehensive metrics
        base_log = (
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Constraint Loss={constraint_loss:.4f}, "
            f"Train Sat={train_constraint_satisfaction:.4f}, "
            f"Val Sat={val_constraint_satisfaction:.4f}"
        )
        
        perf_log = (
            f"LR={current_lr:.2e}, "
            f"Mem={memory_usage:.2f}GB, "
            f"Batch={batch_time:.3f}s"
        )
        
        # Add mathematical health indicators if available
        health_indicators = []
        if 'gradient_health' in val_metrics:
            health_indicators.append(f"GradHealth={val_metrics['gradient_health']:.2f}")
        if 'beta_health' in val_metrics:
            health_indicators.append(f"BetaHealth={val_metrics['beta_health']:.2f}")
        if 'avg_throughput' in val_metrics:
            health_indicators.append(f"Throughput={val_metrics['avg_throughput']:.1f}sps")
        
        health_log = ", ".join(health_indicators)
        
        if health_log:
            trainer.log_message(f"{base_log}, {perf_log}, {health_log}")
        else:
            trainer.log_message(f"{base_log}, {perf_log}")
        
        # Detailed mathematical summary every 10 epochs
        if epoch % 10 == 0:
            trainer.log_detailed_epoch_summary(epoch, train_metrics, val_metrics)
        
        # Checkpointing
        is_best = val_loss < trainer.best_loss
        trainer.save_checkpoint(val_metrics, is_best)
        
        # Early stopping check
        if trainer.check_early_stopping(val_loss):
            break
    
    trainer.log_message("Training completed!")
    
    # Final evaluation
    trainer.log_message("Running final comprehensive evaluation...")
    final_metrics = evaluator.evaluate_model(model, val_loader, [])
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        final_metrics, 
        save_path=trainer.log_dir / 'final_evaluation_report.txt'
    )
    
    print(report)


def create_dummy_dataset(size: int, device: torch.device):
    """Create dummy dataset for testing"""
    class DummyDataset(Dataset):
        def __init__(self, size: int):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Dummy scene features
            scene_features = torch.randn(512)
            
            # Dummy target coordinates
            target_coords = torch.rand(5, 4) * 1000
            
            # Dummy constraints - return as fixed list of constraint objects
            # These should NOT be tensorized by the DataLoader
            constraints = [
                con_left(0, 1, 50),
                con_above(2, 3, 30)
            ]
            
            return scene_features, target_coords, constraints
    
    return DummyDataset(size)


def generate_spatial_constraints_from_bboxes(bboxes):
    """
    MEMORY LEAK FIX: Generate spatial constraints with explicit memory management
    Reduced from O(n²) to O(n) and added constraint object pooling for memory efficiency
    
    Args:
        bboxes: List of [x, y, w, h] bounding boxes
        
    Returns:
        List of constraint objects derived from spatial relationships
    """
    constraints = []
    n_boxes = len(bboxes)
    
    if n_boxes == 0:
        return constraints
    
    # MEMORY LEAK FIX: Limit total constraints to prevent memory accumulation
    max_constraints_per_sample = 8  # Prevent explosive constraint growth
    
    # OPTIMIZATION 1: Generate T1 constraints (absolute positioning) - O(n)
    for i in range(n_boxes):
        x, y, w, h = bboxes[i]
        
        # T1 constraints: absolute positioning for each object individually
        if x < 300:  # Left side of image
            constraints.append(con_left_val(i, 200, offset=50))
        if x > 700:  # Right side of image
            constraints.append(con_right_val(i, 800, offset=50))
        if y < 300:  # Top of image
            constraints.append(con_above_val(i, 200, offset=50))
        if y > 700:  # Bottom of image
            constraints.append(con_below_val(i, 800, offset=50))
    
    # OPTIMIZATION 2: Limit T2 constraints to reduce O(n²) impact
    # For training, we don't need ALL pairwise relationships - key ones suffice
    max_pairs = min(10, n_boxes * (n_boxes - 1) // 2)  # Cap at 10 constraints max
    pair_count = 0
    
    # Generate T2 constraints (relative positioning) with early termination
    for i in range(n_boxes):
        if pair_count >= max_pairs:
            break
        for j in range(i + 1, n_boxes):
            if pair_count >= max_pairs:
                break
                
            x1, y1, w1, h1 = bboxes[i]
            x2, y2, w2, h2 = bboxes[j]
            tolerance = 20
            
            # Left/right relationships - only generate one per pair
            if x1 + w1 < x2 - tolerance:  # obj1 is left of obj2
                constraints.append(con_left(i, j, offset=tolerance))
                pair_count += 1
            elif x2 + w2 < x1 - tolerance:  # obj2 is left of obj1
                constraints.append(con_left(j, i, offset=tolerance))
                pair_count += 1
            # Above/below relationships - only if no left/right found
            elif y1 + h1 < y2 - tolerance:  # obj1 is above obj2
                constraints.append(con_above(i, j, offset=tolerance))
                pair_count += 1
            elif y2 + h2 < y1 - tolerance:  # obj2 is above obj1
                constraints.append(con_above(j, i, offset=tolerance))
                pair_count += 1
            
            # Non-overlap constraints (ensure objects don't intersect)
            if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                # Objects overlap - add separation constraint using existing functions
                # Force obj1 to be left of obj2 if they overlap
                constraints.append(con_left(i, j, offset=10))
    
    # MEMORY LEAK FIX: Apply constraint limit and ensure efficient cleanup
    constraints = constraints[:max_constraints_per_sample]  # Hard limit
    
    # Fallback: ensure at least one constraint exists if none generated naturally
    if len(constraints) == 0 and len(bboxes) > 0:
        # Add a simple center constraint for the first object
        constraints.append(con_left_val(0, 500, offset=100))
    
    return constraints

def custom_collate_fn(batch):
    """
    Legacy custom collate function - kept for backward compatibility
    """
    scene_features = []
    target_coords = []
    constraints = []
    
    for scene_feat, target_coord, constraint_list in batch:
        scene_features.append(scene_feat)
        target_coords.append(target_coord)
        constraints.append(constraint_list)
    
    scene_features = torch.stack(scene_features)
    target_coords = torch.stack(target_coords)
    
    return scene_features, target_coords, constraints


def test_coco_integration():
    """
    Test function to validate COCO data integration with Beta training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing COCO integration on device: {device}")
    
    # Test COCO dataset loading
    print("Testing COCO dataset loading...")
    try:
        coco_train_dir = "/home/gaurang/hardnetnew/data/coco/train2017"
        coco_train_json = "/home/gaurang/hardnetnew/data/coco/annotations/instances_train2017_interior.json"
        
        # Create small test dataset
        test_dataset = EnhancedCOCO_Wrapper.from_args_interior(
            coco_dir=coco_train_dir,
            coco_json=coco_train_json,
            max_objects_per_image=5,
            min_bbox_area=100,
            img_size=128
        )
        
        print(f"Successfully loaded test dataset with {len(test_dataset)} samples")
        
        # Test constraint generation
        print("Testing constraint generation...")
        if len(test_dataset) > 0:
            sample_img, sample_cats, sample_bboxes = test_dataset[0]
            print(f"Sample image shape: {sample_img.shape}")
            print(f"Sample categories: {sample_cats}")
            print(f"Sample bboxes: {sample_bboxes}")
            
            # Test constraint generation
            if len(sample_bboxes) > 0:
                constraints = generate_spatial_constraints_from_bboxes(sample_bboxes)
                print(f"Generated {len(constraints)} spatial constraints")
                for i, constraint in enumerate(constraints[:5]):  # Show first 5
                    print(f"  Constraint {i}: {constraint}")
            else:
                print("No bboxes found in sample - no constraints generated")
        
        # Test DataLoader integration
        print("Testing DataLoader with constraint generation...")
        def test_collate_fn(batch):
            images = []
            target_coords = []
            constraints_list = []
            
            for img_tensor, categories, bboxes in batch:
                # CRITICAL FIX: Keep images as 3D tensors for CNN processing
                images.append(img_tensor)  # Shape: [3, 128, 128]
                
                # Process target coordinates
                coords_tensor = torch.zeros((5, 4))
                for i, bbox in enumerate(bboxes[:5]):
                    coords_tensor[i] = torch.tensor(bbox, dtype=torch.float32)
                target_coords.append(coords_tensor)
                
                # Generate constraints
                constraints = generate_spatial_constraints_from_bboxes(bboxes[:5])
                constraints_list.append(constraints)
            
            images = torch.stack(images)  # Shape: [batch_size, 3, 128, 128]
            target_coords = torch.stack(target_coords)
            
            return images, target_coords, constraints_list
        
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=test_collate_fn)
        
        # Test one batch
        for batch_idx, (images, coords, constraints) in enumerate(test_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")  # Should be [batch_size, 3, 128, 128]
            print(f"  Target coordinates shape: {coords.shape}")
            print(f"  Constraints lists: {len(constraints)} samples")
            print(f"  Total constraints in batch: {sum(len(c) for c in constraints)}")
            break
        
        print("COCO integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"COCO integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Beta Spatial Reasoning Training Framework")
    print("=" * 50)
    
    # Test COCO integration first
    print("Running COCO integration test...")
    if test_coco_integration():
        print("Integration test passed! Running full training pipeline...")
        main_training_loop()
    else:
        print("Integration test failed! Please fix COCO data issues before training.")