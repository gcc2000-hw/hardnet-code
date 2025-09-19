
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
from collections import defaultdict,namedtuple
from abc import ABC, abstractmethod
from constraint_language_v2 import ConstraintT5, ConstraintT6, ConstraintOR, ConstraintNOT

@dataclass
class SoftConstraintViolation:
    """Represents a soft constraint violation with penalty information."""
    constraint_type: str
    constraint_name: str
    violation_magnitude: float
    penalty_value: float
    constraint_index: int
    

@dataclass 
class TemperatureSchedule:
    """Configuration for temperature scheduling during training."""
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    schedule_type: str = "exponential"  # "linear", "exponential", "cosine"
    total_epochs: int = 1000
    warmup_epochs: int = 100
    

class SoftConstraintPenalty(ABC):    
    @abstractmethod
    def compute_penalty(self, neural_output: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, List[SoftConstraintViolation]]:
        pass
    
    @abstractmethod
    def get_constraint_info(self) -> Dict[str, Any]:
        pass


class DistanceConstraintPenalty(SoftConstraintPenalty):
    def __init__(self, 
                 distance_constraints: List[ConstraintT5],
                 penalty_weight: float = 1.0,
                 constraint_names: Optional[List[str]] = None):
        self.distance_constraints = distance_constraints
        self.penalty_weight = penalty_weight
        self.constraint_names = constraint_names or [f"distance_{i}" for i in range(len(distance_constraints))]
        

    def compute_penalty(self, 
                    neural_output: torch.Tensor, 
                    temperature: float) -> Tuple[torch.Tensor, List[SoftConstraintViolation]]:
        if not self.distance_constraints:
            return torch.tensor(0.0, device=neural_output.device), []
        
        # Ensure proper format
        if neural_output.dim() == 1:
            neural_output = neural_output.unsqueeze(0)
        elif neural_output.dim() == 0:
            return torch.tensor(0.0, device=neural_output.device), []
        
        batch_size = neural_output.shape[0]
        total_features = neural_output.shape[1] if neural_output.dim() > 1 else neural_output.numel()
        
        if total_features % 4 != 0:
            return torch.tensor(0.0, device=neural_output.device), []
        
        n_objects = total_features // 4
        total_penalty = torch.tensor(0.0, device=neural_output.device)
        violations = []
        
        for i, constraint in enumerate(self.distance_constraints):
            obj1_id = constraint.o1
            obj2_id = constraint.o2  
            target_distance = constraint.offset
            
            # Validate object IDs
            if obj1_id >= n_objects or obj2_id >= n_objects:
                continue
            
            try:
                # Extract object properties
                obj1_x = neural_output[:, obj1_id * 4 + 0]
                obj1_y = neural_output[:, obj1_id * 4 + 1] 
                obj1_w = neural_output[:, obj1_id * 4 + 2]
                obj1_h = neural_output[:, obj1_id * 4 + 3]
                
                obj2_x = neural_output[:, obj2_id * 4 + 0]
                obj2_y = neural_output[:, obj2_id * 4 + 1]
                obj2_w = neural_output[:, obj2_id * 4 + 2]
                obj2_h = neural_output[:, obj2_id * 4 + 3]
            except IndexError:
                continue
            
            # Compute centers
            center1_x = obj1_x + obj1_w / 2
            center1_y = obj1_y + obj1_h / 2
            center2_x = obj2_x + obj2_w / 2
            center2_y = obj2_y + obj2_h / 2
            
            # Actual distance
            actual_distance = torch.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            
            # Distance error (should be minimized)
            distance_error = torch.abs(actual_distance - target_distance)
            error_magnitude = distance_error.mean().item()
            
            # CRITICAL FIX: Only report violations above threshold
            violation_threshold = 0.05  # 5% tolerance in normalized coordinates
            is_violation = error_magnitude > violation_threshold
            
            # Penalty increases with error (POSITIVE penalty for violations)
            # Use LINEAR error instead of squared to prevent explosion
            penalty = distance_error.mean()
            
            # FIXED SCALING: Prevent penalty explosion during curriculum transitions
            # Use gentle temperature scaling with clamping
            temperature_factor = max(0.1, min(2.0, temperature))  # Clamp temperature 0.1-2.0
            scaled_penalty = penalty / temperature_factor
            
            # GENTLE PROGRESSIVE SCALING: Prevent double amplification 
            progressive_weight = min(1.0, 0.5 + 0.5 / (temperature + 0.5))  # Range: 0.5-1.0
            
            # CLAMP FINAL PENALTY: Prevent extreme values during transitions
            final_penalty = self.penalty_weight * progressive_weight * scaled_penalty
            final_penalty = torch.clamp(final_penalty, max=2.0)  # EMERGENCY FIX: Reduced from 10.0
            total_penalty += final_penalty
            
            # CRITICAL FIX: Only append violations if error exceeds threshold
            if is_violation:
                violations.append(SoftConstraintViolation(
                    constraint_type="T5_distance",
                    constraint_name=self.constraint_names[i],
                    violation_magnitude=error_magnitude,
                    penalty_value=final_penalty.item(),
                    constraint_index=i
                ))
        
        return total_penalty, violations
    
    def get_constraint_info(self) -> Dict[str, Any]:
        return {
            'type': 'distance_constraints',
            'count': len(self.distance_constraints),
            'penalty_weight': self.penalty_weight,
            'constraint_names': self.constraint_names
        }


class DisjunctiveConstraintPenalty(SoftConstraintPenalty):
    def __init__(self,
                 or_constraints: List[ConstraintOR], 
                 penalty_weight: float = 1.0,
                 constraint_names: Optional[List[str]] = None):
        self.or_constraints = or_constraints
        self.penalty_weight = penalty_weight  
        self.constraint_names = constraint_names or [f"or_{i}" for i in range(len(or_constraints))]
        
        # Initialize logger for diagnostic outputs
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def compute_penalty(self,
                    neural_output: torch.Tensor,
                    temperature: float) -> Tuple[torch.Tensor, List[SoftConstraintViolation]]:
        
        if not self.or_constraints:
            return torch.tensor(0.0, device=neural_output.device), []
        
        # Ensure proper input format
        if neural_output.dim() == 1:
            neural_output = neural_output.unsqueeze(0)
        elif neural_output.dim() == 0:
            return torch.tensor(0.0, device=neural_output.device), []
            
        batch_size = neural_output.shape[0]
        total_features = neural_output.shape[1] if neural_output.dim() > 1 else neural_output.numel()
        
        # Validate format
        if total_features % 4 != 0:
            return torch.tensor(0.0, device=neural_output.device), []
        
        n_objects = total_features // 4
        
        total_penalty = torch.tensor(0.0, device=neural_output.device)
        violations = []
        
        for i, or_constraint in enumerate(self.or_constraints):
            sub_constraints = or_constraint.c
            sub_penalties = []
            
            # Evaluate each sub-constraint in the OR group
            for sub_constraint in sub_constraints:
                sub_penalty = self._evaluate_sub_constraint(sub_constraint, neural_output, n_objects)
                if sub_penalty is not None:
                    sub_penalties.append(sub_penalty)
            
            if sub_penalties:
                try:
                    penalty_tensor = torch.stack(sub_penalties, dim=-1)
                    
                    # For OR: penalty when ALL sub-constraints are violated
                    # We want: penalty = 0 when at least one constraint is satisfied
                    #         penalty > 0 when all constraints are violated
                    
                    # Use minimum violation (if min is 0, at least one constraint is satisfied)
                    min_violation = torch.min(penalty_tensor, dim=-1)[0]
                    
                    # FIX: Return max violation to ensure ANY violation contributes to loss
                    # This preserves the penalty from the most violating sample in the batch
                    or_penalty = min_violation.max() if min_violation.numel() > 0 else torch.tensor(0.0, device=neural_output.device)
                    violation_magnitude = or_penalty.item()
                    
                    # CLAMP PENALTY: Prevent extreme values during transitions
                    clamped_penalty = torch.clamp(self.penalty_weight * or_penalty, max=2.0)
                    
                    
                    total_penalty += clamped_penalty
                    
                    # CRITICAL FIX: Only append violations above threshold
                    violation_threshold = 0.01  # Lower threshold for OR constraints
                    if violation_magnitude > violation_threshold:
                        violations.append(SoftConstraintViolation(
                            constraint_type="OR_disjunction",
                            constraint_name=self.constraint_names[i],
                            violation_magnitude=violation_magnitude,
                            penalty_value=clamped_penalty.item(),
                            constraint_index=i
                        ))
                except RuntimeError:
                    continue
        
        return total_penalty, violations

    
    def _evaluate_sub_constraint(self, constraint: Any, neural_output: torch.Tensor, n_objects: int) -> Optional[torch.Tensor]:
        try:
            # FIXED: Handle T1 constraints (single object with value comparison)
            if hasattr(constraint, 'o1') and hasattr(constraint, 'val') and not hasattr(constraint, 'o2'):
                # T1 constraint: single object vs value (like x < 250, x > 500)
                obj_id = constraint.o1
                var_id = getattr(constraint, 'v1', 0)  # 0=x, 1=y, 2=w, 3=h
                val = constraint.val
                constraint_type = constraint.c  # 'lt', 'gt', etc.
                
                if obj_id >= n_objects:
                    return None
                
                # Get the variable value (x, y, w, h)
                obj_var = neural_output[:, obj_id * 4 + var_id]
                
                # Calculate violation based on constraint type
                if constraint_type == 'lt':
                    # x < val: violation when x >= val
                    violation = F.relu(obj_var - val)
                elif constraint_type == 'gt':  
                    # x > val: violation when x <= val
                    violation = F.relu(val - obj_var)
                else:
                    return torch.tensor(0.0, device=neural_output.device)
                
                # Convert violation to penalty (higher violation = higher penalty)
                penalty = violation.mean(dim=0) if violation.dim() > 0 else violation
                
                
                return penalty
            
            elif hasattr(constraint, 'o1') and hasattr(constraint, 'o2'):
                # T2 constraint: two object comparison
                obj1_id = constraint.o1
                obj2_id = constraint.o2
                
                # Validate object IDs
                if obj1_id >= n_objects or obj2_id >= n_objects:
                    return None
                
                # Safe indexing
                obj1_x = neural_output[:, obj1_id * 4 + 0]
                obj1_w = neural_output[:, obj1_id * 4 + 2]
                obj2_x = neural_output[:, obj2_id * 4 + 0] 
                
                # Check horizontal separation (simplified)
                separation = obj2_x - (obj1_x + obj1_w)
                raw_penalty = F.relu(-separation)  # Penalty when objects overlap
                # CLAMP SUB-CONSTRAINT PENALTY: Prevent extreme overlap penalties
                clamped_penalty = torch.clamp(raw_penalty, max=2.0)
                penalty = clamped_penalty.mean(dim=0)
                return penalty
            else:
                # Unknown constraint type
                return torch.tensor(0.0, device=neural_output.device)
        except (IndexError, RuntimeError):
            # Return None if evaluation fails
            return None
    
    def get_constraint_info(self) -> Dict[str, Any]:
        return {
            'type': 'or_constraints',
            'count': len(self.or_constraints),
            'penalty_weight': self.penalty_weight,
            'constraint_names': self.constraint_names
        }


class ComplexMultiObjectPenalty(SoftConstraintPenalty):
    
    def __init__(self,
                 complex_constraints: List[ConstraintT6],
                 penalty_weight: float = 1.0, 
                 constraint_names: Optional[List[str]] = None):
        self.complex_constraints = complex_constraints
        self.penalty_weight = penalty_weight
        self.constraint_names = constraint_names or [f"complex_{i}" for i in range(len(complex_constraints))]
    
    def compute_penalty(self,
                    neural_output: torch.Tensor,
                    temperature: float) -> Tuple[torch.Tensor, List[SoftConstraintViolation]]:
        """Compute penalty for complex multi-object constraint violations with safe dimension handling."""
        if not self.complex_constraints:
            return torch.tensor(0.0, device=neural_output.device), []
        
        # Ensure proper input format
        if neural_output.dim() == 1:
            neural_output = neural_output.unsqueeze(0)
        elif neural_output.dim() == 0:
            return torch.tensor(0.0, device=neural_output.device), []
            
        batch_size = neural_output.shape[0]
        total_features = neural_output.shape[1] if neural_output.dim() > 1 else neural_output.numel()
        
        # Validate format
        if total_features % 4 != 0:
            return torch.tensor(0.0, device=neural_output.device), []
        
        n_objects = total_features // 4
        
        total_penalty = torch.tensor(0.0, device=neural_output.device)
        violations = []
        
        for i, constraint in enumerate(self.complex_constraints):
            try:
                # Example: alignment constraint between 3 objects
                penalty = self._compute_alignment_penalty(constraint, neural_output, temperature, n_objects)
                if penalty is not None:
                    violation_magnitude = penalty.item()
                    # CLAMP PENALTY: Prevent extreme values during transitions
                    clamped_penalty = torch.clamp(self.penalty_weight * penalty, max=2.0)
                    total_penalty += clamped_penalty
                    
                    # CRITICAL FIX: Only append violations above threshold
                    violation_threshold = 0.02  # Threshold for complex constraints
                    if violation_magnitude > violation_threshold:
                        violations.append(SoftConstraintViolation(
                            constraint_type="T6_complex",
                            constraint_name=self.constraint_names[i],
                            violation_magnitude=violation_magnitude,
                            penalty_value=clamped_penalty.item(),
                            constraint_index=i
                        ))
            except (IndexError, RuntimeError):
                # Skip constraints that fail
                continue
        
        return total_penalty, violations
    
    def _compute_alignment_penalty(self, constraint: ConstraintT6, neural_output: torch.Tensor, 
                                temperature: float, n_objects: int) -> Optional[torch.Tensor]:
        try:
            # Validate object IDs
            if (constraint.o1 >= n_objects or constraint.o2 >= n_objects or 
                constraint.o3 >= n_objects):
                return None
            
            # Safe indexing for x positions
            if neural_output.dim() == 2:
                obj1_x = neural_output[:, constraint.o1 * 4 + 0]
                obj2_x = neural_output[:, constraint.o2 * 4 + 0]
                obj3_x = neural_output[:, constraint.o3 * 4 + 0]
            else:
                # Handle 1D case
                obj1_x = neural_output[constraint.o1 * 4 + 0]
                obj2_x = neural_output[constraint.o2 * 4 + 0]
                obj3_x = neural_output[constraint.o3 * 4 + 0]
            
            # Penalize when objects are not aligned
            alignment_error = torch.var(torch.stack([obj1_x, obj2_x, obj3_x]))
            # SAFE TEMPERATURE SCALING: Prevent extreme division by small temperature
            temperature_factor = max(0.1, min(2.0, temperature))  # Clamp temperature 0.1-2.0
            scaled_penalty = alignment_error / temperature_factor
            # CLAMP ALIGNMENT PENALTY: Prevent extreme values
            return torch.clamp(scaled_penalty, max=2.0)
            
        except (IndexError, RuntimeError):
            return None
    
    def get_constraint_info(self) -> Dict[str, Any]:
        return {
            'type': 'complex_constraints', 
            'count': len(self.complex_constraints),
            'penalty_weight': self.penalty_weight,
            'constraint_names': self.constraint_names
        }


class TemperatureScheduler:
    
    def __init__(self, schedule_config: TemperatureSchedule):
        self.config = schedule_config
        self.current_epoch = 0
        
    def get_temperature(self, epoch: Optional[int] = None) -> float:
        """Get current temperature based on epoch."""
        if epoch is not None:
            self.current_epoch = epoch
            
        # Warmup phase: keep initial temperature
        if self.current_epoch < self.config.warmup_epochs:
            return self.config.initial_temperature
        
        # Progress through schedule
        progress = (self.current_epoch - self.config.warmup_epochs) / (self.config.total_epochs - self.config.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        if self.config.schedule_type == "linear":
            temperature = self.config.initial_temperature + progress * (self.config.final_temperature - self.config.initial_temperature)
        elif self.config.schedule_type == "exponential": 
            # Exponential decay: T(t) = T_init * (T_final/T_init)^progress
            ratio = self.config.final_temperature / self.config.initial_temperature
            temperature = self.config.initial_temperature * (ratio ** progress)
        elif self.config.schedule_type == "cosine":
            # Cosine annealing
            temperature = self.config.final_temperature + 0.5 * (self.config.initial_temperature - self.config.final_temperature) * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")
        
        return max(temperature, self.config.final_temperature)  # Ensure we don't go below final temp
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch


class SoftConstraintHandler(nn.Module):
    
    def __init__(self,
                 distance_constraints: Optional[List[ConstraintT5]] = None,
                 or_constraints: Optional[List[ConstraintOR]] = None,
                 complex_constraints: Optional[List[ConstraintT6]] = None,
                 temperature_schedule: Optional[TemperatureSchedule] = None,
                 enable_logging: bool = True):
        super(SoftConstraintHandler, self).__init__()
        
        # Initialize constraint handlers
        self.penalty_handlers = []
        
        if distance_constraints:
            self.penalty_handlers.append(DistanceConstraintPenalty(distance_constraints))
            
        if or_constraints:
            self.penalty_handlers.append(DisjunctiveConstraintPenalty(or_constraints))
            
        if complex_constraints:
            self.penalty_handlers.append(ComplexMultiObjectPenalty(complex_constraints))
        
        # Temperature scheduling
        self.temperature_scheduler = TemperatureScheduler(
            temperature_schedule or TemperatureSchedule()
        )
        
        # Logging and statistics
        self.enable_logging = enable_logging
        self.logger = self._setup_logging()
        self.violation_history = []
        self.temperature_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for soft constraint operations."""
        logger = logging.getLogger('SoftConstraintHandler')
        if self.enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def forward(self, neural_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        current_temperature = self.temperature_scheduler.get_temperature()
        
        total_penalty = torch.tensor(0.0, device=neural_output.device)
        all_violations = []
        penalty_breakdown = {}
        
        # Compute penalties from all handlers
        for handler in self.penalty_handlers:
            penalty, violations = handler.compute_penalty(neural_output, current_temperature)
            
            # CLAMP INDIVIDUAL HANDLER PENALTY: Prevent accumulation explosions
            clamped_handler_penalty = torch.clamp(penalty, max=5.0)  # EMERGENCY FIX: Reduced from 50.0
            total_penalty += clamped_handler_penalty
            all_violations.extend(violations)
            
            handler_info = handler.get_constraint_info()
            penalty_breakdown[handler_info['type']] = {
                'penalty': penalty.item(),
                'violations': len(violations),
                'constraint_count': handler_info['count']
            }
        
        # FINAL TOTAL PENALTY CLAMP: Ultimate safety net against explosions
        total_penalty = torch.clamp(total_penalty, max=5.0)  # EMERGENCY FIX: Reduced from 100.0
        
        # Track statistics
        if self.training:
            self._record_statistics(current_temperature, total_penalty.item(), all_violations)
        
        
        constraint_info = {
            'total_penalty': total_penalty.item(),
            'temperature': current_temperature,
            'penalty_breakdown': penalty_breakdown,
            'violations': all_violations,
            'n_violations': len(all_violations)
        }
        
        return total_penalty, constraint_info
    
    def _record_statistics(self, temperature: float, total_penalty: float, violations: List[SoftConstraintViolation]):
        """Record constraint violation statistics."""
        self.temperature_history.append(temperature)
        self.violation_history.append({
            'epoch': self.temperature_scheduler.current_epoch,
            'temperature': temperature,
            'total_penalty': total_penalty,
            'n_violations': len(violations),
            'violations': violations
        })
        
        # Log periodically
        if len(self.violation_history) % 100 == 0:
            self._log_statistics_summary()
    
    def _log_statistics_summary(self):
        """Log summary of recent constraint statistics."""
        recent_violations = self.violation_history[-100:]
        
        avg_penalty = np.mean([v['total_penalty'] for v in recent_violations])
        avg_n_violations = np.mean([v['n_violations'] for v in recent_violations])
        current_temp = self.temperature_scheduler.get_temperature()
        
        self.logger.info(
            f"Soft Constraint Stats (last 100 batches): "
            f"Avg penalty: {avg_penalty:.3f}, "
            f"Avg violations: {avg_n_violations:.1f}, "
            f"Temperature: {current_temp:.3f}"
        )
    
    def update_epoch(self, epoch: int):
        """Update training epoch for temperature scheduling."""
        self.temperature_scheduler.update_epoch(epoch)
        
        if self.enable_logging:
            new_temp = self.temperature_scheduler.get_temperature()
            if epoch % 50 == 0:  # Log every 50 epochs
                self.logger.info(f"Epoch {epoch}: Temperature = {new_temp:.4f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive constraint handling statistics."""
        if not self.violation_history:
            return {'no_data': True}
        
        return {
            'total_epochs_tracked': len(self.violation_history),
            'current_temperature': self.temperature_scheduler.get_temperature(),
            'temperature_range': [min(self.temperature_history), max(self.temperature_history)],
            'average_penalty': np.mean([v['total_penalty'] for v in self.violation_history]),
            'penalty_trend': [v['total_penalty'] for v in self.violation_history[-20:]],  # Last 20 records
            'constraint_handlers': [h.get_constraint_info() for h in self.penalty_handlers]
        }
    
    def reset_statistics(self):
        """Reset all collected statistics."""
        self.violation_history.clear()
        self.temperature_history.clear()
        self.logger.info("Soft constraint statistics reset")


# Integration utilities
def create_soft_constraint_handler_from_phase1(non_affine_constraints: List[Any],
                                              temperature_config: Optional[TemperatureSchedule] = None) -> SoftConstraintHandler:

    # Separate constraints by type
    distance_constraints = [c for c in non_affine_constraints if isinstance(c, ConstraintT5)]
    or_constraints = [c for c in non_affine_constraints if isinstance(c, ConstraintOR)]  
    complex_constraints = [c for c in non_affine_constraints if isinstance(c, ConstraintT6)]
    
    return SoftConstraintHandler(
        distance_constraints=distance_constraints,
        or_constraints=or_constraints,
        complex_constraints=complex_constraints,
        temperature_schedule=temperature_config
    )


# Example usage and testing
def create_test_soft_constraints():
    """Create test soft constraints for demonstration."""
    # Mock constraints for testing
    distance_constraints = [
        ConstraintT5("mdisteq", 0, 1, 150),  # Objects 0,1 should be 150 units apart
        ConstraintT5("mdisteq", 1, 2, 200),  # Objects 1,2 should be 200 units apart
    ]
    
    # Mock OR constraint (disjoint objects)
    or_constraints = [
        ConstraintOR([
            # Mock sub-constraints for horizontal OR vertical separation
        ])
    ]
    
    complex_constraints = [
        ConstraintT6("align", 0, 1, 2, 0),  # 3-object alignment
    ]
    
    return distance_constraints, or_constraints, complex_constraints


if __name__ == "__main__":
    print("=== SPRING SOFT CONSTRAINT HANDLER - PHASE 4 ===\n")
    
    # Test 1: Create soft constraint handler
    print("TEST 1: Soft Constraint Handler Creation")
    distance_constraints, or_constraints, complex_constraints = create_test_soft_constraints()
    
    temperature_config = TemperatureSchedule(
        initial_temperature=2.0,
        final_temperature=0.1,
        schedule_type="exponential",
        total_epochs=500,
        warmup_epochs=50
    )
    
    soft_handler = SoftConstraintHandler(
        distance_constraints=distance_constraints,
        or_constraints=or_constraints, 
        complex_constraints=complex_constraints,
        temperature_schedule=temperature_config,
        enable_logging=True
    )
    
    print(f" Created soft constraint handler")
    print(f"Distance constraints: {len(distance_constraints)}")
    print(f"OR constraints: {len(or_constraints)}")
    print(f"Complex constraints: {len(complex_constraints)}")
    
    # Test 2: Temperature scheduling
    print(f"\nTEST 2: Temperature Scheduling")
    epochs_to_test = [0, 25, 50, 100, 250, 500]
    for epoch in epochs_to_test:
        soft_handler.update_epoch(epoch)
        temp = soft_handler.temperature_scheduler.get_temperature()
        print(f"  Epoch {epoch:3d}: Temperature = {temp:.4f}")
    
    # Test 3: Soft constraint penalty computation
    print(f"\nTEST 3: Soft Constraint Penalty Computation")
    batch_size = 2
    n_objects = 3
    n_variables = n_objects * 4  # [x, y, width, height] per object
    
    # Create test neural output with some constraint violations
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, n_variables) * 100  # Random layout
    
    print(f"Input shape: {test_input.shape}")
    print(f"Input sample: {test_input[0].tolist()}")
    
    # Compute penalties
    soft_handler.train()
    penalty_loss, constraint_info = soft_handler(test_input)
    
    print(f"\nSoft Constraint Results:")
    print(f"Total penalty: {penalty_loss.item():.4f}")
    print(f"Current temperature: {constraint_info['temperature']:.4f}")
    print(f"Number of violations: {constraint_info['n_violations']}")
    
    print(f"\nPenalty Breakdown:")
    for constraint_type, info in constraint_info['penalty_breakdown'].items():
        print(f"  {constraint_type}: penalty={info['penalty']:.4f}, violations={info['violations']}")
    
    # Test 4: Integration with different temperatures
    print(f"\nTEST 4: Temperature Effect on Penalties")
    soft_handler.eval()
    
    test_temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    for temp in test_temperatures:
        soft_handler.temperature_scheduler.config.initial_temperature = temp
        soft_handler.temperature_scheduler.config.final_temperature = temp
        soft_handler.update_epoch(100)  # Force temperature
        
        penalty, info = soft_handler(test_input)
        print(f"  Temperature {temp:3.1f}: Penalty = {penalty.item():.4f}")
    
    # Test 5: Statistics tracking
    print(f"\nTEST 5: Statistics Tracking")
    soft_handler.train()
    
    # Simulate training epochs
    for epoch in range(0, 101, 20):
        soft_handler.update_epoch(epoch)
        penalty, info = soft_handler(test_input)
    
    stats = soft_handler.get_statistics()
    print(f"Epochs tracked: {stats['total_epochs_tracked']}")
    print(f"Current temperature: {stats['current_temperature']:.4f}")
    print(f"Average penalty: {stats['average_penalty']:.4f}")
    print(f"Temperature range: [{stats['temperature_range'][0]:.3f}, {stats['temperature_range'][1]:.3f}]")
    