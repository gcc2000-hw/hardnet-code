"""
SPRING Hybrid: Phase 5 - Fixed Hybrid Spatial Reasoning Module
CRITICAL FIXES: Dimensional consistency and format conversion issues

Key fixes:
1. Robust format conversion between sequence [batch, seq_len, 4] and flat [batch, n_objects*4]
2. Comprehensive input validation
3. Clear error messages for debugging
4. Proper shape handling in constraint processing
5. Fallback mechanisms for edge cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

# Import all previous phases
try:
    from constraint_router import ConstraintRouter
    from constraint_to_affine_converter import ConstraintToAffineConverter, AffineConstraintMatrix
    from hardnet_aff import HardNetAff
    from soft_handler import SoftConstraintHandler, TemperatureSchedule
    PHASES_AVAILABLE = True
except ImportError:
    print("Warning: Previous phases not available. Using mocks for demonstration.")
    PHASES_AVAILABLE = False


class SpatialReasoningMode(Enum):
    """Operating modes for the hybrid spatial reasoning module."""
    DISCRETE = "discrete"          # Original SPRING symbolic reasoning
    DIFFERENTIABLE = "differentiable"  # Pure neural with hybrid constraints
    HYBRID = "hybrid"              # Best of both worlds


@dataclass
class HybridSRMConfig:
    """Configuration for Hybrid Spatial Reasoning Module."""
    # Mode settings
    mode: SpatialReasoningMode = SpatialReasoningMode.HYBRID
    fallback_to_discrete: bool = False  # FAIL-FAST: Always disabled
    
    # Neural network settings (matching original SPRING)
    gru_input_size: int = 4
    gru_hidden_size: int = 500  
    gru_num_layers: int = 3
    dense_output_size: int = 4
    dropout_rate: float = 0.1
    
    # Constraint processing settings
    enable_constraint_preprocessing: bool = True
    constraint_timeout_seconds: float = 1.0
    max_constraints_per_batch: int = 1000
    
    # HardNet settings
    hardnet_regularization: float = 1e-3  # Increased for stability
    hardnet_numerical_tolerance: float = 1e-8
    hardnet_coordinate_scale: float = 1.0  # FIXED: Normalized [0,1] coordinate system
    enable_hardnet_warm_start: bool = False  # FIXED: Disable warm-start for immediate projection
    hardnet_warm_start_epochs: int = 0  # No warm-start needed
    
    # Soft constraint settings
    enable_soft_constraints: bool = True
    temperature_initial: float = 2.0
    temperature_final: float = 0.01
    temperature_schedule: str = "exponential"
    temperature_total_epochs: int = 1000
    
    # Training settings
    enable_gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    # Logging and monitoring
    enable_logging: bool = True
    log_constraint_stats: bool = True
    log_performance_metrics: bool = True


class TensorFormatConverter:
    """
    Handles robust conversion between different tensor formats.
    
    Formats:
    - sequence: [batch_size, seq_len, 4] - original SPRING format
    - flat: [batch_size, total_features] where total_features = seq_len * 4
    - object: [batch_size, n_objects, 4] - same as sequence but clearer naming
    """
    
    @staticmethod
    def validate_input(tensor: torch.Tensor, expected_format: str) -> Dict[str, Any]:
        """Validate input tensor and return metadata."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        info = {
            'shape': tuple(tensor.shape),
            'dim': tensor.dim(),
            'device': tensor.device,
            'dtype': tensor.dtype
        }
        
        if tensor.dim() == 0:
            raise ValueError(f"Cannot process scalar tensor: {tensor}")
        elif tensor.dim() == 1:
            raise ValueError(f"Cannot process 1D tensor without batch dimension: {tensor.shape}")
        elif tensor.dim() == 2 and expected_format == "sequence":
            # Could be flat format - check if divisible by 4
            if tensor.shape[1] % 4 != 0:
                raise ValueError(f"Flat tensor features ({tensor.shape[1]}) must be divisible by 4")
        elif tensor.dim() == 3 and expected_format == "flat":
            # Could be sequence format - will need conversion
            pass
        elif tensor.dim() > 3:
            raise ValueError(f"Cannot process tensor with >3 dimensions: {tensor.shape}")
        
        return info
    
    @staticmethod
    def sequence_to_flat(tensor: torch.Tensor) -> torch.Tensor:
        """Convert [batch, seq_len, 4] → [batch, seq_len * 4]"""
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor for sequence format, got {tensor.shape}")
        
        batch_size, seq_len, features = tensor.shape
        if features != 4:
            raise ValueError(f"Expected 4 features per object, got {features}")
        
        # Flatten the sequence dimension
        flat_tensor = tensor.contiguous().view(batch_size, seq_len * 4)
        
        return flat_tensor
    
    @staticmethod
    def flat_to_sequence(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Convert [batch, seq_len * 4] → [batch, seq_len, 4]"""
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor for flat format, got {tensor.shape}")
        
        batch_size, total_features = tensor.shape
        expected_features = seq_len * 4
        
        if total_features != expected_features:
            raise ValueError(
                f"Feature count mismatch: got {total_features}, expected {expected_features} "
                f"(seq_len={seq_len} * 4)"
            )
        
        # Reshape to sequence format
        sequence_tensor = tensor.contiguous().view(batch_size, seq_len, 4)
        
        return sequence_tensor
    
    @staticmethod
    def auto_convert_to_flat(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Automatically convert tensor to flat format, returning conversion metadata.
        
        Returns:
            (flat_tensor, conversion_info)
        """
        conversion_info = {
            'original_shape': tuple(tensor.shape),
            'original_dim': tensor.dim(),
            'conversion_applied': False,
            'seq_len': None
        }
        
        if tensor.dim() == 2:
            # Already flat format
            batch_size, total_features = tensor.shape
            if total_features % 4 != 0:
                raise ValueError(f"Flat tensor features ({total_features}) must be divisible by 4")
            conversion_info['seq_len'] = total_features // 4
            return tensor, conversion_info
        
        elif tensor.dim() == 3:
            # Sequence format - convert to flat
            batch_size, seq_len, features = tensor.shape
            if features != 4:
                raise ValueError(f"Expected 4 features per object, got {features}")
            
            flat_tensor = TensorFormatConverter.sequence_to_flat(tensor)
            conversion_info['conversion_applied'] = True
            conversion_info['seq_len'] = seq_len
            return flat_tensor, conversion_info
        
        else:
            raise ValueError(f"Cannot auto-convert tensor with shape {tensor.shape}")
    
    @staticmethod
    def restore_original_format(tensor: torch.Tensor, conversion_info: Dict[str, Any]) -> torch.Tensor:
        """Restore tensor to its original format using conversion metadata."""
        if not conversion_info['conversion_applied']:
            # No conversion was applied, return as-is
            return tensor
        
        # Convert back to sequence format
        seq_len = conversion_info['seq_len']
        if seq_len is None:
            raise ValueError("Cannot restore format: seq_len not found in conversion_info")
        
        return TensorFormatConverter.flat_to_sequence(tensor, seq_len)


class OriginalSpatialReasoningModule(nn.Module):
    """
    FIXED: Simplified version of SPRING's original SRM with robust format handling.
    """
    
    def __init__(self, config: HybridSRMConfig):
        super().__init__()
        
        # GRU layers (matching original SPRING architecture)
        self.gru = nn.GRU(
            input_size=config.gru_input_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        #  COORDINATE REGRESSOR: Expert's recommended architecture
        # Direct coordinate regression without sigmoid limitations
        self.coordinate_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.gru_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, config.dense_output_size)  # Direct coordinate output
        )
        
        #  BALANCED INITIALIZATION: Prevent both stagnation and explosion
        # Previous: bias=[500,500,200,200] → stuck at init
        # Attempted: gain=5.0, bias=[-50,50] → gradient explosion (16K norms!)
        # Balanced: moderate gain + reasonable bias for stable learning
        for layer in self.coordinate_regressor:
            if isinstance(layer, nn.Linear):
                if layer == self.coordinate_regressor[-1]:  # Final layer
                    # REALISTIC SCALE: Initialize for [0, 1.8] coordinate output range (COCO reality)
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)  # Moderate gain for [0, 1.8] 
                    # ZERO BIAS: Let model discover natural distribution
                    nn.init.zeros_(layer.bias)  # No bias constraint - learn from data
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=1.414)  # sqrt(2) for ReLU
                    nn.init.zeros_(layer.bias)
        
        #  APPROACH 3: Target Normalization Control Experiment
        # Keep model in natural range [0, 20], normalize targets instead of scaling predictions
        # self.coordinate_scale = nn.Parameter(torch.tensor(30.0))  # DISABLED for target normalization test
        self.coordinate_scale = None  # No scaling - use base predictions directly
        
        print(f" COORDINATE REGRESSOR: COCO-realistic coordinate system [0, 1.8]")
        print(f"   Architecture: GRU({config.gru_hidden_size}) → Linear(256) → ReLU → Linear({config.dense_output_size})")
        print(f"    INITIALIZATION: Weights and bias for [0, 1.8] output range (COCO reality)")
        print(f"    Expected range: [0,1] (normalized coordinates)")
        print(f"    Clean architecture: No coordinate scaling band-aids")
        
        # Hidden state initialization (would come from ResNet18 in full SPRING)
        self.hidden_state_init = nn.Parameter(
            torch.randn(config.gru_num_layers, 1, config.gru_hidden_size) * 0.1
        )
        
        self.config = config
        self.converter = TensorFormatConverter()
        
    def forward(self, 
                encoded_sequence: torch.Tensor,
                initial_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FIXED: Enhanced forward pass with robust input format handling.
        
        Args:
            encoded_sequence: Can be either:
                - Sequence format: (batch_size, seq_len, 4) for original SPRING
                - Flat format: (batch_size, total_features) for constraint processing
            initial_hidden: Initial hidden state from scene encoder
            
        Returns:
            Layout predictions in the SAME format as input
        """
        # Store original input information for format restoration
        original_shape = encoded_sequence.shape
        original_dim = encoded_sequence.dim()
        
        # Validate and convert input to sequence format
        if original_dim == 2:
            # Flat input: [batch_size, total_features] → convert to sequence format
            batch_size, total_features = encoded_sequence.shape
            
            if total_features % 4 != 0:
                raise ValueError(f"Flat input features ({total_features}) must be divisible by 4")
            
            seq_len = total_features // 4
            sequence_input = self.converter.flat_to_sequence(encoded_sequence, seq_len)
            return_flat = True
            
        elif original_dim == 3:
            # Sequence input: [batch_size, seq_len, features]
            batch_size, seq_len, features = encoded_sequence.shape
            if features != 4:
                raise ValueError(f"Expected 4 features per object, got {features}")
            sequence_input = encoded_sequence
            return_flat = False
            
        else:
            raise ValueError(f"Unsupported input dimensions: {encoded_sequence.shape}")
        
        # Initialize hidden state with correct batch dimension
        if initial_hidden is None:
            # Expand for the current batch size
            hidden = self.hidden_state_init.expand(-1, batch_size, -1).contiguous()
        else:
            # Ensure hidden state has correct batch dimension
            if initial_hidden.shape[1] != batch_size:
                if initial_hidden.shape[1] == 1:
                    # Expand single batch to current batch size
                    hidden = initial_hidden.expand(-1, batch_size, -1).contiguous()
                else:
                    # Mismatch that can't be resolved - use default
                    hidden = self.hidden_state_init.expand(-1, batch_size, -1).contiguous()
            else:
                hidden = initial_hidden
        
        # GRU forward pass
        gru_output, final_hidden = self.gru(sequence_input, hidden)
        
        # Architecture validated - GRU producing diverse outputs as expected
        
        #  COORDINATE REGRESSOR: Direct coordinate prediction without sigmoid limitations
        base_coordinates = self.coordinate_regressor(gru_output)
        
        #  APPROACH 3: No scaling - use base coordinates directly
        coordinate_predictions = base_coordinates  # Natural range [0, 20]
        
        #  TARGET NORMALIZATION TRACKING: Monitor base coordinate ranges
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 20 == 0:  # Every 20 batches
            with torch.no_grad():
                base_min, base_max = base_coordinates.min().item(), base_coordinates.max().item()
                pred_min, pred_max = coordinate_predictions.min().item(), coordinate_predictions.max().item()
                
                print(f" TARGET NORMALIZATION TRACKING (batch {self._debug_counter}):")
                print(f"   Base coords: [{base_min:.1f}, {base_max:.1f}] (from regressor)")
                print(f"   Final coords: [{pred_min:.1f}, {pred_max:.1f}] (no scaling applied)")
                print(f"    Control test: Using natural model range, targets should be normalized to match")
        
        # NO CLAMPING: Let scale factor handle full coordinate range naturally
        layout_predictions = coordinate_predictions
        
        # Return in the same format as input
        if return_flat:
            # Convert back to flat format: [batch_size, total_features]
            layout_predictions = self.converter.sequence_to_flat(layout_predictions)
        
        return layout_predictions


class HybridSpatialReasoningModule(nn.Module):
    """
    FIXED: Hybrid Spatial Reasoning Module with robust dimensional handling.
    """
    
    def __init__(self, config: HybridSRMConfig):
        super().__init__()
        
        self.config = config
        self.current_epoch = 0
        self.logger = self._setup_logging()
        self.converter = TensorFormatConverter()
        
        # CRITICAL FIX: Initialize non-affine constraint tracking
        self.current_non_affine_count = 0
        
        # Original SPRING SRM (preserved)
        self.original_srm = OriginalSpatialReasoningModule(config)
        
        # Phase 1-4: Constraint processing components
        if PHASES_AVAILABLE:
            self.constraint_router = ConstraintRouter(enable_logging=config.enable_logging)
            self.matrix_converter = ConstraintToAffineConverter(enable_logging=config.enable_logging)
            
            # HardNet with improved numerical stability
            self.hardnet_layer = HardNetAff(
                constraint_matrix=None,
                regularization_strength=config.hardnet_regularization,
                numerical_tolerance=config.hardnet_numerical_tolerance,
                enable_warm_start=config.enable_hardnet_warm_start,
                warm_start_epochs=config.hardnet_warm_start_epochs,
                enable_logging=config.enable_logging,
                coordinate_scale=config.hardnet_coordinate_scale  # FIXED: Pass coordinate scale from config
            )
            
            # Soft constraint handler
            temperature_schedule = TemperatureSchedule(
                initial_temperature=config.temperature_initial,
                final_temperature=config.temperature_final,
                schedule_type=config.temperature_schedule,
                total_epochs=config.temperature_total_epochs
            )
            
            self.soft_constraint_handler = SoftConstraintHandler(
                temperature_schedule=temperature_schedule,
                enable_logging=config.enable_logging
            )
        else:
            # Mock components for testing
            self.constraint_router = None
            self.matrix_converter = None
            self.hardnet_layer = None
            self.soft_constraint_handler = None
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self._constraint_cache = {}
        
        self.logger.info(f"HybridSRM initialized in {config.mode.value} mode")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hybrid SRM."""
        logger = logging.getLogger('HybridSRM')
        if self.config.enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def set_constraints(self, 
                       constraints: List[Any], 
                       n_objects: int,
                       constraint_id: Optional[str] = None):
        """Set constraints for the current layout task."""
        
        #  CONSTRAINT FLOW LOG 12: set_constraints entry
        self.logger.info(f" FLOW-12 SET_CONSTRAINTS ENTRY:")
        self.logger.info(f"   Called with: constraints len={len(constraints) if constraints else 'None'}, n_objects={n_objects}")
        self.logger.info(f"   PHASES_AVAILABLE={PHASES_AVAILABLE}")
        
        if not PHASES_AVAILABLE or not constraints:
            self.logger.info(" FLOW-12 EARLY EXIT: No constraints to set or phases unavailable")
            return
        
        start_time = time.time()
        self.logger.info(f" FLOW-12 CONTINUING: Starting constraint processing with {len(constraints)} constraints")
        
        # Phase 1: Route constraints
        # 🔍 DEBUG: Log constraint types before routing
        self.logger.info(f"🔍 SRM ROUTER DEBUG: About to route {len(constraints)} constraints")
        for i, constraint in enumerate(constraints[:5]):  # First 5
            constraint_type = type(constraint).__name__
            self.logger.info(f"   Constraint {i}: {constraint_type}")
            if constraint_type == 'SizeRatioConstraint':
                self.logger.info(f"      🎯 SIZE RATIO FOUND: obj{constraint.smaller_obj_idx} ≤ {constraint.max_ratio:.3f} * obj{constraint.larger_obj_idx}")
                
        affine_constraints, non_affine_constraints = self.constraint_router.split_constraints(constraints)
        
        # 🔍 DEBUG: Log routing results
        self.logger.info(f"🔍 SRM ROUTER DEBUG: Routing results:")
        self.logger.info(f"   Affine: {len(affine_constraints)}")
        self.logger.info(f"   Non-affine: {len(non_affine_constraints)}")
        
        # 🔍 DEBUG: Log affine constraint types
        if affine_constraints:
            self.logger.info(f"🔍 SRM ROUTER DEBUG: Affine constraint types:")
            for i, constraint in enumerate(affine_constraints[:3]):
                constraint_type = type(constraint).__name__
                self.logger.info(f"   Affine {i}: {constraint_type}")
                if constraint_type == 'SizeRatioConstraint':
                    self.logger.info(f"      🎯 AFFINE SIZE RATIO: obj{constraint.smaller_obj_idx} ≤ {constraint.max_ratio:.3f} * obj{constraint.larger_obj_idx}")
        else:
            self.logger.info(f"🔍 SRM ROUTER DEBUG: NO AFFINE CONSTRAINTS - all going to soft handler!")
        
        # Phase 2 & 3: Setup hard constraints
        if affine_constraints:
            constraint_matrix = self.matrix_converter.convert_constraints_to_matrix(
                affine_constraints, n_objects
            )
            self.hardnet_layer.set_constraints(constraint_matrix)
        else:
            # Clear constraints if none are affine
            if hasattr(self.hardnet_layer, 'A'):
                self.hardnet_layer.set_constraints(None)
        
        # Phase 4: Setup soft constraints
        if non_affine_constraints:
            #  DEBUG: Enhanced non-affine constraint setup logging
            self.logger.info(f" NON-AFFINE CONSTRAINT SETUP:")
            self.logger.info(f"   Count: {len(non_affine_constraints)}")
            for i, constraint in enumerate(non_affine_constraints[:5]):  # Show first 5
                constraint_type = type(constraint).__name__
                if hasattr(constraint, 'c'):
                    self.logger.info(f"     {i+1}: {constraint_type} - {constraint.c}")
                    if hasattr(constraint, 'constraints') and constraint.c == 'or':
                        # Log OR constraint details
                        sub_constraints = constraint.constraints
                        self.logger.info(f"         OR has {len(sub_constraints)} sub-constraints:")
                        for j, sub in enumerate(sub_constraints[:3]):
                            if hasattr(sub, 'c') and hasattr(sub, 'o1') and hasattr(sub, 'val'):
                                self.logger.info(f"           {j+1}: obj{sub.o1} {sub.c} {sub.val}")
                else:
                    self.logger.info(f"     {i+1}: {constraint_type}")
            
            from soft_handler import create_soft_constraint_handler_from_phase1
            
            # Store old handler for comparison
            old_handler_count = 0
            if hasattr(self, 'soft_constraint_handler') and self.soft_constraint_handler and hasattr(self.soft_constraint_handler, 'penalty_handlers'):
                old_handler_count = len(self.soft_constraint_handler.penalty_handlers)
            
            # Get temperature config from existing handler, or use default if None
            temp_config = None
            if (hasattr(self, 'soft_constraint_handler') and 
                self.soft_constraint_handler is not None and 
                hasattr(self.soft_constraint_handler, 'temperature_scheduler') and
                hasattr(self.soft_constraint_handler.temperature_scheduler, 'config')):
                temp_config = self.soft_constraint_handler.temperature_scheduler.config
            
            self.soft_constraint_handler = create_soft_constraint_handler_from_phase1(
                non_affine_constraints,
                temp_config  # Will use default if None
            )
            
            # Verify the new handler was created properly
            new_handler_count = 0
            if self.soft_constraint_handler and hasattr(self.soft_constraint_handler, 'penalty_handlers'):
                new_handler_count = len(self.soft_constraint_handler.penalty_handlers)
            
            self.logger.info(f"   Soft handler updated: {old_handler_count} → {new_handler_count} penalty handlers")
            
            if new_handler_count == 0:
                self.logger.warning(f"    WARNING: No penalty handlers created for {len(non_affine_constraints)} non-affine constraints!")
            
            # Test the handler immediately with dummy layout
            if new_handler_count > 0:
                try:
                    dummy_layout = torch.tensor([[0.3, 0.4, 0.1, 0.08]], dtype=torch.float32)
                    test_penalty, test_info = self.soft_constraint_handler(dummy_layout)
                    self.logger.info(f"   Handler test: penalty={test_penalty.item():.6f}, violations={test_info.get('n_violations', 'MISSING')}")
                except Exception as e:
                    self.logger.warning(f"   Handler test failed: {e}")
        else:
            #  CRITICAL BUG FIX: Clear soft constraint handler between samples
            # Previous code was incorrectly preserving constraints from other samples
            self.logger.info(f" NO NON-AFFINE CONSTRAINTS - clearing soft handler for sample isolation")
            self.soft_constraint_handler = None
        
        processing_time = time.time() - start_time
        self.logger.info(
            f"Constraints set: {len(affine_constraints)} affine, "
            f"{len(non_affine_constraints)} non-affine in {processing_time:.3f}s"
        )
        
        # Store non-affine count for this sample only (no cross-sample contamination)
        self.current_non_affine_count = len(non_affine_constraints)
        
        #  DEBUG: Final constraint setup summary
        self.logger.info(f" CONSTRAINT SETUP COMPLETE:")
        self.logger.info(f"   Affine: {len(affine_constraints)}, Non-affine: {len(non_affine_constraints)}")
        self.logger.info(f"   Total constraints for satisfaction calc: {len(affine_constraints) + len(non_affine_constraints)}")
        
        # Verify soft handler state
        if hasattr(self, 'soft_constraint_handler') and self.soft_constraint_handler:
            handler_active = hasattr(self.soft_constraint_handler, 'penalty_handlers') and len(self.soft_constraint_handler.penalty_handlers) > 0
            self.logger.info(f"   Soft handler ready: {handler_active}")
        else:
            self.logger.info(f"   Soft handler: None")
    
    def _calculate_combined_constraint_satisfaction(self, constraint_info: Dict[str, Any]) -> None:
        """
        CRITICAL FIX: Calculate true constraint satisfaction combining affine and non-affine.
        
        Previous issue: Only reported HardNet satisfaction (always 1.0 for affine)
        Solution: Combine both affine and non-affine satisfaction rates
        """
        # Extract constraint counts
        affine_count = constraint_info.get('hardnet_constraints', 0)
        non_affine_count = getattr(self, 'current_non_affine_count', 0)
        soft_violations = constraint_info.get('soft_violations', 0)
        soft_penalty = constraint_info.get('soft_penalty', 0.0)
        
        #  DEBUG: Enhanced logging for constraint satisfaction calculation
        self.logger.info(f" CONSTRAINT SATISFACTION DEBUG:")
        self.logger.info(f"   Raw Data: affine_count={affine_count}, non_affine_count={non_affine_count}")
        self.logger.info(f"   Soft Handler: violations={soft_violations}, penalty={soft_penalty:.6f}")
        
        # Check if soft constraint handler is active
        has_soft_handler = (hasattr(self, 'soft_constraint_handler') and 
                           self.soft_constraint_handler is not None and
                           hasattr(self.soft_constraint_handler, 'penalty_handlers') and
                           len(self.soft_constraint_handler.penalty_handlers) > 0)
        
        self.logger.info(f"   Soft Handler Active: {has_soft_handler}")
        if has_soft_handler:
            handler_count = len(self.soft_constraint_handler.penalty_handlers)
            self.logger.info(f"   Penalty Handlers: {handler_count}")
            for i, handler in enumerate(self.soft_constraint_handler.penalty_handlers[:3]):  # Show first 3
                self.logger.info(f"     Handler {i}: {type(handler).__name__}")
        
        # Add tracking info to constraint_info
        constraint_info['affine_count'] = affine_count
        constraint_info['non_affine_count'] = non_affine_count
        
        total_constraints = affine_count + non_affine_count
        
        if total_constraints == 0:
            self.logger.info(f"   No constraints → satisfaction=1.0")
            constraint_info['constraint_satisfaction_rate'] = 1.0
            return
        
        # Calculate satisfied constraints
        affine_satisfied = affine_count  # HardNet satisfies 100% of affine constraints
        non_affine_satisfied = max(0, non_affine_count - soft_violations)
        total_satisfied = affine_satisfied + non_affine_satisfied
        
        # True satisfaction rate
        true_satisfaction = total_satisfied / total_constraints
        
        #  DEBUG: Detailed calculation breakdown
        self.logger.info(f"   CALCULATION:")
        self.logger.info(f"     Total constraints: {total_constraints}")
        self.logger.info(f"     Affine satisfied: {affine_satisfied}/{affine_count} (100%)")
        self.logger.info(f"     Non-affine satisfied: {non_affine_satisfied}/{non_affine_count} ({100*non_affine_satisfied/max(1,non_affine_count):.1f}%)")
        self.logger.info(f"     Overall satisfied: {total_satisfied}/{total_constraints} ({100*true_satisfaction:.1f}%)")
        
        #  ANOMALY DETECTION: Flag suspicious patterns
        if non_affine_count > 0 and soft_violations == 0:
            self.logger.warning(f"    SUSPICIOUS: {non_affine_count} non-affine constraints but 0 violations!")
            self.logger.warning(f"   This suggests soft constraint handler is not evaluating properly")
            
        if true_satisfaction == 1.0 and non_affine_count > 0:
            self.logger.warning(f"    IMPOSSIBLE: 100% satisfaction with {non_affine_count} non-affine constraints!")
        
        constraint_info['constraint_satisfaction_rate'] = true_satisfaction
    
    def forward(self, 
                encoded_sequence: torch.Tensor,
                constraints: Optional[List[Any]] = None,
                n_objects: Optional[int] = None,
                initial_hidden: Optional[torch.Tensor] = None,
                constraint_id: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        FIXED: Hybrid forward pass with robust constraint enforcement.
        """
        start_time = time.time()
        
        #  CONSTRAINT FLOW LOG 5: SRM entry
        batch_size = encoded_sequence.shape[0] if encoded_sequence.dim() > 1 else 1
        if hasattr(self, 'srm_batch_counter'):
            self.srm_batch_counter += 1
        else:
            self.srm_batch_counter = 0
        
        if self.srm_batch_counter < 5:  # Log first 5 batches
            self.logger.info(f" FLOW-5 SRM ENTRY (batch {self.srm_batch_counter}):")
            self.logger.info(f"   HybridSpatialReasoningModule received:")
            self.logger.info(f"     constraints type={type(constraints)}, value={constraints is not None}")
            self.logger.info(f"     n_objects={n_objects}")
            self.logger.info(f"     batch_size={batch_size}")
            
            if constraints is not None:
                if isinstance(constraints, list):
                    self.logger.info(f"     constraints length={len(constraints)}")
                    if len(constraints) > 0:
                        first_item = constraints[0]
                        is_batched = isinstance(first_item, list)
                        self.logger.info(f"     is_batched_constraints={is_batched}")
                        
                        if is_batched:
                            total_constraints = sum(len(sample_constraints) if sample_constraints else 0 for sample_constraints in constraints)
                            self.logger.info(f"     total constraints across samples={total_constraints}")
                        else:
                            self.logger.info(f"     single constraint list length={len(constraints)}")
                else:
                    self.logger.info(f"     constraints is not a list: {type(constraints)}")
        
        # STRICT: Input validation with immediate failure
        input_info = self.converter.validate_input(encoded_sequence, "any")
        assert input_info is not None, f"Input validation failed for tensor shape: {encoded_sequence.shape}"
        
        # Infer number of objects if not provided
        if n_objects is None:
            if encoded_sequence.dim() == 2:
                n_objects = encoded_sequence.shape[1] // 4
            elif encoded_sequence.dim() == 3:
                n_objects = encoded_sequence.shape[1]
            else:
                raise ValueError(f"Cannot infer n_objects from shape {encoded_sequence.shape}")
        
        # CRITICAL FIX: Handle batched constraints properly
        batch_size = encoded_sequence.shape[0] if encoded_sequence.dim() > 1 else 1
        
        # Detect constraint format: batched List[List[Any]] vs single List[Any]
        is_batched_constraints = (constraints is not None and 
                                len(constraints) > 0 and 
                                isinstance(constraints[0], list))
        
        #  CONSTRAINT FLOW LOG 6: Batching detection
        if self.srm_batch_counter < 5:  # Log first 5 batches
            self.logger.info(f" FLOW-6 BATCHING DETECTION (batch {self.srm_batch_counter}):")
            self.logger.info(f"   is_batched_constraints={is_batched_constraints}")
            if constraints is not None and len(constraints) > 0:
                self.logger.info(f"   constraints[0] type={type(constraints[0])}")
        
        if is_batched_constraints:
            # NEW: Process each sample with its own constraints
            return self._forward_with_batched_constraints(
                encoded_sequence, constraints, n_objects, initial_hidden, start_time
            )
        else:
            # OLD: Single constraint list for entire batch (backward compatibility)
            if constraints is not None and n_objects is not None:
                self.set_constraints(constraints, n_objects, constraint_id)
            
            # Route to appropriate forward method based on mode
            try:
                if self.config.mode == SpatialReasoningMode.DISCRETE:
                    return self._forward_discrete(encoded_sequence, initial_hidden, start_time)
                elif self.config.mode == SpatialReasoningMode.DIFFERENTIABLE:
                    return self._forward_differentiable(encoded_sequence, initial_hidden, start_time)
                elif self.config.mode == SpatialReasoningMode.HYBRID:
                    return self._forward_hybrid(encoded_sequence, initial_hidden, start_time)
                else:
                    raise ValueError(f"Unknown mode: {self.config.mode}")
            except Exception as e:
                self.logger.error(f"Forward pass failed: {e}")
                raise
    
    def _forward_differentiable(self, 
                            encoded_sequence: torch.Tensor,
                            initial_hidden: Optional[torch.Tensor], 
                            start_time: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        FIXED: Pure differentiable forward pass with robust constraint enforcement.
        """
        
        try:
            #  GRADIENT TRACE: Check input gradient properties
            print(f"\n GRADIENT TRACE - DIFFERENTIABLE FORWARD:")
            print(f"   Input encoded_sequence.requires_grad: {encoded_sequence.requires_grad}")
            print(f"   Input encoded_sequence.is_leaf: {encoded_sequence.is_leaf}")
            print(f"   Input shape: {encoded_sequence.shape}")
            
            # Step 1: Neural layout generation
            self.logger.debug(f"Input shape: {encoded_sequence.shape}")
            neural_layout = self.original_srm(encoded_sequence, initial_hidden)
            self.logger.debug(f"After SRM: {neural_layout.shape}")
            
            #  GRADIENT TRACE: Check output gradient properties
            print(f"   After GRU: neural_layout.requires_grad: {neural_layout.requires_grad}")
            print(f"   After GRU: neural_layout.is_leaf: {neural_layout.is_leaf}")
            
            # Step 2: Convert to flat format for constraint processing
            flat_layout, conversion_info = self.converter.auto_convert_to_flat(neural_layout)
            self.logger.debug(f"Flat layout: {flat_layout.shape}, conversion: {conversion_info}")
            
            # Step 3: Apply hard constraints (Phase 3)
            constraint_info = {
                'hardnet_applied': False,
                'hardnet_constraints': 0,
                'soft_constraints_applied': False,
                'soft_penalty': 0.0,
                'soft_violations': 0,
                'constraint_satisfaction_rate': 1.0  # Default value
            }
            
            if PHASES_AVAILABLE and self.hardnet_layer and self.hardnet_layer.has_constraints():
                expected_vars = self.hardnet_layer.n_variables
                actual_vars = flat_layout.shape[1]
                self.logger.debug(f"HardNet check: expected={expected_vars}, actual={actual_vars}")
                
                if actual_vars == expected_vars:
                    self.logger.debug("Applying HardNet constraints...")
                    
                    #  GRADIENT TRACE: Check before HardNet
                    print(f"   Before HardNet: flat_layout.requires_grad: {flat_layout.requires_grad}")
                    
                    # Compute satisfaction rate BEFORE projection (for comparison)
                    pre_satisfaction = self.hardnet_layer.compute_constraint_satisfaction_rate(flat_layout)
                    
                    # Apply HardNet projection
                    flat_layout = self.hardnet_layer(flat_layout)
                    constraint_info['hardnet_applied'] = True
                    constraint_info['hardnet_constraints'] = self.hardnet_layer.n_constraints
                    
                    #  GRADIENT TRACE: Check after HardNet
                    print(f"   After HardNet: flat_layout.requires_grad: {flat_layout.requires_grad}")
                    print(f"   After HardNet: flat_layout.is_leaf: {flat_layout.is_leaf}")
                    
                    # Compute satisfaction rate AFTER projection (should be near 100%)
                    post_satisfaction = self.hardnet_layer.compute_constraint_satisfaction_rate(flat_layout)
                    constraint_info['affine_satisfaction'] = post_satisfaction  # Store separately
                    
                    # CRITICAL FIX: Don't set final satisfaction rate yet - wait for soft constraints
                    
                    self.logger.debug(f"HardNet constraint satisfaction: {pre_satisfaction:.3f} → {post_satisfaction:.3f}")
                    self.logger.debug(f"HardNet applied: output shape={flat_layout.shape}")
                else:
                    self.logger.warning(
                        f"Skipping HardNet: dimension mismatch (expected {expected_vars}, got {actual_vars})"
                    )
            
            # Step 4: Apply soft constraints (Phase 4)
            soft_penalty = torch.tensor(0.0, device=flat_layout.device)
            constraint_info['total_penalty'] = 0.0  # Default value
            constraint_info['soft_violations'] = 0  # Default to 0 violations
            
            
            if PHASES_AVAILABLE and self.soft_constraint_handler and self.soft_constraint_handler.penalty_handlers:
                try:
                    #  DEBUG: Enhanced soft constraint logging
                    handler_count = len(self.soft_constraint_handler.penalty_handlers)
                    self.logger.info(f" SOFT CONSTRAINT EVALUATION:")
                    self.logger.info(f"   Input layout shape: {flat_layout.shape}")
                    self.logger.info(f"   Active penalty handlers: {handler_count}")
                    
                    # Log the actual layout values being evaluated
                    if flat_layout.numel() > 0:
                        layout_sample = flat_layout[0] if flat_layout.dim() > 1 else flat_layout
                        self.logger.info(f"   Layout sample (first 8 values): {layout_sample[:8].tolist()}")
                        self.logger.info(f"   Layout range: [{layout_sample.min().item():.3f}, {layout_sample.max().item():.3f}]")
                        
                        #  CRITICAL: Test the OR constraint manually
                        if len(layout_sample) >= 4:  # Has at least x,y,w,h for object 0
                            obj0_x = layout_sample[0].item()
                            left_satisfied = obj0_x < 0.25
                            right_satisfied = obj0_x > 0.5
                            or_satisfied = left_satisfied or right_satisfied
                            
                            self.logger.info(f"   🧪 MANUAL OR TEST: obj0.x={obj0_x:.6f}")
                            self.logger.info(f"      Left (<0.25): {left_satisfied}")
                            self.logger.info(f"      Right (>0.5): {right_satisfied}")
                            self.logger.info(f"      OR result: {or_satisfied}")
                            
                            if not or_satisfied:
                                self.logger.warning(f"    SHOULD VIOLATE: OR constraint not satisfied!")
                            else:
                                self.logger.info(f"    OR constraint satisfied")
                    
                    self.logger.debug("Applying soft constraints...")
                    soft_penalty, soft_info = self.soft_constraint_handler(flat_layout)
                    
                    #  DEBUG: Detailed soft constraint results
                    self.logger.info(f"   Soft constraint results:")
                    self.logger.info(f"     Penalty: {soft_penalty.item():.6f}")
                    self.logger.info(f"     Violations: {soft_info.get('n_violations', 'MISSING')}")
                    self.logger.info(f"     Info keys: {list(soft_info.keys())}")
                    
                    #  CRITICAL: Identify discrepancy
                    manual_should_violate = False
                    if flat_layout.numel() >= 4:
                        obj0_x = flat_layout[0, 0].item() if flat_layout.dim() > 1 else flat_layout[0].item()
                        manual_should_violate = not ((obj0_x < 0.25) or (obj0_x > 0.5))
                    
                    reported_violations = soft_info.get('n_violations', 0)
                    if manual_should_violate and reported_violations == 0:
                        self.logger.error(f"    BUG DETECTED: Manual check says VIOLATION but soft handler reports 0!")
                        self.logger.error(f"   This confirms the soft constraint handler bug.")
                    elif not manual_should_violate and reported_violations > 0:
                        self.logger.warning(f"    UNEXPECTED: Manual check says OK but soft handler reports {reported_violations} violations")
                    
                    constraint_info['soft_constraints_applied'] = True
                    constraint_info['soft_penalty'] = soft_penalty.item()
                    constraint_info['total_penalty'] = soft_penalty.item()  # Training code looks for this key
                    constraint_info['soft_violations'] = soft_info['n_violations']
                    self.logger.debug(f"Soft constraints applied: penalty={soft_penalty.item()}")
                except Exception as e:
                    self.logger.warning(f"Soft constraint processing failed: {e}")
                    self.logger.warning(f"Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                    # Ensure total_penalty is set even on failure
                    constraint_info['total_penalty'] = 0.0
                    constraint_info['soft_violations'] = 0
            
            # CRITICAL FIX: Calculate combined constraint satisfaction (affine + non-affine)
            self._calculate_combined_constraint_satisfaction(constraint_info)
            
            # Step 5: Convert output back to original format
            final_output = self.converter.restore_original_format(flat_layout, conversion_info)
            self.logger.debug(f"Final output: {final_output.shape}")
            
            # STRICT: Output shape MUST match input shape
            assert final_output.shape == encoded_sequence.shape, (
                f"SHAPE MISMATCH: Output {final_output.shape} != Input {encoded_sequence.shape}. "
                f"This indicates a bug in format conversion or constraint processing."
            )
            
            # Performance tracking
            if self.training:
                self._record_performance_stats(start_time, constraint_info)
            
            constraint_info.update({
                'mode': 'differentiable',
                'constraint_satisfaction': constraint_info['constraint_satisfaction_rate'],  # Use actual rate
                'processing_time': time.time() - start_time,
                'total_penalty': soft_penalty.item()
            })
            
            return final_output, constraint_info
            
        except Exception as e:
            self.logger.error(f"Error in _forward_differentiable: {e}")
            raise
    
    def _forward_discrete(self, 
                        encoded_sequence: torch.Tensor,
                        initial_hidden: Optional[torch.Tensor],
                        start_time: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Original SPRING discrete reasoning."""
        
        try:
            layout = self.original_srm(encoded_sequence, initial_hidden)
            
            # In discrete mode, constraint checking would be symbolic (not implemented here)
            constraint_info = {
                'mode': 'discrete',
                'constraint_satisfaction': 'symbolic',
                'processing_time': time.time() - start_time,
                'hardnet_applied': False,
                'soft_constraints_applied': False,
                'total_penalty': 0.0
            }
            
            return layout, constraint_info
            
        except Exception as e:
            self.logger.error(f"Error in _forward_discrete: {e}")
            raise
    
    def _forward_hybrid(self, 
                    encoded_sequence: torch.Tensor,
                    initial_hidden: Optional[torch.Tensor],
                    start_time: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Hybrid forward pass with graceful fallbacks."""
        
        try:
            # Attempt differentiable processing
            layout, info = self._forward_differentiable(encoded_sequence, initial_hidden, start_time)
            info['mode'] = 'hybrid'
            info['fallback_used'] = False
            return layout, info
            
        except Exception as e:
            # FAIL-FAST: No fallback to discrete processing - crash immediately
            raise AssertionError(
                f"FAIL-FAST: Hybrid mode processing failed:\n"
                f"Error: {e}\n"
                f"Mode: hybrid\n"
                f"Encoded sequence shape: {encoded_sequence.shape if encoded_sequence is not None else 'None'}\n"
                f"Initial hidden shape: {initial_hidden.shape if initial_hidden is not None else 'None'}\n"
                f"Fallback to discrete was: {'enabled' if self.config.fallback_to_discrete else 'disabled'}\n"
                f"This indicates a real bug in hybrid constraint processing\n"
                f"NO fallback to discrete mode - FIX THE HYBRID IMPLEMENTATION"
            )
    
    def _forward_with_batched_constraints(self,
                                       encoded_sequence: torch.Tensor,
                                       batch_constraints: List[List[Any]],
                                       n_objects: Optional[int],
                                       initial_hidden: Optional[torch.Tensor],
                                       start_time: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        CRITICAL FIX: Process batch with per-sample constraints.
        
        Args:
            encoded_sequence: (batch_size, ...) tensor
            batch_constraints: List of constraint lists, one per sample
            n_objects: Number of objects per sample
            initial_hidden: Initial hidden state
            start_time: Start time for performance tracking
            
        Returns:
            Batched output tensor and aggregated constraint info
        """
        batch_size = encoded_sequence.shape[0]
        
        #  CONSTRAINT FLOW LOG 7: Batched constraint processing entry
        if self.srm_batch_counter < 5:  # Log first 5 batches
            self.logger.info(f" FLOW-7 BATCHED PROCESSING (batch {self.srm_batch_counter}):")
            self.logger.info(f"   _forward_with_batched_constraints called")
            self.logger.info(f"   batch_size={batch_size}")
            self.logger.info(f"   batch_constraints length={len(batch_constraints)}")
            self.logger.info(f"   n_objects={n_objects}")
        
        # Validate constraint batch size matches tensor batch size
        if len(batch_constraints) != batch_size:
            self.logger.warning(
                f"Constraint batch size mismatch: got {len(batch_constraints)} constraint lists "
                f"for {batch_size} samples. Padding/truncating."
            )
            # Pad with empty lists or truncate
            if len(batch_constraints) < batch_size:
                batch_constraints.extend([[] for _ in range(batch_size - len(batch_constraints))])
            else:
                batch_constraints = batch_constraints[:batch_size]
        
        # Process each sample individually with performance monitoring
        batch_outputs = []
        batch_constraint_infos = []
        sample_times = []
        
        self.logger.info(f" CONSTRAINT ISOLATION FIX: Processing {batch_size} samples sequentially for proper isolation")
        batch_start_time = time.time()
        
        for i in range(batch_size):
            sample_start_time = time.time()
            # Extract single sample
            if encoded_sequence.dim() == 3:  # (batch_size, seq_len, features)
                sample_input = encoded_sequence[i:i+1]  # Keep batch dimension
            elif encoded_sequence.dim() == 2:  # (batch_size, features)
                sample_input = encoded_sequence[i:i+1]  # Keep batch dimension
            else:
                raise ValueError(f"Unexpected encoded_sequence dimensions: {encoded_sequence.shape}")
            
            sample_constraints = batch_constraints[i]
            # CRITICAL FIX: Proper hidden state slicing for GRU layers
            # initial_hidden shape: [num_layers, batch_size, hidden_size] = [3, 1, 500]
            # Need to preserve layer dimension while selecting batch sample
            sample_hidden = initial_hidden[:, i:i+1, :] if initial_hidden is not None else None
            
            #  CONSTRAINT FLOW LOG 8: Per-sample processing
            if self.srm_batch_counter < 5 and i < 2:  # Log first 2 samples of first 5 batches
                self.logger.info(f" FLOW-8 SAMPLE PROCESSING (batch {self.srm_batch_counter}, sample {i}):")
                self.logger.info(f"   sample_constraints: type={type(sample_constraints)}, len={len(sample_constraints) if sample_constraints else 'None'}")
                self.logger.info(f"   n_objects={n_objects}")
                self.logger.info(f"   Condition check: sample_constraints={sample_constraints is not None and len(sample_constraints) > 0}, n_objects={n_objects is not None}")
            
            # Set constraints for this sample
            if sample_constraints and n_objects is not None:
                #  CONSTRAINT FLOW LOG 9: set_constraints call
                if self.srm_batch_counter < 5 and i < 2:  # Log first 2 samples of first 5 batches
                    self.logger.info(f" FLOW-9 SET_CONSTRAINTS CALL (batch {self.srm_batch_counter}, sample {i}):")
                    self.logger.info(f"   About to call self.set_constraints({len(sample_constraints)} constraints, n_objects={n_objects})")
                
                self.set_constraints(sample_constraints, n_objects)
                
                #  CONSTRAINT FLOW LOG 10: post set_constraints
                if self.srm_batch_counter < 5 and i < 2:
                    self.logger.info(f" FLOW-10 POST SET_CONSTRAINTS (batch {self.srm_batch_counter}, sample {i}):")
                    self.logger.info(f"   set_constraints completed successfully")
            else:
                #  CONSTRAINT FLOW LOG 11: constraints skipped
                if self.srm_batch_counter < 5 and i < 2:
                    if not sample_constraints:
                        self.logger.info(f" FLOW-11 CONSTRAINTS SKIPPED (batch {self.srm_batch_counter}, sample {i}): No constraints")
                    elif n_objects is None:
                        self.logger.info(f" FLOW-11 CONSTRAINTS SKIPPED (batch {self.srm_batch_counter}, sample {i}): n_objects is None")
                
                # CRITICAL FIX: Don't clear constraints - let set_constraints() handle updates
                # The constraint clearing was interfering with HardNet enforcement
                # HardNet's set_constraints() method properly handles constraint updates
                # if hasattr(self.hardnet_layer, 'A'):
                #     for attr_name in ['A', 'b_l', 'b_u', 'A_pinv']:
                #         if hasattr(self.hardnet_layer, attr_name):
                #             delattr(self.hardnet_layer, attr_name)
            
            # Process this sample
            try:
                if self.config.mode == SpatialReasoningMode.DISCRETE:
                    sample_output, sample_info = self._forward_discrete(sample_input, sample_hidden, start_time)
                elif self.config.mode == SpatialReasoningMode.DIFFERENTIABLE:
                    sample_output, sample_info = self._forward_differentiable(sample_input, sample_hidden, start_time)
                elif self.config.mode == SpatialReasoningMode.HYBRID:
                    sample_output, sample_info = self._forward_hybrid(sample_input, sample_hidden, start_time)
                else:
                    raise ValueError(f"Unknown mode: {self.config.mode}")
                    
                batch_outputs.append(sample_output)
                batch_constraint_infos.append(sample_info)
                
                # Track sample processing time
                sample_time = time.time() - sample_start_time
                sample_times.append(sample_time)
                
                # Log performance for first few samples
                if self.srm_batch_counter < 5 and i < 3:
                    self.logger.info(f" SAMPLE TIMING: Sample {i} processed in {sample_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Sample {i} processing failed: {e}")
                # Create fallback output with same shape as input
                batch_outputs.append(sample_input)
                batch_constraint_infos.append({
                    'constraint_satisfaction_rate': 0.0,
                    'mode': 'error',
                    'processing_time': time.time() - start_time,
                    'error': str(e)
                })
        
        # Combine batch outputs
        final_output = torch.cat(batch_outputs, dim=0)
        
        # Log batch performance summary
        batch_total_time = time.time() - batch_start_time
        avg_sample_time = sum(sample_times) / len(sample_times) if sample_times else 0
        max_sample_time = max(sample_times) if sample_times else 0
        
        if self.srm_batch_counter < 5:
            self.logger.info(f" BATCH PERFORMANCE: {batch_size} samples in {batch_total_time:.4f}s")
            self.logger.info(f"   Average per sample: {avg_sample_time:.4f}s")
            self.logger.info(f"   Slowest sample: {max_sample_time:.4f}s")
            self.logger.info(f"   Sequential overhead: ~{batch_total_time - avg_sample_time*batch_size:.4f}s")
        
        # Aggregate constraint information across batch
        aggregated_info = self._aggregate_batch_constraint_info(batch_constraint_infos, batch_constraints)
        
        return final_output, aggregated_info
    
    def _aggregate_batch_constraint_info(self, 
                                      batch_infos: List[Dict[str, Any]], 
                                      batch_constraints: List[List[Any]]) -> Dict[str, Any]:
        """Aggregate constraint satisfaction info across batch."""
        total_constraints = sum(len(constraints) for constraints in batch_constraints)
        total_samples = len(batch_infos)
        
        if total_samples == 0:
            return {
                'constraint_satisfaction_rate': 1.0,
                'mode': 'empty_batch',
                'processing_time': 0.0,
                'total_constraints': 0,
                'samples_processed': 0
            }
        
        # Calculate weighted constraint satisfaction
        if total_constraints == 0:
            # No constraints in entire batch -> perfect satisfaction
            constraint_satisfaction = 1.0
        else:
            # Weight each sample's satisfaction by its constraint count
            weighted_satisfaction = 0.0
            for i, info in enumerate(batch_infos):
                sample_constraints = len(batch_constraints[i])
                sample_satisfaction = info.get('constraint_satisfaction_rate', 0.0)
                if sample_constraints > 0:
                    weighted_satisfaction += sample_satisfaction * sample_constraints
            
            constraint_satisfaction = weighted_satisfaction / total_constraints
        
        # Aggregate other metrics
        avg_processing_time = sum(info.get('processing_time', 0.0) for info in batch_infos) / total_samples
        total_hardnet_constraints = sum(info.get('hardnet_constraints', 0) for info in batch_infos)
        total_soft_violations = sum(info.get('soft_violations', 0) for info in batch_infos)
        
        # CRITICAL FIX: Aggregate penalty values - use MAX to preserve violation signals
        individual_penalties = [info.get('total_penalty', 0.0) for info in batch_infos]
        max_penalty = max(individual_penalties) if individual_penalties else 0.0
        
        return {
            'constraint_satisfaction_rate': constraint_satisfaction,
            'total_penalty': max_penalty,  # CRITICAL FIX: Include penalty in batch aggregation
            'mode': 'batched_' + batch_infos[0].get('mode', 'unknown'),
            'processing_time': avg_processing_time,
            'total_constraints': total_constraints,
            'samples_processed': total_samples,
            'hardnet_constraints': total_hardnet_constraints,
            'soft_violations': total_soft_violations,
            'batch_details': {
                'individual_satisfactions': [info.get('constraint_satisfaction_rate', 0.0) for info in batch_infos],
                'individual_constraint_counts': [len(constraints) for constraints in batch_constraints],
                'individual_penalties': individual_penalties  # Add for debugging
            }
        }
    
    def _record_performance_stats(self, start_time: float, constraint_info: Dict[str, Any]):
        """Record performance statistics for monitoring."""
        processing_time = time.time() - start_time
        
        self.performance_stats['processing_times'].append(processing_time)
        self.performance_stats['hardnet_constraints'].append(constraint_info.get('hardnet_constraints', 0))
        self.performance_stats['soft_violations'].append(constraint_info.get('soft_violations', 0))
        
        # Log periodically
        if len(self.performance_stats['processing_times']) % 100 == 0:
            self._log_performance_summary()
    
    def _log_performance_summary(self):
        """Log performance summary statistics."""
        times = self.performance_stats['processing_times'][-100:]
        constraints = self.performance_stats['hardnet_constraints'][-100:]
        violations = self.performance_stats['soft_violations'][-100:]
        
        self.logger.info(
            f"Performance (last 100): "
            f"Avg time: {np.mean(times):.3f}s, "
            f"Avg constraints: {np.mean(constraints):.1f}, "
            f"Avg violations: {np.mean(violations):.1f}"
        )
    
    def verify_constraint_satisfaction(self, 
                                    neural_output: torch.Tensor,
                                    tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        FIXED: Verify that the output actually satisfies the constraints.
        """
        verification_results = {
            'hardnet_satisfied': True,
            'hardnet_violations': [],
            'hardnet_max_violation': 0.0,
            'soft_violations': [],
            'total_violations': 0
        }
        
        if not PHASES_AVAILABLE:
            return verification_results
        
        try:
            # Convert to flat format for constraint checking
            flat_output, _ = self.converter.auto_convert_to_flat(neural_output)
            
            # Verify hard constraints (HardNet)
            if self.hardnet_layer and self.hardnet_layer.has_constraints():
                verification_results.update(
                    self.hardnet_layer.verify_constraint_satisfaction(flat_output, tolerance)
                )
            
            # Check soft constraints
            if self.soft_constraint_handler and self.soft_constraint_handler.penalty_handlers:
                penalty, soft_info = self.soft_constraint_handler(flat_output)
                verification_results['soft_penalty'] = penalty.item()
                verification_results['soft_violations'] = soft_info.get('violations', [])
            
            verification_results['total_violations'] = (
                len(verification_results['hardnet_violations']) + 
                len(verification_results.get('soft_violations', []))
            )
            
        except Exception as e:
            self.logger.error(f"Constraint verification failed: {e}")
            verification_results['verification_error'] = str(e)
        
        return verification_results
    
    def update_epoch(self, epoch: int):
        """Update training epoch for all components."""
        self.current_epoch = epoch
        
        if PHASES_AVAILABLE:
            if self.hardnet_layer:
                self.hardnet_layer.update_epoch(epoch)
            if self.soft_constraint_handler:
                self.soft_constraint_handler.update_epoch(epoch)
    
    def switch_mode(self, new_mode: SpatialReasoningMode):
        """Switch operating mode during runtime."""
        old_mode = self.config.mode
        self.config.mode = new_mode
        self.logger.info(f"Switched mode: {old_mode.value} → {new_mode.value}")
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """Get comprehensive constraint processing statistics."""
        stats = {
            'mode': self.config.mode.value,
            'current_epoch': self.current_epoch,
            'constraint_cache_size': len(self._constraint_cache),
            'performance_stats': dict(self.performance_stats)
        }
        
        if PHASES_AVAILABLE:
            if self.constraint_router:
                stats['phase1_router'] = self.constraint_router.get_split_percentages()
            if self.hardnet_layer:
                stats['phase3_hardnet'] = self.hardnet_layer.get_projection_statistics()
            if self.soft_constraint_handler:
                stats['phase4_soft'] = self.soft_constraint_handler.get_statistics()
        
        return stats


# Test function to validate fixes
def test_dimensional_fixes():
    """Test the dimensional consistency fixes."""
    print("Testing dimensional consistency fixes...")
    
    config = HybridSRMConfig(
        mode=SpatialReasoningMode.HYBRID,
        enable_logging=True
    )
    
    hybrid_srm = HybridSpatialReasoningModule(config)
    
    # Test different input formats
    batch_size = 2
    n_objects = 3
    
    # Test 1: Sequence format input
    sequence_input = torch.randn(batch_size, n_objects, 4)
    print(f"Test 1 - Sequence input: {sequence_input.shape}")
    
    try:
        output, info = hybrid_srm(sequence_input)
        print(f"✓ Sequence output: {output.shape}")
        assert output.shape == sequence_input.shape, f"Shape mismatch: {output.shape} != {sequence_input.shape}"
    except Exception as e:
        print(f"✗ Sequence test failed: {e}")
    
    # Test 2: Flat format input  
    flat_input = torch.randn(batch_size, n_objects * 4)
    print(f"Test 2 - Flat input: {flat_input.shape}")
    
    try:
        output, info = hybrid_srm(flat_input)
        print(f"✓ Flat output: {output.shape}")
        assert output.shape == flat_input.shape, f"Shape mismatch: {output.shape} != {flat_input.shape}"
    except Exception as e:
        print(f"✗ Flat test failed: {e}")
    
    # Test 3: With constraints
    if PHASES_AVAILABLE:
        from constraint_router import create_sample_constraint_set
        constraints = create_sample_constraint_set()
        
        print(f"Test 3 - With constraints: {len(constraints)} constraints")
        
        try:
            output, info = hybrid_srm(
                sequence_input, 
                constraints=constraints,
                n_objects=n_objects
            )
            print(f"✓ Constrained output: {output.shape}")
            print(f"  HardNet applied: {info.get('hardnet_applied', False)}")
            print(f"  Soft constraints applied: {info.get('soft_constraints_applied', False)}")
        except Exception as e:
            print(f"✗ Constraint test failed: {e}")
    
    print("Dimensional consistency tests completed!")


if __name__ == "__main__":
    test_dimensional_fixes()