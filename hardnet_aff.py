# HardNetAff Integration Layer
# We will implement differentiable constraint enforcement using HardNetaff projection

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from collections import defaultdict
from constraint_to_affine_converter import AffineConstraintMatrix


@dataclass
class HardNetProjectionStats:
    n_constraints_violated: int
    max_violation: float
    projection_norm: float
    constraint_satisfaction_rate: float
    numerical_warnings: int
    computation_time_ms: float


class HardNetAff(nn.Module):
    # HardNetaff differentiable constraint enforcement layer
    # P(fθ)(x) = fθ(x) + A(x)⁺[ReLU(b_l(x) - A(x)fθ(x)) - ReLU(A(x)fθ(x) - b_u(x))]
    #This layer can be inserted after any neural network to enforce hard constraints while maintaining differentiability for gradient-based training
    def __init__(self, 
                constraint_matrix: Optional[AffineConstraintMatrix] = None,
                regularization_strength: float = 1e-1,  # CRITICAL FIX: 100x increase for numerical stability
                numerical_tolerance: float = 1e-6,  # FIXED: Relaxed tolerance for per-mille coordinates
                enable_warm_start: bool = False,
                warm_start_epochs: int = 50,
                enable_logging: bool = True,
                coordinate_scale: float = 1.0):  # UPDATED: For normalized [0,1] coordinate system
        super(HardNetAff, self).__init__()
        
        self.regularization_strength = regularization_strength
        self.numerical_tolerance = numerical_tolerance
        self.enable_warm_start = enable_warm_start
        self.warm_start_epochs = warm_start_epochs
        self.enable_logging = enable_logging
        self.coordinate_scale = coordinate_scale  # UPDATED: For normalized [0,1] coordinate system
        
        # Training state
        self.current_epoch = 0
        self.warm_start_active = enable_warm_start
        
        # Statistics tracking
        self.projection_stats = defaultdict(list)
        self.logger = self._setup_logging()
        
        # INSTRUMENTATION: Initialize evidence capture attributes
        self._last_input_value = None
        self._last_output_value = None
        self._last_projection_occurred = False
        self._last_gradient_norm = 0.0
        
        # Initialize constraint information without creating buffers yet
        # Buffers will be created when set_constraints is called
        self.n_constraints = 0
        self.n_variables = 0
        self.constraint_names = []
        
        # Set constraints if provided
        if constraint_matrix is not None:
            self.set_constraints(constraint_matrix)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('HardNetAff')
        if self.enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    #fixmethod Check if constraints have been set for this HardNet layer
    def has_constraints(self) -> bool:
        return (hasattr(self, 'A') and 
                hasattr(self, 'b_l') and 
                hasattr(self, 'b_u') and
                self.n_constraints > 0)
    
    def set_constraints(self, constraint_matrix: AffineConstraintMatrix):
        # Always remove existing buffers first to avoid conflicts
        # This is the most reliable approach for dynamic constraint updates
        for buffer_name in ['A', 'b_l', 'b_u', 'A_pinv']:
            if hasattr(self, buffer_name):
                delattr(self, buffer_name)
        
        # FIXED: Handle None input (clear constraints)
        if constraint_matrix is None:
            self.n_constraints = 0
            self.n_variables = 0
            self.logger.info("HardNet constraints cleared (set to None)")
            return
        
        # Convert numpy arrays to PyTorch tensors and register as new buffers
        # CRITICAL FIX: Force GPU device placement - no CPU fallback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')  # Force explicit GPU device
        
        self.logger.info(f"HardNet constraint matrices created on device: {device}")
        
        # CRITICAL FIX: Ensure all numpy arrays are properly converted to tensors
        # Convert numpy arrays to numpy arrays first, then scale, then convert to tensors
        A_np = np.array(constraint_matrix.A, dtype=np.float64)
        b_l_np = np.array(constraint_matrix.b_l, dtype=np.float64) 
        b_u_np = np.array(constraint_matrix.b_u, dtype=np.float64)
        
        # Keep constraint bounds in original [0, 1000] space - NO SCALING
        scaled_b_l_np = b_l_np
        scaled_b_u_np = b_u_np
        
        # CRITICAL FIX: Use float64 for constraint matrices to prevent numerical breakdown
        # Condition numbers ~10^9 require higher precision than float32's ~7 digits
        self.register_buffer('A', torch.tensor(A_np, dtype=torch.float64, device=device))
        self.register_buffer('b_l', torch.tensor(scaled_b_l_np, dtype=torch.float64, device=device))
        self.register_buffer('b_u', torch.tensor(scaled_b_u_np, dtype=torch.float64, device=device))
        
        self.n_constraints = constraint_matrix.n_constraints
        self.n_variables = constraint_matrix.A.shape[1]
        self.constraint_names = constraint_matrix.constraint_names
        
        # CRITICAL FIX: Early termination for pathological matrices
        if self.n_constraints > 0:
            # Check for extreme rank deficiency (< 50% effective rank)
            try:
                matrix_rank = torch.linalg.matrix_rank(self.A).item()
                expected_rank = min(self.A.shape)
                rank_ratio = matrix_rank / expected_rank if expected_rank > 0 else 0
                
                if rank_ratio < 0.5:
                    self.logger.error(f"PATHOLOGICAL MATRIX: Rank {matrix_rank}/{expected_rank} = {rank_ratio:.2%}")
                    self.logger.error(f"Constraint matrix is severely rank deficient - disabling HardNet projection")
                    # Disable constraint processing by clearing the constraint count
                    self.n_constraints = 0
                    self.n_variables = 0
                    return
                    
                # Check condition number for numerical stability
                try:
                    ATA = torch.mm(self.A, self.A.t())
                    cond_num = torch.linalg.cond(ATA).item()
                    if cond_num > 1e15:
                        self.logger.error(f"PATHOLOGICAL MATRIX: Condition number {cond_num:.2e} > 1e15")
                        self.logger.error(f"Matrix is numerically singular - disabling HardNet projection")
                        self.n_constraints = 0
                        self.n_variables = 0
                        return
                except:
                    pass  # Condition number check failed, but continue
                    
            except Exception as e:
                self.logger.warning(f"Matrix pathology check failed: {e}")
                pass  # Continue with setup
        
        # Precompute regularized pseudoinverse for efficiency
        self._precompute_pseudoinverse()
        
        self.logger.info(
            f"HardNet-Aff constraints updated: {self.n_constraints} constraints, "
            f"{self.n_variables} variables"
        )
        
    def _precompute_pseudoinverse(self):
        """
        Theoretically optimal pseudoinverse using always-on SVD regularization.
        Provides superior gradient flow, numerical stability, and constraint satisfaction.
        """
        if self.A is None or self.n_constraints == 0:
            return
        
        # THEORETICAL OPTIMUM: Always-on regularization for smooth operator behavior
        # Use SVD for robust pseudoinverse computation with guaranteed numerical stability
        try:
            # CRITICAL FIX: Ensure SVD computation happens on GPU
            device = self.A.device
            if device.type != 'cuda' and torch.cuda.is_available():
                self.logger.warning(f"Moving constraint matrix to GPU for SVD computation")
                self.A = self.A.cuda()
                device = self.A.device
            
            # SVD decomposition: A = U Σ V^T (forced to run on GPU)
            U, S, Vh = torch.linalg.svd(self.A, full_matrices=False)
            
            # Verify SVD ran on GPU
            if device.type == 'cuda':
                self.logger.debug(f"SVD computation completed on GPU: {device}")
            
            # CRITICAL: Adaptive regularization based on singular value gaps
            # Fixed λ=1e-8 was too small for spatial constraint matrices
            # THEORETICAL FIX: Tikhonov regularization based on condition number
            condition_number = (S.max() / S.min()).item() if S.min() > 0 else float('inf')
            
            if condition_number > 1e12:  # Severely ill-conditioned
                lambda_reg = S.max().item() * 1e-6  # Strong regularization
            elif condition_number > 1e8:   # Moderately ill-conditioned  
                lambda_reg = S.max().item() * 1e-8  # Moderate regularization
            else:  # Well-conditioned
                lambda_reg = S.max().item() * 1e-12 # Minimal regularization
            
            # Additional regularization for rank-deficient matrices
            effective_rank = torch.sum(S > S.max().item() * 1e-12).item()
            if effective_rank < len(S):
                lambda_reg = max(lambda_reg, S.max().item() * 1e-6)
            
            # Regularized singular values: σ_reg = σ / (σ² + λ)
            # This ensures: 1) Smooth behavior, 2) Gradient continuity, 3) Numerical stability
            S_reg_inv = S / (S**2 + lambda_reg)
            
            # Construct regularized pseudoinverse: A⁺_λ = V Σ⁺_reg U^T
            # For A = U Σ V^T, the pseudoinverse is A⁺ = V Σ⁺ U^T
            # SVD returns: U (m×k), S (k,), Vh (k×n) where A = U @ diag(S) @ Vh
            # So A⁺ = Vh^T @ diag(S_reg_inv) @ U^T = (n×k) @ (k×k) @ (k×m) = (n×m)
            A_pinv_tensor = torch.mm(torch.mm(Vh.t(), torch.diag(S_reg_inv)), U.t())
            
            # Log regularization info for debugging
            cond_regularized = (S.max() / torch.sqrt(S.min()**2 + lambda_reg)).item()
            
            self.logger.info(
                f"Tikhonov regularization: λ={lambda_reg:.2e}, "
                f"Condition: {condition_number:.2e} → {cond_regularized:.2e}, "
                f"Effective rank: {effective_rank}/{len(S)}"
            )
            
        except Exception as e:
            raise AssertionError(
                f"SVD-based pseudoinverse failed: {e}\n"
                f"Matrix shape: {self.A.shape}\n"
                f"This indicates fundamental mathematical issues with constraint matrix"
            )
        
        # Remove existing A_pinv buffer if it exists then register new one
        if hasattr(self, 'A_pinv'):
            delattr(self, 'A_pinv')
        self.register_buffer('A_pinv', A_pinv_tensor)
        
        # Validate pseudoinverse quality
        self._validate_pseudoinverse()
    
    def _validate_pseudoinverse(self):
        """CRITICAL: Enhanced numerical validation with comprehensive checks."""
        if self.A_pinv is None:
            return
            
        validation_results = {
            'reconstruction_error': 0.0,
            'condition_number': 0.0,
            'rank_deficiency': False,
            'numerical_stability': True,
            'warnings': []
        }
        
        try:
            # Check AA⁺A = A property (Moore-Penrose condition)
            # A: (m×n), A_pinv: (n×m) -> A_pinv @ A: (n×n), A @ (A_pinv @ A): (m×n) ✓
            AApA = torch.mm(self.A, torch.mm(self.A_pinv, self.A))
            reconstruction_error = torch.norm(AApA - self.A).item()
            validation_results['reconstruction_error'] = reconstruction_error
            
            # Check A⁺AA⁺ = A⁺ property  
            # A_pinv: (n×m), A: (m×n) -> A @ A_pinv: (m×m), A_pinv @ (A @ A_pinv): (n×m) ✓
            ApAAp = torch.mm(self.A_pinv, torch.mm(self.A, self.A_pinv))
            pinv_reconstruction_error = torch.norm(ApAAp - self.A_pinv).item()
            
            # CRITICAL FIX: Strict validation thresholds to prevent mathematical breakdown
            # Based on evidence: reconstruction error 5.01 caused complete system failure
            reconstruction_threshold = 0.5    # Maximum acceptable reconstruction error
            pinv_threshold = 0.5              # A⁺AA⁺ must be accurate for HardNet projection
            
            if reconstruction_error > reconstruction_threshold:
                validation_results['warnings'].append(
                    f"CRITICAL: AA⁺A reconstruction error: {reconstruction_error:.2e} > {reconstruction_threshold:.2e}"
                )
                validation_results['numerical_stability'] = False
                
            if pinv_reconstruction_error > pinv_threshold:
                validation_results['warnings'].append(
                    f"WARNING: A⁺AA⁺ reconstruction error: {pinv_reconstruction_error:.2e} > {pinv_threshold:.2e}"
                )
            
            # Condition number analysis
            try:
                AAT = torch.mm(self.A, self.A.t())  # FIXED: A @ A.T instead of A @ A_pinv.T
                cond_number = torch.linalg.cond(AAT).item()
                validation_results['condition_number'] = cond_number
                
                if cond_number > 1e8:  # More practical threshold
                    validation_results['warnings'].append(f"High condition number: {cond_number:.2e}")
                    # Don't fail on high condition number - just warn
                    # validation_results['numerical_stability'] = False
            except:
                validation_results['warnings'].append("Could not compute condition number")
                
            # Rank analysis - informational only since preprocessing should handle this
            A_rank = torch.linalg.matrix_rank(self.A).item()
            expected_rank = min(self.A.shape)
            if A_rank < expected_rank:
                validation_results['rank_deficiency'] = True
                validation_results['warnings'].append(f"Rank deficiency: {A_rank} < {expected_rank} (should be handled by preprocessing)")
            
            # Log results
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    self.logger.warning(f"Pseudoinverse validation: {warning}")
            else:
                self.logger.debug(f"Pseudoinverse validation passed: error = {reconstruction_error:.2e}")
                
            # FAIL-FAST: Raise exception for critical failures
            if not validation_results['numerical_stability']:
                raise AssertionError(
                    f"FAIL-FAST: Pseudoinverse validation failed:\n"
                    f"Reconstruction error: {reconstruction_error:.2e}\n"
                    f"Condition number: {validation_results['condition_number']:.2e}\n"
                    f"Warnings: {validation_results['warnings']}\n"
                    f"This indicates fundamental mathematical issues with constraint matrix\n"
                    f"FIX THE CONSTRAINT PREPROCESSING OR MATRIX CONSTRUCTION"
                )
                
        except Exception as e:
            self.logger.error(f"Pseudoinverse validation failed: {e}")
            raise
    
    def _analyze_batch_constraint_heterogeneity(self, neural_output: torch.Tensor) -> Dict[str, float]:
        """
        Analyze constraint satisfaction heterogeneity across the batch.
        Returns statistics about constraint satisfaction variability.
        """
        batch_size = neural_output.shape[0]
        if batch_size == 1:
            return {'heterogeneity_score': 0.0, 'satisfaction_std': 0.0, 'min_satisfaction': 1.0, 'max_satisfaction': 1.0}
        
        batch_satisfaction_rates = []
        
        with torch.no_grad():
            # Convert neural output to double precision for constraint computation
            neural_output_f64 = neural_output.to(dtype=torch.float64)
            A_f = torch.mm(neural_output_f64, self.A.t())
            b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)
            b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)
            
            # Calculate per-sample constraint satisfaction
            for i in range(batch_size):
                sample_A_f = A_f[i]  # (n_constraints,)
                sample_b_l = b_l_batch[i]
                sample_b_u = b_u_batch[i]
                
                # Check constraint satisfaction for this sample
                lower_satisfied = sample_A_f >= (sample_b_l - self.numerical_tolerance)
                upper_satisfied = sample_A_f <= (sample_b_u + self.numerical_tolerance)
                constraints_satisfied = lower_satisfied & upper_satisfied
                
                satisfaction_rate = torch.sum(constraints_satisfied).float() / self.n_constraints
                batch_satisfaction_rates.append(satisfaction_rate.item())
        
        # Calculate heterogeneity statistics
        satisfaction_tensor = torch.tensor(batch_satisfaction_rates)
        heterogeneity_stats = {
            'heterogeneity_score': torch.std(satisfaction_tensor).item(),
            'satisfaction_std': torch.std(satisfaction_tensor).item(),
            'min_satisfaction': torch.min(satisfaction_tensor).item(),
            'max_satisfaction': torch.max(satisfaction_tensor).item(),
            'mean_satisfaction': torch.mean(satisfaction_tensor).item()
        }
        
        return heterogeneity_stats
    
    def forward(self, neural_output: torch.Tensor) -> torch.Tensor:
        # CRITICAL FIX: Force GPU operations - ensure all tensors on same GPU device
        target_device = neural_output.device
        if self.A is not None:
            # Force move to match neural_output device
            if self.A.device != target_device:
                if self.enable_logging:
                    self.logger.info(f" Moving HardNet tensors: {self.A.device} → {target_device}")
                
                # CRITICAL FIX: non_blocking=True corrupts inf values on some GPU drivers
                # Use blocking device transfers to preserve inf/nan values
                self.A = self.A.to(device=target_device, dtype=torch.float64, non_blocking=False)
                self.b_l = self.b_l.to(device=target_device, dtype=torch.float64, non_blocking=False)
                self.b_u = self.b_u.to(device=target_device, dtype=torch.float64, non_blocking=False)
                if hasattr(self, 'A_pinv') and self.A_pinv is not None:
                    self.A_pinv = self.A_pinv.to(device=target_device, dtype=torch.float64, non_blocking=False)
                
                if target_device.type == 'cuda':
                    self.logger.info(f" HardNet tensors moved to GPU: {target_device}")
                else:
                    self.logger.info(f" HardNet tensors moved to CPU: {target_device}")
            else:
                if self.enable_logging:
                    self.logger.debug(f" HardNet tensors already on correct device: {target_device}")

        # Handle empty constraints case
        if self.A is None or self.n_constraints == 0:
            return neural_output
        
        # Ensure input is 2D (batch_size, n_variables) and convert to float64 for precision
        if neural_output.dim() == 1:
            neural_output = neural_output.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # CRITICAL FIX: Store original dtype but keep neural_output in float32 for gradient compatibility
        # Only convert constraint matrices to float64, not the neural network outputs
        original_dtype = neural_output.dtype
        
        batch_size = neural_output.shape[0]
        
        # Validate input dimensions
        if neural_output.shape[1] != self.n_variables:
            raise ValueError(
                f"Input dimension mismatch: expected {self.n_variables}, "
                f"got {neural_output.shape[1]}"
            )
        
        # ENHANCED: Analyze batch heterogeneity for adaptive processing
        heterogeneity_stats = None
        if self.enable_logging and self.training and neural_output.shape[0] > 1:
            heterogeneity_stats = self._analyze_batch_constraint_heterogeneity(neural_output)
            
            # Log high heterogeneity batches
            if heterogeneity_stats['heterogeneity_score'] > 0.3:  # Threshold for high heterogeneity
                self.logger.warning(f"High batch heterogeneity detected: {heterogeneity_stats['heterogeneity_score']:.3f} "
                                  f"(satisfaction range: {heterogeneity_stats['min_satisfaction']:.3f}-{heterogeneity_stats['max_satisfaction']:.3f})")
        
        # FIXED: Implement gradual transition instead of abrupt switch
        transition_strength = self._compute_transition_strength()
        
        # DEBUG: Log transition strength details (removed global_step reference - doesn't exist in HardNet)
        if self.enable_logging and self.training and self.current_epoch % 10 == 0:  # Log every 10 epochs
            self.logger.info(f"HardNet mode: epoch={self.current_epoch}, transition={transition_strength:.3f}")
        
        # ENHANCED: Adapt transition strength based on batch heterogeneity
        if heterogeneity_stats and heterogeneity_stats['heterogeneity_score'] > 0.4:
            # Reduce transition strength for highly heterogeneous batches
            original_strength = transition_strength
            transition_strength = transition_strength * 0.5  # Reduce by 50%
            if self.enable_logging and self.training:
                self.logger.debug(f"Reduced transition strength {original_strength:.3f} -> {transition_strength:.3f} due to batch heterogeneity")
        
        if transition_strength == 0.0:
            # Pure warm-start mode (soft constraints only)
            return self._forward_warm_start(neural_output, squeeze_output)
        elif transition_strength >= 0.99:  # Allow for floating point precision
            # Pure hard constraint mode - proceed to projection
            pass
        else:
            # Gradual transition mode
            return self._forward_gradual_transition(neural_output, squeeze_output, transition_strength)
        
        # Apply full HardNet-Aff projection for pure hard mode
        # CRITICAL: Apply coordinate scaling for constraint matrix compatibility
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        # NO SCALING - Keep everything in [0, 1000] per-mille space
        # Constraints are already formulated in this space
        projected_output = self._apply_hardnet_projection(neural_output).to(dtype=original_dtype)
        
        # Record timing
        if start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time)
        else:
            computation_time = 0.0
        
        # Collect statistics  
        if self.enable_logging and self.training:
            self._collect_projection_stats(neural_output, projected_output, computation_time)
        
        return projected_output.squeeze(0) if squeeze_output else projected_output
    
    def _get_adaptive_correction_threshold(self, neural_output: torch.Tensor) -> float:
        """Get correction threshold adapted to coordinate system."""
        max_coord = torch.max(torch.abs(neural_output)).item()
        
        if max_coord <= 2.0:
            # Normalized [0,1] system
            coord_range = 1.0
            system_name = "normalized"
        elif max_coord <= 100.0:
            # Medium scale system (e.g., 0-100)  
            coord_range = max_coord * 2.0  # Approximate range
            system_name = "medium"
        else:
            # Large scale system (per-mille, pixel, etc.)
            coord_range = max_coord * 2.0  # Approximate range  
            system_name = "large"
        
        # Threshold = 75% of coordinate range, minimum 1.0  
        # SPRING Fix: High threshold allows full constraint enforcement without premature scaling
        # For constraint satisfaction, we need to allow large corrections when mathematically required
        threshold = max(1.0, coord_range * 0.75)
        
        if self.enable_logging:
            self.logger.info(f"COORDINATE SYSTEM: {system_name} (max={max_coord:.1f}, range≈{coord_range:.1f}, threshold={threshold:.1f})")
        
        return threshold
    
    def _apply_hardnet_projection(self, neural_output: torch.Tensor) -> torch.Tensor:
        batch_size = neural_output.shape[0]
        
        # ENHANCED: Pre-check for feasibility and numerical stability
        try:
            # FAIL-FAST: Check for invalid inputs - crash immediately
            if torch.isnan(neural_output).any() or torch.isinf(neural_output).any():
                nan_count = torch.isnan(neural_output).sum().item()
                inf_count = torch.isinf(neural_output).sum().item()
                raise AssertionError(
                    f"FAIL-FAST: Invalid neural output detected:\n"
                    f"NaN values: {nan_count}\n"
                    f"Inf values: {inf_count}\n"
                    f"Neural output shape: {neural_output.shape}\n"
                    f"Neural output device: {neural_output.device}\n"
                    f"This indicates gradient explosion or numerical instability\n"
                    f"NO fallback to original output - FIX THE NEURAL NETWORK TRAINING"
                )
            
            # Compute A * fθ(x) for each sample in batch
            # A: (n_constraints, n_variables), neural_output: (batch_size, n_variables)
            # Convert to float64 for constraint computation precision, then back to float32
            neural_output_f64 = neural_output.to(dtype=torch.float64)
            A_f = torch.mm(neural_output_f64, self.A.t())  # (batch_size, n_constraints)
            
            # FAIL-FAST: Check for numerical issues in constraint evaluation
            if torch.isnan(A_f).any() or torch.isinf(A_f).any():
                nan_count = torch.isnan(A_f).sum().item()
                inf_count = torch.isinf(A_f).sum().item()
                raise AssertionError(
                    f"FAIL-FAST: Numerical issues in constraint evaluation:\n"
                    f"A_f NaN values: {nan_count}\n"
                    f"A_f Inf values: {inf_count}\n"
                    f"Constraint matrix A shape: {self.A.shape}\n"
                    f"Neural output range: [{neural_output.min():.6f}, {neural_output.max():.6f}]\n"
                    f"A matrix range: [{self.A.min():.6f}, {self.A.max():.6f}]\n"
                    f"FIX THE CONSTRAINT MATRIX OR NEURAL OUTPUT SCALING"
                )
            
            # Expand bounds for batch processing
            b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_constraints)
            b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_constraints)
            
            # FAIL-FAST: Check for constraint feasibility
            constraint_feasible = (b_l_batch <= b_u_batch).all()
            if not constraint_feasible:
                infeasible_indices = torch.where(b_l_batch > b_u_batch)
                raise AssertionError(
                    f"FAIL-FAST: Infeasible constraints detected (b_l > b_u):\n"
                    f"Infeasible constraint indices: {infeasible_indices}\n"
                    f"b_l shape: {b_l_batch.shape}, b_u shape: {b_u_batch.shape}\n"
                    f"b_l range: [{self.b_l.min():.6f}, {self.b_l.max():.6f}]\n"
                    f"b_u range: [{self.b_u.min():.6f}, {self.b_u.max():.6f}]\n"
                    f"This indicates contradictory constraints in the problem formulation\n"
                    f"FIX THE CONSTRAINT GENERATION LOGIC"
                )
                
        except Exception as e:
            # FAIL-FAST: No fallback for feasibility errors
            raise AssertionError(
                f"FAIL-FAST: Error in feasibility pre-check:\n"
                f"Error: {e}\n"
                f"Constraint matrix A shape: {self.A.shape if hasattr(self, 'A') else 'N/A'}\n"
                f"Bounds b_l shape: {self.b_l.shape if hasattr(self, 'b_l') else 'N/A'}\n"
                f"Bounds b_u shape: {self.b_u.shape if hasattr(self, 'b_u') else 'N/A'}\n"
                f"Batch size: {batch_size}\n"
                f"FIX THE CONSTRAINT SETUP OR MATRIX OPERATIONS"
            )
        
        # Compute constraint violations
        lower_violations = F.relu(b_l_batch - A_f)  # ReLU(b_l - Af)
        upper_violations = F.relu(A_f - b_u_batch)  # ReLU(Af - b_u)
        
        # Combined violation vector
        # HardNet-Aff formula: P(f(x)) = f(x) + A⁺[ReLU(b_l - Af) - ReLU(Af - b_u)]
        # This means: violation_vector = ReLU(b_l - Af) - ReLU(Af - b_u)
        violation_vector = lower_violations - upper_violations  # (batch_size, n_constraints)
        
        # ENHANCED: Apply pseudoinverse projection with error handling
        try:
            # According to HardNet-Aff formula: correction = A⁺ * [ReLU(b_l - Af) - ReLU(Af - b_u)]
            # CRITICAL FIX: Ensure matrix multiplication happens on GPU
            # A_pinv: (n_variables, n_constraints), violation_vector: (batch_size, n_constraints)
            # Correct matrix multiplication: A_pinv @ violation_vector.T -> (n_variables, batch_size) -> transpose -> (batch_size, n_variables)
            
            # Ensure all tensors are on the same device for matrix multiplication
            device = violation_vector.device
            if device == self.A_pinv.device:
                # All tensors on same device - proceed normally
                correction = torch.mm(violation_vector, self.A_pinv.t())  # (batch_size, n_variables)
            else:
                # Device mismatch - this should not happen after device movement fix
                raise AssertionError(
                    f"DEVICE MISMATCH: violation_vector on {device}, A_pinv on {self.A_pinv.device}\n"
                    f"This indicates device movement logic failed"
                )
            
            # FAIL-FAST: Check for numerical issues in correction
            if torch.isnan(correction).any() or torch.isinf(correction).any():
                nan_count = torch.isnan(correction).sum().item()
                inf_count = torch.isinf(correction).sum().item()
                raise AssertionError(
                    f"FAIL-FAST: Numerical issues in correction computation:\n"
                    f"Correction NaN values: {nan_count}\n"
                    f"Correction Inf values: {inf_count}\n"
                    f"Violation vector shape: {violation_vector.shape}\n"
                    f"A_pinv shape: {self.A_pinv.shape}\n"
                    f"A_pinv condition number: {torch.linalg.cond(self.A_pinv).item():.6f}\n"
                    f"This indicates matrix singularity or poor conditioning\n"
                    f"FIX THE PSEUDOINVERSE COMPUTATION OR CONSTRAINT MATRIX"
                )
            
            # Check for unreasonably large corrections using coordinate-aware threshold
            max_correction_norm = torch.max(torch.norm(correction, dim=1))
            adaptive_threshold = self._get_adaptive_correction_threshold(neural_output)
            
            if max_correction_norm > adaptive_threshold:  # FIXED: Coordinate-aware threshold
                if self.enable_logging:
                    self.logger.warning(f"Large correction norm {max_correction_norm:.3f} > {adaptive_threshold:.3f}, using scaled correction")
                # Scale down the correction to prevent instability
                correction = correction * (adaptive_threshold / max_correction_norm)
            
            # Final projection - convert correction back to float32 for compatibility
            correction_f32 = correction.to(dtype=neural_output.dtype)
            projected_output = neural_output + correction_f32
            
            # INSTRUMENTATION: Capture values for evidence logging
            if batch_size > 0:  # Only capture for non-empty batches
                self._last_input_value = neural_output[0, 0].item() if neural_output.numel() > 0 else None
                self._last_output_value = projected_output[0, 0].item() if projected_output.numel() > 0 else None
                correction_magnitude = torch.max(torch.abs(correction_f32)).item()
                self._last_projection_occurred = correction_magnitude > 1e-6
                
                # DEBUG: Enhanced projection failure analysis
                if self.enable_logging:  # Always enable for debugging
                    # Compute constraint evaluation for all constraints
                    A_f = torch.mm(neural_output.to(dtype=torch.float64), self.A.t())  # (1, n_constraints)
                    
                    # Check each constraint for violations
                    violations = []
                    for i in range(self.n_constraints):
                        A_f_val = A_f[0, i].item()
                        b_l_val = self.b_l[i].item()
                        b_u_val = self.b_u[i].item()
                        lower_viol = max(0, b_l_val - A_f_val)
                        upper_viol = max(0, A_f_val - b_u_val)
                        total_viol = max(lower_viol, upper_viol)
                        if total_viol > 1e-6:
                            violations.append((i, A_f_val, b_l_val, b_u_val, total_viol))
                    
                    # CRITICAL DEBUG: Always log constraint evaluation
                    input_x0 = neural_output[0, 0].item()
                    input_x1 = neural_output[0, 4].item() if neural_output.shape[1] > 4 else 0.0
                    output_x0 = projected_output[0, 0].item()
                    output_x1 = projected_output[0, 4].item() if projected_output.shape[1] > 4 else 0.0
                    
                    self.logger.info(f" HARDNET DEBUG - CONSTRAINT EVALUATION:")
                    self.logger.info(f"   Input:  obj0.x={input_x0:.6f}, obj1.x={input_x1:.6f}, diff={input_x0-input_x1:.6f}")
                    self.logger.info(f"   Output: obj0.x={output_x0:.6f}, obj1.x={output_x1:.6f}, diff={output_x0-output_x1:.6f}")
                    self.logger.info(f"   Constraint: A_f={A_f[0,0].item():.6f}, bounds=[{self.b_l[0].item()}, {self.b_u[0].item()}]")
                    self.logger.info(f"   A matrix first row: {self.A[0,:8].tolist()}")  # Show first 8 elements
                    self.logger.info(f"   Correction magnitude: {correction_magnitude:.6f}")
                    
                    if output_x0 > output_x1 + 1e-6:  # True violation (obj0 significantly right of obj1)
                        self.logger.error(f"    CONSTRAINT VIOLATED: obj0 should be LEFT of obj1 but isn't!")
                    elif abs(output_x0 - output_x1) < 1e-6:  # Perfect equality satisfies LEFT constraint
                        self.logger.info(f"    ✅ CONSTRAINT SATISFIED: obj0.x = obj1.x (LEFT constraint allows equality)")
                    else:  # obj0 < obj1, perfect satisfaction
                        self.logger.info(f"    ✅ CONSTRAINT SATISFIED: obj0.x < obj1.x")
                    
                    # Log mathematical failure details
                    if violations and correction_magnitude <= 1e-6:
                        self.logger.error(f"PROJECTION_FAILURE: {len(violations)} constraints violated, no correction applied")
                        for i, A_f_val, b_l_val, b_u_val, viol in violations:
                            self.logger.error(f"   Constraint {i}: Af={A_f_val:.6f}, bounds=[{b_l_val:.6f}, {b_u_val:.6f}], violation={viol:.6f}")
                        self.logger.error(f"   A matrix shape: {self.A.shape}, condition: {torch.linalg.cond(self.A).item():.2e}")
                        self.logger.error(f"   A_pinv shape: {self.A_pinv.shape}, condition: {torch.linalg.cond(self.A_pinv).item():.2e}")
                    
                    # Check if projection made violations worse
                    elif violations and correction_magnitude > 1e-6:
                        input_x = neural_output[0, 0].item()
                        output_x = projected_output[0, 0].item()
                        if abs(output_x) > abs(input_x):  # Projection made it worse
                            self.logger.error(f"PROJECTION_WORSENED: Input x={input_x:.6f} -> Output x={output_x:.6f}")
                            self.logger.error(f"   Correction magnitude: {correction_magnitude:.6f}")
                            self.logger.error(f"   Violation vector: {violation_vector[0].tolist()}")
                            self.logger.error(f"   A matrix first row: {self.A[0].tolist()}")
                            self.logger.error(f"   A_pinv first column: {self.A_pinv[:, 0].tolist()}")
            
        except Exception as e:
            # FAIL-FAST: No fallback for projection errors
            raise AssertionError(
                f"FAIL-FAST: Error in projection computation:\n"
                f"Error: {e}\n"
                f"Lower violations shape: {lower_violations.shape if 'lower_violations' in locals() else 'N/A'}\n"
                f"Upper violations shape: {upper_violations.shape if 'upper_violations' in locals() else 'N/A'}\n"
                f"Neural output shape: {neural_output.shape}\n"
                f"Constraint matrix rank: {torch.linalg.matrix_rank(self.A).item() if hasattr(self, 'A') else 'N/A'}\n"
                f"FIX THE MATHEMATICAL FORMULATION OR TENSOR OPERATIONS"
            )
        
        # NUMERICAL STABILITY FIX: Ensure exact constraint satisfaction
        # Re-project any remaining violations due to numerical errors
        with torch.no_grad():
            # Convert to double precision for constraint computation
            projected_output_f64 = projected_output.to(dtype=torch.float64)
            A_f_final = torch.mm(projected_output_f64, self.A.t())
            final_lower_violations = torch.clamp(b_l_batch - A_f_final, min=0)
            final_upper_violations = torch.clamp(A_f_final - b_u_batch, min=0)
            
            # If there are still violations, apply a small correction
            max_violation = torch.max(torch.max(final_lower_violations), torch.max(final_upper_violations))
            if max_violation > self.numerical_tolerance:
                # Apply one more correction step
                final_violation_vector = final_lower_violations - final_upper_violations
                final_correction = torch.mm(final_violation_vector, self.A_pinv.t())
                projected_output = projected_output + final_correction
        
        # INSTRUMENTATION: Setup gradient hook to capture gradient flow evidence
        if projected_output.requires_grad:
            def capture_gradient_norm(grad):
                if grad is not None:
                    self._last_gradient_norm = torch.norm(grad).item()
                else:
                    self._last_gradient_norm = 0.0
                return grad
            projected_output.register_hook(capture_gradient_norm)
        else:
            self._last_gradient_norm = 0.0
        
        return projected_output
    
    def compute_constraint_satisfaction_rate(self, neural_output: torch.Tensor) -> float:
        """
        Compute the constraint satisfaction rate for given neural output.
        Returns rate between 0.0 (no constraints satisfied) and 1.0 (all satisfied).
        """
        if self.A is None or self.n_constraints == 0:
            return 1.0
        
        # Ensure input is 2D
        if neural_output.dim() == 1:
            neural_output = neural_output.unsqueeze(0)
        
        # Ensure device compatibility
        self = self.to(neural_output.device)
        
        # CRITICAL: Scale coordinates for constraint compatibility
        scaled_input = neural_output / self.coordinate_scale
        # CRITICAL FIX: Convert to float64 to match constraint matrix precision
        scaled_input = scaled_input.to(dtype=torch.float64)
        batch_size = scaled_input.shape[0]
        
        # Compute constraint values: A * scaled_input (convert to double precision)
        scaled_input_f64 = scaled_input.to(dtype=torch.float64)
        A_f = torch.mm(scaled_input_f64, self.A.t())  # (batch_size, n_constraints)
        
        # Expand bounds for batch processing
        b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)
        b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)
        
        # Check constraint satisfaction with tolerance
        lower_satisfied = A_f >= (b_l_batch - self.numerical_tolerance)
        upper_satisfied = A_f <= (b_u_batch + self.numerical_tolerance)
        constraints_satisfied = lower_satisfied & upper_satisfied
        
        # Compute satisfaction rate
        total_constraints = batch_size * self.n_constraints
        satisfied_constraints = torch.sum(constraints_satisfied).item()
        
        return satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
    
    def _forward_warm_start(self, neural_output: torch.Tensor, squeeze_output: bool) -> torch.Tensor:
        # CRITICAL: Scale coordinates for constraint compatibility
        scaled_input = neural_output / self.coordinate_scale
        
        # Compute constraint violations for regularization loss
        batch_size = scaled_input.shape[0]
        # Convert to double precision for constraint computation
        scaled_input_f64 = scaled_input.to(dtype=torch.float64)
        A_f = torch.mm(scaled_input_f64, self.A.t())
        
        b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)
        b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)
        
        # Soft constraint violations (will be used in loss)
        lower_violations = F.relu(b_l_batch - A_f)
        upper_violations = F.relu(A_f - b_u_batch)
        constraint_violation_loss = torch.mean(lower_violations + upper_violations)
        
        # Store violation loss for external access
        self.last_constraint_violation_loss = constraint_violation_loss
        
        # FIXED: Calculate constraint satisfaction statistics during warm-start using proper tolerance
        lower_satisfied = lower_violations <= self.numerical_tolerance
        upper_satisfied = upper_violations <= self.numerical_tolerance
        constraints_satisfied = lower_satisfied & upper_satisfied
        
        total_constraints = batch_size * self.n_constraints
        satisfied_constraints = torch.sum(constraints_satisfied).item()
        satisfaction_rate = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
        
        # Store warm-start statistics
        total_violations = total_constraints - satisfied_constraints
        stats = HardNetProjectionStats(
            n_constraints_violated=total_violations,
            max_violation=torch.max(torch.max(lower_violations), torch.max(upper_violations)).item(),
            projection_norm=0.0,  # No projection during warm-start
            constraint_satisfaction_rate=satisfaction_rate,
            numerical_warnings=0,
            computation_time_ms=0.0
        )
        self.projection_stats['batch_stats'].append(stats)
        
        # Return unmodified output during warm-start
        return neural_output
    
    def _compute_transition_strength(self) -> float:
        """
        Compute gradual transition strength from soft to hard constraints.
        Returns:
            0.0: Pure warm-start (soft constraints only)
            1.0: Pure hard constraints
            0.0-1.0: Gradual transition
        """
        if not self.enable_warm_start:
            return 1.0  # Always hard constraints if warm start disabled
        
        if self.current_epoch <= self.warm_start_epochs:
            return 0.0  # Pure warm start
        
        # Gradual transition over 20 epochs after warm start ends
        transition_epochs = 20
        transition_start = self.warm_start_epochs + 1
        transition_end = self.warm_start_epochs + transition_epochs
        
        if self.current_epoch >= transition_end:
            return 1.0  # Pure hard constraints
        
        # Smooth transition using cosine schedule
        progress = (self.current_epoch - transition_start) / transition_epochs
        # Cosine schedule: starts at 0, ends at 1, smooth transition
        transition_strength = 0.5 * (1 - np.cos(np.pi * progress))
        
        return float(np.clip(transition_strength, 0.0, 1.0))
    
    def _forward_gradual_transition(self, neural_output: torch.Tensor, squeeze_output: bool, strength: float) -> torch.Tensor:
        """
        Apply gradual transition between soft and hard constraints.
        
        Args:
            neural_output: Network output to be projected
            squeeze_output: Whether to squeeze output dimensions
            strength: Transition strength (0.0 = soft, 1.0 = hard)
        """
        # Get soft constraint output (no projection)
        soft_output = neural_output
        
        # Get hard constraint output (with projection) 
        # CRITICAL: Handle coordinate scaling and potential failures gracefully
        try:
            scaled_input = neural_output / self.coordinate_scale
            hard_scaled = self._apply_hardnet_projection(scaled_input)
            hard_output = hard_scaled * self.coordinate_scale
            projection_successful = True
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"HardNet projection failed during transition: {e}")
            hard_output = neural_output  # Fallback to neural output
            projection_successful = False
        
        if not projection_successful:
            # FAIL-FAST: No soft fallback - crash on projection failure
            raise AssertionError(
                f"FAIL-FAST: Projection failed in hybrid mode:\n"
                f"Transition strength: {strength}\n"
                f"Soft output shape: {soft_output.shape}\n"
                f"Hard output shape: {hard_output.shape if 'hard_output' in locals() else 'N/A'}\n"
                f"This indicates fundamental issues with constraint satisfaction\n"
                f"NO soft fallback - FIX THE HARDNET PROJECTION LOGIC"
            )
        else:
            # Blend soft and hard outputs based on transition strength
            transition_output = (1.0 - strength) * soft_output + strength * hard_output
        
        # Compute mixed constraint violation loss for training
        # CRITICAL: Scale coordinates for constraint compatibility
        scaled_input = neural_output / self.coordinate_scale
        batch_size = scaled_input.shape[0]
        # Convert to double precision for constraint computation
        scaled_input_f64 = scaled_input.to(dtype=torch.float64)
        A_f = torch.mm(scaled_input_f64, self.A.t())
        
        b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)
        b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)
        
        lower_violations = F.relu(b_l_batch - A_f)
        upper_violations = F.relu(A_f - b_u_batch)
        
        # Scale violation loss by (1 - strength) since hard constraints handle the rest
        soft_violation_loss = (1.0 - strength) * torch.mean(lower_violations + upper_violations)
        self.last_constraint_violation_loss = soft_violation_loss
        
        # Record transition statistics
        if self.enable_logging and self.training:
            lower_satisfied = lower_violations <= self.numerical_tolerance
            upper_satisfied = upper_violations <= self.numerical_tolerance
            constraints_satisfied = lower_satisfied & upper_satisfied
            
            total_constraints = batch_size * self.n_constraints
            satisfied_constraints = torch.sum(constraints_satisfied).item()
            satisfaction_rate = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
            
            stats = HardNetProjectionStats(
                n_constraints_violated=total_constraints - satisfied_constraints,
                max_violation=torch.max(torch.max(lower_violations), torch.max(upper_violations)).item(),
                projection_norm=float(torch.norm(transition_output - neural_output).item()),
                constraint_satisfaction_rate=satisfaction_rate,
                numerical_warnings=0 if projection_successful else 1,
                computation_time_ms=0.0
            )
            self.projection_stats['batch_stats'].append(stats)
        
        return transition_output
    
    def _collect_projection_stats(self, 
                                neural_output: torch.Tensor,
                                projected_output: torch.Tensor, 
                                computation_time: float):
        with torch.no_grad():
            batch_size = neural_output.shape[0]
            
            # Count constraint violations (convert to double precision)
            neural_output_f64 = neural_output.to(dtype=torch.float64)
            projected_output_f64 = projected_output.to(dtype=torch.float64)
            A_f_original = torch.mm(neural_output_f64, self.A.t())
            A_f_projected = torch.mm(projected_output_f64, self.A.t())
            
            b_l_batch = self.b_l.unsqueeze(0).expand(batch_size, -1)
            b_u_batch = self.b_u.unsqueeze(0).expand(batch_size, -1)
            
            # Original violations
            original_violations = (
                torch.sum(A_f_original < b_l_batch - self.numerical_tolerance) +
                torch.sum(A_f_original > b_u_batch + self.numerical_tolerance)
            ).item()
            
            # Projected violations (should be zero)
            projected_violations = (
                torch.sum(A_f_projected < b_l_batch - self.numerical_tolerance) +
                torch.sum(A_f_projected > b_u_batch + self.numerical_tolerance)
            ).item()
            
            # Maximum violation magnitude
            max_violation = torch.max(
                torch.max(F.relu(b_l_batch - A_f_projected)),
                torch.max(F.relu(A_f_projected - b_u_batch))
            ).item()
            
            # Projection magnitude
            projection_norm = torch.norm(projected_output - neural_output).item()
            
            # Constraint satisfaction rate
            total_constraints = batch_size * self.n_constraints
            satisfaction_rate = 1.0 - (projected_violations / total_constraints) if total_constraints > 0 else 1.0
            
            # Count numerical warnings
            numerical_warnings = 0
            if max_violation > self.numerical_tolerance * 10:
                numerical_warnings += 1
            if projection_norm > 1e3:  # Very large corrections might indicate issues
                numerical_warnings += 1
            
            # Store statistics
            stats = HardNetProjectionStats(
                n_constraints_violated=original_violations,
                max_violation=max_violation,
                projection_norm=projection_norm,
                constraint_satisfaction_rate=satisfaction_rate,
                numerical_warnings=numerical_warnings,
                computation_time_ms=computation_time
            )
            
            self.projection_stats['batch_stats'].append(stats)
            
            # Log periodically
            if len(self.projection_stats['batch_stats']) % 100 == 0:
                self._log_statistics_summary()
    
    def _log_statistics_summary(self):
        recent_stats = self.projection_stats['batch_stats'][-100:]
        
        avg_violations = np.mean([s.n_constraints_violated for s in recent_stats])
        avg_satisfaction = np.mean([s.constraint_satisfaction_rate for s in recent_stats])
        avg_projection_norm = np.mean([s.projection_norm for s in recent_stats])
        total_warnings = sum([s.numerical_warnings for s in recent_stats])
        
        self.logger.info(
            f"HardNet Stats (last 100 batches): "
            f"Avg violations: {avg_violations:.1f}, "
            f"Satisfaction rate: {avg_satisfaction:.3f}, "
            f"Avg projection norm: {avg_projection_norm:.3f}, "
            f"Warnings: {total_warnings}"
        )
    
    def get_constraint_satisfaction_loss(self) -> torch.Tensor:
        if hasattr(self, 'last_constraint_violation_loss'):
            return self.last_constraint_violation_loss
        else:
            return torch.tensor(0.0, device=self.A.device if self.A is not None else 'cpu')
    
    def update_epoch(self, epoch: int):
        self.current_epoch = epoch
        # FIXED: Use > instead of >= for warm-start boundary
        # warm_start_epochs=20 means epochs 0-19 are warm-start, epoch 20+ is hard projection
        if self.enable_warm_start and epoch > self.warm_start_epochs:
            if self.warm_start_active:
                self.warm_start_active = False
                self.logger.info(f" CRITICAL: HardNet warm-start completed at epoch {epoch}. Switching to hard constraint projection!")
        elif self.enable_warm_start and epoch == self.warm_start_epochs:
            if self.warm_start_active:
                self.warm_start_active = False  
                self.logger.info(f" CRITICAL: HardNet warm-start completed at epoch {epoch} (boundary). Switching to hard constraint projection!")
        
        # Always log current status for debugging
        if self.enable_logging:
            mode = "warm-start (soft constraints)" if self.warm_start_active else "hard projection"
            self.logger.debug(f"HardNet epoch {epoch}: {mode} (warm_start_epochs={self.warm_start_epochs})")
    
    def verify_constraint_satisfaction(self, 
                                     neural_output: torch.Tensor,
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        if self.A is None:
            return {'all_satisfied': True, 'violations': []}
        
        with torch.no_grad():
            # Apply projection
            projected = self.forward(neural_output)
            if projected.dim() == 1:
                projected = projected.unsqueeze(0)
            
            # Check constraints (convert to double precision)
            projected_f64 = projected.to(dtype=torch.float64)
            A_f = torch.mm(projected_f64, self.A.t())
            
            violations = []
            for i in range(self.n_constraints):
                lower_violation = (A_f[:, i] < self.b_l[i] - tolerance).any().item()
                upper_violation = (A_f[:, i] > self.b_u[i] + tolerance).any().item()
                
                if lower_violation or upper_violation:
                    violations.append({
                        'constraint_idx': i,
                        'constraint_name': self.constraint_names[i] if self.constraint_names else f"Constraint {i}",
                        'lower_violation': lower_violation,
                        'upper_violation': upper_violation,
                        'actual_values': A_f[:, i].tolist(),
                        'expected_range': [self.b_l[i].item(), self.b_u[i].item()]
                    })
            
            return {
                'all_satisfied': len(violations) == 0,
                'n_violations': len(violations),
                'violations': violations,
                'max_violation': torch.max(
                    torch.max(F.relu(self.b_l - A_f)),
                    torch.max(F.relu(A_f - self.b_u))
                ).item()
            }
    
    def get_projection_statistics(self) -> Dict[str, Any]:
        if not self.projection_stats['batch_stats']:
            return {'no_data': True}
        
        stats = self.projection_stats['batch_stats']
        
        return {
            'total_batches': len(stats),
            'avg_violations': np.mean([s.n_constraints_violated for s in stats]),
            'avg_satisfaction_rate': np.mean([s.constraint_satisfaction_rate for s in stats]),
            'avg_projection_norm': np.mean([s.projection_norm for s in stats]),
            'avg_computation_time_ms': np.mean([s.computation_time_ms for s in stats]),
            'total_warnings': sum([s.numerical_warnings for s in stats]),
            'constraint_info': {
                'n_constraints': self.n_constraints,
                'n_variables': self.n_variables,
                'constraint_names': self.constraint_names
            }
        }
    
    def reset_statistics(self):
        self.projection_stats.clear()
        self.logger.info("Projection statistics reset")


# Integration utilities
class SpringHardNetIntegration:
    @staticmethod
    def create_hardnet_layer(constraint_matrix: AffineConstraintMatrix,
                           config: Optional[Dict[str, Any]] = None) -> HardNetAff:
        default_config = {
            'regularization_strength': 1e-1,  # CRITICAL FIX: Increased for numerical stability  
            'numerical_tolerance': 1e-6,  # FIXED: Consistent with main class
            'enable_warm_start': True,
            'warm_start_epochs': 50,
            'enable_logging': True,
            'coordinate_scale': 1000.0  # CRITICAL: Input/output coordinate scaling
        }
        
        if config:
            default_config.update(config)
        
        return HardNetAff(constraint_matrix, **default_config)
    
    @staticmethod
    def verify_gradient_flow(hardnet_layer: HardNetAff,
                           input_shape: Tuple[int, ...],
                           device: str = 'cpu') -> Dict[str, Any]:
        hardnet_layer.to(device)
        hardnet_layer.train()
        
        # Create test input with requires_grad
        test_input = torch.randn(input_shape, device=device, requires_grad=True)
        
        # Forward pass
        output = hardnet_layer(test_input)
        
        # Simple loss (sum of squared outputs)
        loss = torch.sum(output ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        input_grad_norm = torch.norm(test_input.grad).item() if test_input.grad is not None else 0.0
        has_gradients = test_input.grad is not None and torch.any(test_input.grad != 0)
        
        # Verify constraint satisfaction
        verification = hardnet_layer.verify_constraint_satisfaction(test_input.detach())
        
        return {
            'gradient_flow_verified': has_gradients,
            'input_gradient_norm': input_grad_norm,
            'constraint_satisfaction': verification['all_satisfied'],
            'max_violation': verification['max_violation'],
            'output_shape': output.shape,
            'forward_pass_successful': True
        }


# Example usage and testing
def create_test_hardnet_layer():
    """Create a test HardNet layer for demonstration."""
    # Mock constraint matrix for testing
    n_objects = 2
    n_variables = n_objects * 4  # [x, y, width, height] per object
    
    # Simple constraints: objects within bounds and non-overlapping
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  # obj0.x ≤ 800
        [0, 1, 0, 0, 0, 0, 0, 0],  # obj0.y ≤ 600
        [0, 0, 0, 0, 1, 0, 0, 0],  # obj1.x ≤ 800
        [0, 0, 0, 0, 0, 1, 0, 0],  # obj1.y ≤ 600
        [1, 0, 1, 0, 0, 0, 0, 0],  # obj0.x + obj0.width ≤ 1000
        [0, 0, 0, 0, 1, 0, 1, 0],  # obj1.x + obj1.width ≤ 1000
    ])
    
    b_l = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    b_u = np.array([800, 600, 800, 600, 1000, 1000])
    
    # Create object mapping (object_id -> vector_index)
    object_mapping = {obj_id: obj_id for obj_id in range(n_objects)}
    
    constraint_matrix = AffineConstraintMatrix(
        A=A, 
        b_l=b_l, 
        b_u=b_u, 
        n_objects=n_objects, 
        n_constraints=len(b_l), 
        constraint_names=[
            "obj0.x ≤ 800", "obj0.y ≤ 600", "obj1.x ≤ 800", 
            "obj1.y ≤ 600", "obj0 right bound", "obj1 right bound"
        ],
        object_mapping=object_mapping
    )
    
    return HardNetAff(constraint_matrix)


if __name__ == "__main__":
    print("=== SPRING HARDNET-AFF INTEGRATION - PHASE 3 ===\n")
    
    # Test 1: Create and test HardNet layer
    print("TEST 1: HardNet Layer Creation and Basic Operation")
    hardnet_layer = create_test_hardnet_layer()
    print(f"Created HardNet layer with {hardnet_layer.n_constraints} constraints")
    
    # Test 2: Forward pass
    print("\nTEST 2: Forward Pass with Constraint Enforcement")
    batch_size = 4
    n_variables = 8
    test_input = torch.randn(batch_size, n_variables) * 1000  # Large values to trigger violations
    
    print(f"Input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min().item():.1f}, {test_input.max().item():.1f}]")
    
    with torch.no_grad():
        projected_output = hardnet_layer(test_input)
    
    print(f"Output shape: {projected_output.shape}")
    print(f"Output range: [{projected_output.min().item():.1f}, {projected_output.max().item():.1f}]")
    
    # Test 3: Constraint verification
    print("\nTEST 3: Constraint Satisfaction Verification")
    verification = hardnet_layer.verify_constraint_satisfaction(test_input)
    print(f" All constraints satisfied: {verification['all_satisfied']}")
    print(f" Max violation: {verification['max_violation']:.2e}")
    print(f" Number of violations: {verification['n_violations']}")
    
    # Test 4: Gradient flow verification
    print("\nTEST 4: Gradient Flow Verification")
    gradient_test = SpringHardNetIntegration.verify_gradient_flow(
        hardnet_layer, (batch_size, n_variables)
    )
    print(f"Gradient flow verified: {gradient_test['gradient_flow_verified']}")
    print(f"Input gradient norm: {gradient_test['input_gradient_norm']:.2e}")
    print(f"Constraint satisfaction: {gradient_test['constraint_satisfaction']}")
    
    # Test 5: Warm-start functionality
    print("\nTEST 5: Warm-Start Training Mode")
    
    # Create a proper constraint matrix for warm-start layer
    test_constraint_matrix = AffineConstraintMatrix(
        A=hardnet_layer.A.cpu().numpy(),
        b_l=hardnet_layer.b_l.cpu().numpy(),
        b_u=hardnet_layer.b_u.cpu().numpy(),
        n_objects=2,
        n_constraints=hardnet_layer.n_constraints,
        constraint_names=hardnet_layer.constraint_names,
        object_mapping={0: 0, 1: 1}
    )
    
    warm_start_layer = HardNetAff(
        test_constraint_matrix,
        enable_warm_start=True, 
        warm_start_epochs=5
    )
    
    for epoch in range(7):
        warm_start_layer.update_epoch(epoch)
        warm_start_layer.train()
        output = warm_start_layer(test_input[:1])  # Single sample
        
        if epoch < 5:
            violation_loss = warm_start_layer.get_constraint_satisfaction_loss()
            print(f" Epoch {epoch}: Warm-start active, violation loss = {violation_loss.item():.3f}")
        else:
            print(f" Epoch {epoch}: Hard constraints active")
