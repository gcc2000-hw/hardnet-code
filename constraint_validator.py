"""
Constraint Validation and Feasibility Checker
Critical fixes for SPRING-HardNet constraint satisfaction issues.

Addresses theoretical problems identified in analysis:
1. Infeasible constraint sets (b_l > b_u, negative bounds)
2. Coordinate scaling inconsistencies  
3. Mathematical inconsistencies in constraint formulation
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from constraint_to_affine_converter import AffineConstraintMatrix

@dataclass
class ConstraintValidationResult:
    """Results of constraint validation analysis."""
    is_feasible: bool
    has_empty_set: bool
    has_scaling_issues: bool
    has_numerical_issues: bool
    violations: List[str]
    warnings: List[str]
    suggestions: List[str]
    corrected_matrix: Optional[AffineConstraintMatrix] = None

class EvaluationConstraintValidator:
    """
    Validates constraint matrices for mathematical feasibility and numerical stability.
    Implements fixes based on HardNet theoretical requirements.
    """
    
    def __init__(self, coordinate_scale: float = 20.0, tolerance: float = 1e-8):  # CRITICAL: Updated for target normalization
        self.coordinate_scale = coordinate_scale
        self.tolerance = tolerance
        self.logger = logging.getLogger('ConstraintValidator')
        
    def validate_constraint_matrix(self, 
                                 constraint_matrix: AffineConstraintMatrix,
                                 expected_coordinate_range: Tuple[float, float] = (0.0, 20.0)) -> ConstraintValidationResult:  # Updated range
        """
        Comprehensive validation of constraint matrix for mathematical feasibility.
        
        Args:
            constraint_matrix: Constraint matrix to validate
            expected_coordinate_range: Expected range of coordinate values
            
        Returns:
            Detailed validation results with fixes
        """
        result = ConstraintValidationResult(
            is_feasible=True,
            has_empty_set=False, 
            has_scaling_issues=False,
            has_numerical_issues=False,
            violations=[],
            warnings=[],
            suggestions=[]
        )
        
        # 1. Check basic feasibility: b_l <= b_u
        infeasible_mask = constraint_matrix.b_l > constraint_matrix.b_u + self.tolerance
        if np.any(infeasible_mask):
            result.is_feasible = False
            result.has_empty_set = True
            infeasible_indices = np.where(infeasible_mask)[0]
            for idx in infeasible_indices:
                result.violations.append(
                    f"Constraint {idx}: Lower bound {constraint_matrix.b_l[idx]:.6f} > "
                    f"Upper bound {constraint_matrix.b_u[idx]:.6f}"
                )
            
        # 2. Check coordinate range compatibility
        coord_min, coord_max = expected_coordinate_range
        
        # Check if upper bounds are compatible with positive coordinates
        if coord_min >= 0 and np.any(constraint_matrix.b_u < -self.tolerance):
            result.is_feasible = False
            negative_upper_mask = constraint_matrix.b_u < -self.tolerance
            negative_indices = np.where(negative_upper_mask)[0]
            for idx in negative_indices:
                result.violations.append(
                    f"Constraint {idx}: Negative upper bound {constraint_matrix.b_u[idx]:.6f} "
                    f"incompatible with positive coordinates [{coord_min}, {coord_max}]"
                )
                
        # 3. Check for scaling inconsistencies
        # Detect if constraint bounds seem to be in different scale than coordinates
        bound_magnitude = max(
            np.abs(constraint_matrix.b_l[np.isfinite(constraint_matrix.b_l)]).max() if np.any(np.isfinite(constraint_matrix.b_l)) else 0,
            np.abs(constraint_matrix.b_u[np.isfinite(constraint_matrix.b_u)]).max() if np.any(np.isfinite(constraint_matrix.b_u)) else 0
        )
        coord_magnitude = max(abs(coord_min), abs(coord_max))
        
        if bound_magnitude > 0 and coord_magnitude > 0:
            scale_ratio = coord_magnitude / bound_magnitude
            if scale_ratio > 100 or scale_ratio < 0.01:
                result.has_scaling_issues = True
                result.warnings.append(
                    f"Scale mismatch: coordinate range {coord_magnitude:.2f} vs "
                    f"constraint bounds {bound_magnitude:.6f} (ratio: {scale_ratio:.2f})"
                )
                result.suggestions.append(
                    f"Consider scaling constraint bounds by factor {1/self.coordinate_scale}"
                )
        
        # 4. Check matrix properties
        A = constraint_matrix.A
        
        # Check for numerical issues
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            result.has_numerical_issues = True
            result.violations.append("Constraint matrix A contains NaN or Inf values")
            
        # Check matrix rank vs dimensions
        try:
            rank = np.linalg.matrix_rank(A, tol=self.tolerance)
            expected_rank = min(A.shape)
            if rank < expected_rank:
                result.warnings.append(
                    f"Rank deficient matrix: rank {rank} < min dimension {expected_rank}"
                )
                result.suggestions.append("Consider constraint preprocessing to remove redundancy")
        except:
            result.has_numerical_issues = True
            result.violations.append("Failed to compute matrix rank")
            
        # Check condition number
        try:
            if A.shape[0] > 0 and A.shape[1] > 0:
                AAT = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
                cond = np.linalg.cond(AAT)
                if cond > 1e12:
                    result.has_numerical_issues = True
                    result.warnings.append(f"High condition number: {cond:.2e}")
                    result.suggestions.append("Consider regularization or constraint preprocessing")
        except:
            result.warnings.append("Failed to compute condition number")
        
        # 5. Generate corrected matrix if possible
        if not result.is_feasible and result.has_empty_set:
            try:
                corrected = self._attempt_constraint_correction(constraint_matrix, expected_coordinate_range)
                if corrected is not None:
                    result.corrected_matrix = corrected
                    result.suggestions.append("Automatically corrected constraint matrix available")
            except Exception as e:
                result.warnings.append(f"Failed to generate corrected constraints: {e}")
        
        return result
    
    def _attempt_constraint_correction(self, 
                                     constraint_matrix: AffineConstraintMatrix,
                                     coord_range: Tuple[float, float]) -> Optional[AffineConstraintMatrix]:
        """
        Attempt to automatically correct infeasible constraints.
        """
        A_corrected = constraint_matrix.A.copy()
        b_l_corrected = constraint_matrix.b_l.copy()
        b_u_corrected = constraint_matrix.b_u.copy()
        
        coord_min, coord_max = coord_range
        
        # Fix 1: Ensure b_l <= b_u
        swap_mask = b_l_corrected > b_u_corrected
        if np.any(swap_mask):
            b_l_corrected[swap_mask], b_u_corrected[swap_mask] = b_u_corrected[swap_mask], b_l_corrected[swap_mask]
            self.logger.warning(f"Swapped {np.sum(swap_mask)} lower/upper bounds")
        
        # Fix 2: Ensure compatibility with coordinate range
        if coord_min >= 0:
            # For positive coordinates, negative upper bounds are infeasible
            negative_upper_mask = b_u_corrected < 0
            if np.any(negative_upper_mask):
                b_u_corrected[negative_upper_mask] = coord_max * 1.1  # Set to reasonable upper bound
                self.logger.warning(f"Fixed {np.sum(negative_upper_mask)} negative upper bounds")
        
        # Fix 3: Apply coordinate scaling if detected
        bound_magnitude = max(
            np.abs(b_l_corrected[np.isfinite(b_l_corrected)]).max() if np.any(np.isfinite(b_l_corrected)) else 0,
            np.abs(b_u_corrected[np.isfinite(b_u_corrected)]).max() if np.any(np.isfinite(b_u_corrected)) else 0
        )
        coord_magnitude = max(abs(coord_min), abs(coord_max))
        
        if bound_magnitude > 0 and coord_magnitude / bound_magnitude > 100:
            scale_factor = 1.0 / self.coordinate_scale
            finite_l_mask = np.isfinite(b_l_corrected)
            finite_u_mask = np.isfinite(b_u_corrected)
            b_l_corrected[finite_l_mask] *= scale_factor
            b_u_corrected[finite_u_mask] *= scale_factor
            self.logger.warning(f"Applied coordinate scaling factor {scale_factor}")
        
        # Create corrected matrix
        corrected_matrix = AffineConstraintMatrix(
            A=A_corrected,
            b_l=b_l_corrected,
            b_u=b_u_corrected,
            n_objects=constraint_matrix.n_objects,
            n_constraints=len(b_l_corrected),
            constraint_names=[f"corrected_{name}" for name in constraint_matrix.constraint_names],
            object_mapping=constraint_matrix.object_mapping.copy()
        )
        
        # Validate the correction
        validation = self.validate_constraint_matrix(corrected_matrix, coord_range)
        if validation.is_feasible:
            return corrected_matrix
        else:
            return None
    
    def log_validation_results(self, result: ConstraintValidationResult):
        """Log detailed validation results."""
        if result.is_feasible:
            self.logger.info(" Constraint matrix validation PASSED")
        else:
            self.logger.error(" Constraint matrix validation FAILED")
        
        if result.violations:
            self.logger.error("VIOLATIONS:")
            for violation in result.violations:
                self.logger.error(f"  - {violation}")
        
        if result.warnings:
            self.logger.warning("WARNINGS:")
            for warning in result.warnings:
                self.logger.warning(f"  - {warning}")
        
        if result.suggestions:
            self.logger.info("SUGGESTIONS:")
            for suggestion in result.suggestions:
                self.logger.info(f"  - {suggestion}")


def validate_constraints_before_training(constraint_matrix: AffineConstraintMatrix,
                                        coordinate_range: Tuple[float, float] = (0.0, 20.0),
                                        coordinate_scale: float = 20.0) -> AffineConstraintMatrix:  # Updated for target normalization
    """
    Validate and fix constraints before training.
    
    Returns corrected constraint matrix or raises error if unfixable.
    """
    validator = ConstraintValidator(coordinate_scale=coordinate_scale)
    result = validator.validate_constraint_matrix(constraint_matrix, coordinate_range)
    validator.log_validation_results(result)
    
    if not result.is_feasible:
        if result.corrected_matrix is not None:
            logging.getLogger('ConstraintValidator').warning(
                " Using automatically corrected constraint matrix"
            )
            return result.corrected_matrix
        else:
            raise ValueError(
                f"Infeasible constraint matrix detected:\n"
                f"Violations: {result.violations}\n"
                f"Fix the constraint generation logic before training."
            )
    
    return constraint_matrix


if __name__ == "__main__":
    # Test constraint validation
    print("=== CONSTRAINT VALIDATION TEST ===")
    
    # Create infeasible test constraints 
    A_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    b_l_test = np.array([-np.inf, -np.inf])
    b_u_test = np.array([-0.000041, 600])  # Negative upper bound - infeasible!
    
    test_matrix = AffineConstraintMatrix(
        A=A_test,
        b_l=b_l_test,
        b_u=b_u_test,
        n_objects=1,
        n_constraints=2,
        constraint_names=["test_x", "test_y"],
        object_mapping={0: 0}
    )
    
    validator = ConstraintValidator()
    result = validator.validate_constraint_matrix(test_matrix)
    validator.log_validation_results(result)
    
    if result.corrected_matrix:
        print(f"Original b_u: {test_matrix.b_u}")
        print(f"Corrected b_u: {result.corrected_matrix.b_u}")