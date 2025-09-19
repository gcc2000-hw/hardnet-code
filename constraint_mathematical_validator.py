"""
Mathematical Constraint Validation System
Implements rigorous mathematical checks to prevent rank deficiency and infeasible constraint sets
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy.optimize import linprog
import logging
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    VALID = "valid"
    RANK_DEFICIENT = "rank_deficient"
    INFEASIBLE = "infeasible"
    OVERCONSTRAINED = "overconstrained"
    CONTRADICTORY = "contradictory"


@dataclass
class ConstraintValidationReport:
    """Detailed validation report for constraint sets."""
    result: ValidationResult
    original_count: int
    valid_count: int
    rank: int
    condition_number: float
    feasible: bool
    error_messages: List[str]
    removed_constraints: List[str]


class MathematicalConstraintValidator:
    """
    Rigorous mathematical validator for constraint sets.
    Prevents rank deficiency and infeasible constraint sets that break HardNet.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Mathematical thresholds based on HardNet theory
        self.max_condition_number = 1e6  # Beyond this, pseudoinverse becomes unstable
        self.rank_tolerance = 1e-12  # SVD threshold for determining rank
        self.max_constraint_variable_ratio = 0.5  # Conservative: max 0.5 constraints per variable
        
    def validate_constraint_set(self, 
                               constraints: List[Any], 
                               n_objects: int,
                               converter) -> ConstraintValidationReport:
        """
        Comprehensive mathematical validation of constraint set.
        
        Args:
            constraints: List of constraint objects
            n_objects: Number of objects in scene
            converter: Instance of AffineConverter to convert constraints
            
        Returns:
            ConstraintValidationReport with detailed analysis
        """
        n_variables = n_objects * 4  # [x, y, w, h] per object
        error_messages = []
        removed_constraints = []
        
        # Step 1: Check constraint density
        if len(constraints) > n_variables * self.max_constraint_variable_ratio:
            max_allowed = int(n_variables * self.max_constraint_variable_ratio)
            error_messages.append(f"Too many constraints: {len(constraints)} > {max_allowed} (max {self.max_constraint_variable_ratio}:1 ratio)")
            
            # Keep only the most important constraints
            constraints = self._prioritize_constraints(constraints, max_allowed)
            removed_constraints = [f"excess_{i}" for i in range(max_allowed, len(constraints))]
        
        # Step 2: Convert to matrix form
        try:
            constraint_matrix = converter.convert_constraints_to_matrix(constraints, n_objects)
            A = constraint_matrix.A
            b_l = constraint_matrix.b_l  
            b_u = constraint_matrix.b_u
        except Exception as e:
            return ConstraintValidationReport(
                result=ValidationResult.CONTRADICTORY,
                original_count=len(constraints),
                valid_count=0,
                rank=0,
                condition_number=float('inf'),
                feasible=False,
                error_messages=[f"Matrix conversion failed: {e}"],
                removed_constraints=[]
            )
        
        # Step 3: Check rank and condition number
        rank_result = self._check_matrix_rank(A)
        if rank_result['is_rank_deficient']:
            error_messages.extend(rank_result['errors'])
            
            # Try to fix by removing linearly dependent constraints
            try:
                result = self._remove_dependent_constraints(A)
                if len(result) == 2:
                    A_fixed, valid_indices = result
                elif len(result) == 3:
                    A_fixed, valid_indices, metadata = result
                else:
                    self.logger.error(f"Unexpected return from _remove_dependent_constraints: {len(result)} values")
                    A_fixed, valid_indices = A, list(range(len(A)))
                
                constraints = [constraints[i] for i in valid_indices]
                removed_constraints.extend([f"dependent_constraint_{i}" for i in range(len(constraints)) if i not in valid_indices])
            except Exception as e:
                self.logger.error(f"Failed to remove dependent constraints: {e}")
                A_fixed, valid_indices = A, list(range(len(A)))
            
            # Re-check after fixing
            rank_result = self._check_matrix_rank(A_fixed)
            A = A_fixed
            
        # Step 4: Check feasibility
        feasibility_result = self._check_feasibility(A, b_l, b_u)
        if not feasibility_result['is_feasible']:
            error_messages.extend(feasibility_result['errors'])
            
            # Try to fix infeasible constraints
            fixed_constraints = self._fix_infeasible_constraints(constraints, A, b_l, b_u)
            if fixed_constraints:
                constraints = fixed_constraints
                removed_constraints.extend(feasibility_result.get('removed', []))
        
        # Step 5: Final validation
        final_rank = rank_result['rank']
        final_condition = rank_result['condition_number']
        final_feasible = feasibility_result['is_feasible']
        
        # Determine final result
        if final_rank < A.shape[0]:
            result = ValidationResult.RANK_DEFICIENT
        elif not final_feasible:
            result = ValidationResult.INFEASIBLE
        elif final_condition > self.max_condition_number:
            result = ValidationResult.OVERCONSTRAINED
        else:
            result = ValidationResult.VALID
            
        return ConstraintValidationReport(
            result=result,
            original_count=len(constraints),
            valid_count=len(constraints),
            rank=final_rank,
            condition_number=final_condition,
            feasible=final_feasible,
            error_messages=error_messages,
            removed_constraints=removed_constraints
        )
    
    def _check_matrix_rank(self, A: np.ndarray) -> Dict[str, Any]:
        """Check matrix rank and condition number."""
        try:
            # CRITICAL FIX: Ensure A is pure numeric numpy array
            if A.dtype == np.object_:
                # Handle mixed data types - convert to float64
                self.logger.error(f"Object dtype detected in matrix: {A.dtype}")
                # Try to extract numeric values from object array
                A_numeric = np.zeros(A.shape, dtype=np.float64)
                for i in range(A.shape[0]):
                    for j in range(A.shape[1]):
                        val = A[i, j]
                        if hasattr(val, 'item'):  # PyTorch tensor
                            A_numeric[i, j] = float(val.item())
                        elif isinstance(val, (int, float)):
                            A_numeric[i, j] = float(val)
                        else:
                            self.logger.error(f"Cannot convert matrix element to numeric: {type(val)}")
                            raise TypeError(f"Matrix contains non-numeric data: {type(val)}")
                A = A_numeric
            
            # Ensure proper float dtype
            A = A.astype(np.float64, copy=False)
            
            # Compute SVD
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            
            # Determine numerical rank
            tol = max(A.shape) * np.finfo(A.dtype).eps * S[0] if S[0] > 0 else 1e-12
            rank = np.sum(S > tol)
            
            # Compute condition number
            if S[-1] > 0:
                condition_number = S[0] / S[-1]
            else:
                condition_number = float('inf')
                
            errors = []
            is_rank_deficient = rank < A.shape[0]
            
            if is_rank_deficient:
                errors.append(f"Rank deficient matrix: rank {rank} < {A.shape[0]} constraints")
                
            if condition_number > self.max_condition_number:
                errors.append(f"Ill-conditioned matrix: condition number {condition_number:.2e}")
                
            return {
                'rank': rank,
                'condition_number': condition_number,
                'is_rank_deficient': is_rank_deficient,
                'singular_values': S,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'rank': 0,
                'condition_number': float('inf'),
                'is_rank_deficient': True,
                'singular_values': np.array([]),
                'errors': [f"SVD computation failed: {e}"]
            }
    
    def _check_feasibility(self, A: np.ndarray, b_l: np.ndarray, b_u: np.ndarray) -> Dict[str, Any]:
        """Check if constraint set b_l ≤ Ax ≤ b_u is feasible."""
        try:
            # Convert to standard LP form: min 0^T x subject to Ax ≤ b
            # Split b_l ≤ Ax ≤ b_u into Ax ≤ b_u and -Ax ≤ -b_l
            
            A_ineq_list = []
            b_ineq_list = []
            
            # Handle upper bounds: Ax ≤ b_u
            for i, b_u_val in enumerate(b_u):
                if not np.isinf(b_u_val):
                    A_ineq_list.append(A[i, :])
                    b_ineq_list.append(b_u_val)
            
            # Handle lower bounds: b_l ≤ Ax --> -Ax ≤ -b_l  
            for i, b_l_val in enumerate(b_l):
                if not np.isinf(b_l_val):
                    A_ineq_list.append(-A[i, :])
                    b_ineq_list.append(-b_l_val)
            
            if not A_ineq_list:
                # No constraints - always feasible
                return {'is_feasible': True, 'errors': []}
                
            A_ineq = np.vstack(A_ineq_list)
            b_ineq = np.array(b_ineq_list)
            
            # Solve feasibility LP
            c = np.zeros(A.shape[1])  # Minimize 0 (just check feasibility)
            
            result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, method='highs')
            
            errors = []
            if not result.success:
                errors.append(f"Constraint set is infeasible: {result.message}")
                
            return {
                'is_feasible': result.success,
                'errors': errors,
                'linprog_result': result
            }
            
        except Exception as e:
            return {
                'is_feasible': False,
                'errors': [f"Feasibility check failed: {e}"]
            }
    
    def _remove_dependent_constraints(self, A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Remove linearly dependent constraints, keep maximum independent set."""
        try:
            # Use QR decomposition with pivoting to find independent columns
            Q, R, P = np.linalg.qr(A.T, mode='full')  # Transpose to work with columns
            
            # Find rank by checking diagonal of R
            tol = max(A.shape) * np.finfo(A.dtype).eps * np.abs(R[0, 0]) if R.size > 0 else 1e-12
            rank = np.sum(np.abs(np.diag(R)) > tol)
            
            # Keep first 'rank' independent rows
            independent_indices = sorted(P[:rank])
            A_independent = A[independent_indices, :]
            
            self.logger.info(f"Removed {len(A) - len(independent_indices)} dependent constraints")
            return A_independent, independent_indices
            
        except Exception as e:
            self.logger.error(f"Failed to remove dependent constraints: {e}")
            return A, list(range(len(A)))
    
    def _fix_infeasible_constraints(self, 
                                  constraints: List[Any], 
                                  A: np.ndarray, 
                                  b_l: np.ndarray, 
                                  b_u: np.ndarray) -> Optional[List[Any]]:
        """Attempt to fix infeasible constraints by relaxing bounds."""
        # For now, just return None - let the caller handle infeasible sets
        # In future, could implement constraint relaxation strategies
        return None
    
    def _prioritize_constraints(self, constraints: List[Any], max_count: int) -> List[Any]:
        """Prioritize constraints, keeping most important ones."""
        # Simple prioritization: prefer boundary constraints and basic spatial relationships
        # This is a placeholder - could be made more sophisticated
        return constraints[:max_count]


# Quick validation function for use in training
def quick_validate_constraints(constraints: List[Any], n_objects: int, converter) -> bool:
    """Quick validation check - returns True if constraints are mathematically valid."""
    validator = MathematicalConstraintValidator()
    report = validator.validate_constraint_set(constraints, n_objects, converter)
    return report.result == ValidationResult.VALID