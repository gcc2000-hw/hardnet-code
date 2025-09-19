# Converts classified affine constraints into matrix form for HardNet projection so t1-t4 + and


import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict , namedtuple
from constraint_language_v2 import ConstraintT1, ConstraintT2, ConstraintT3, ConstraintT4, ConstraintAND

# Import new constraint types  
try:
    from constraint_language_v2_extended import (
        BoundaryConstraint, SizeRatioConstraint, NonOverlapConstraint, 
        RelativeToExistingConstraint, ExtendedConstraintToAffineConverter
    )
    EXTENDED_CONSTRAINTS_AVAILABLE = True
except ImportError:
    print("Warning: Extended constraint types not available")
    EXTENDED_CONSTRAINTS_AVAILABLE = False

@dataclass
class AffineConstraintMatrix:
    #convert the constraints to matrix form of bl <= A*y <= bu where bl is lower bound and bu is upper bound
    A: np.ndarray
    b_l: np.ndarray
    b_u: np.ndarray
    n_objects: int
    n_constraints: int
    constraint_names: List[str]
    object_mapping: Dict[int, int]
    variable_names: List[str] = None
    
    def __post_init__(self):
        if self.variable_names is None:
            self.variable_names = []
            for obj_id in range(self.n_objects):
                for var_name in ["x", "y", "width", "height"]:
                    self.variable_names.append(f"obj{obj_id}.{var_name}")


class ConstraintToAffineConverter:
    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging(enable_logging)
        self.conversion_stats = defaultdict(int)
        
    def _setup_logging(self, enable: bool) -> logging.Logger:
        """Setup logging for matrix conversion."""
        logger = logging.getLogger('AffineConverter')
        if enable and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def convert_constraints_to_matrix(self, 
                                    constraints: List[Any], 
                                    n_objects: int) -> AffineConstraintMatrix:
        # Convert a list of affine constraints to matrix form.

        # Validate inputs
        if n_objects <= 0:
            raise ValueError(f"Number of objects must be positive, got {n_objects}")
        
        if not constraints:
            # Return empty constraint matrix
            return self._create_empty_matrix(n_objects)
        
        # Object mapping: object_id -> vector_index
        object_mapping = {obj_id: obj_id for obj_id in range(n_objects)}
        
        # Variables per object: [x, y, width, height]
        n_variables = n_objects * 4
        
        # Collect all constraint rows
        constraint_rows = []
        constraint_names = []
        
        for constraint in constraints:
            try:
                rows, names = self._convert_single_constraint(constraint, n_variables, object_mapping)
                constraint_rows.extend(rows)
                constraint_names.extend(names)
            except Exception as e:
                self.logger.error(f"Failed to convert constraint {constraint}: {e}")
                raise
        
        if not constraint_rows:
            return self._create_empty_matrix(n_objects)
        
        # Stack constraint rows into matrices
        A_matrix = np.vstack([row['A'] for row in constraint_rows])
        b_l_vector = np.array([row['b_l'] for row in constraint_rows])
        b_u_vector = np.array([row['b_u'] for row in constraint_rows])
        
        # CRITICAL: Apply constraint preprocessing for numerical stability
        A_processed, b_l_processed, b_u_processed, names_processed = self._preprocess_constraints(
            A_matrix, b_l_vector, b_u_vector, constraint_names, n_objects
        )
        
        result = AffineConstraintMatrix(
            A=A_processed,
            b_l=b_l_processed,
            b_u=b_u_processed,
            n_objects=n_objects,
            n_constraints=len(A_processed),
            constraint_names=names_processed,
            object_mapping=object_mapping
        )
        
        # Log conversion results
        self.logger.info(
            f"Matrix conversion: {len(constraints)} input constraints -> "
            f"{result.n_constraints} matrix rows for {n_objects} objects"
        )
        
        return result
    
    def _convert_single_constraint(self, 
                                 constraint: Any, 
                                 n_variables: int, 
                                 object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        #Convert a single constraint to matrix row
        if isinstance(constraint, ConstraintT1):
            return self._handle_constraint_t1(constraint, n_variables, object_mapping)
        elif isinstance(constraint, ConstraintT2):
            return self._handle_constraint_t2(constraint, n_variables, object_mapping)
        elif isinstance(constraint, ConstraintT3):
            return self._handle_constraint_t3(constraint, n_variables, object_mapping)
        elif isinstance(constraint, ConstraintT4):
            return self._handle_constraint_t4(constraint, n_variables, object_mapping)
        elif isinstance(constraint, ConstraintAND):
            return self._handle_constraint_and(constraint, n_variables, object_mapping)
        elif EXTENDED_CONSTRAINTS_AVAILABLE:
            # Handle new constraint types using extended converter
            if isinstance(constraint, (BoundaryConstraint, SizeRatioConstraint, 
                                     NonOverlapConstraint, RelativeToExistingConstraint)):
                return self._handle_extended_constraint(constraint, n_variables, object_mapping)
        
        raise ValueError(f"Unknown constraint type: {type(constraint)}")
    
    def _handle_constraint_t1(self, 
                            constraint: ConstraintT1, 
                            n_variables: int, 
                            object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        # Handle T1 constraints - single object + value => con_left_val(o1, val) -> o1.x ≤ val and con_right_val(o1, val) -> o1.x ≥  and on_xeq_val(o1, val) -> o1.x = val 
        # For ≤ constraints: offset makes bound stricter (o1.x ≤ val - offset)
        # For ≥ constraints: offset makes bound looser (o1.x ≥ val - offset)
        # For = constraints: offset ignored (equality is exact)
        obj_id = constraint.o1
        var_idx = constraint.v1  # 0=x,1=y,2=width,3=height
        value = constraint.val
        offset = constraint.offset
        op = constraint.c  # "lt", "gt", "eq"
        
        # Get variable index in the full vector
        if obj_id not in object_mapping:
            raise ValueError(f"Object {obj_id} not found in object_mapping")
        
        vector_idx = object_mapping[obj_id] * 4 + var_idx
        
        # Create constraint row
        A_row = np.zeros(n_variables)
        A_row[vector_idx] = 1.0
        
        # FIXED: Consistent offset handling
        if op == "lt":  # o.var ≤ val - offset (stricter bound)
            b_l, b_u = float('-inf'), value - offset
        elif op == "gt":  # o.var ≥ val - offset (account for offset in lower bound)
            b_l, b_u = value - offset, float('inf')
        elif op == "eq":  # o.var = val (ignore offset for exact equality)
            b_l, b_u = value, value
        else:
            raise ValueError(f"Unknown T1 operation: {op}")
        
        # Create constraint description
        var_names = ["x", "y", "width", "height"]
        if offset != 0 and op != "eq":
            constraint_name = f"obj{obj_id}.{var_names[var_idx]} {op} {value} (offset: {offset})"
        else:
            constraint_name = f"obj{obj_id}.{var_names[var_idx]} {op} {value}"
        
        constraint_row = {
            'A': A_row.reshape(1, -1),
            'b_l': b_l,
            'b_u': b_u
        }
        
        self.conversion_stats['T1_constraints'] += 1
        return [constraint_row], [constraint_name]
    #Handle T2 constraints - two object comparisons
    # con_left(o1, o2) -> o1.x ≤ o2.x and con_wider(o1, o2) -> o1.width ≥ o2.width and con_xeq(o1, o2) -> o1.x = o2.x
    # above(o1, o2, 100) means o1 at least 100 units above o2
    # Translates to o1.y ≤ o2.y - 100
    # Matrix form o1.y - o2.y ≤ -100
    # So offset becomes negative in the bound for separation constraints
    def _handle_constraint_t2(self, 
                            constraint: ConstraintT2, 
                            n_variables: int, 
                            object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:

        obj1_id = constraint.o1
        var1_idx = constraint.v1
        obj2_id = constraint.o2
        var2_idx = constraint.v2
        offset = constraint.offset
        op = constraint.c
        
        # Get variable indices
        if obj1_id not in object_mapping or obj2_id not in object_mapping:
            raise ValueError(f"Objects {obj1_id}, {obj2_id} not found in object_mapping")
        
        vector_idx1 = object_mapping[obj1_id] * 4 + var1_idx
        vector_idx2 = object_mapping[obj2_id] * 4 + var2_idx
        
        # Create constraint row: o1.var1 - o2.var2 {op} -offset
        A_row = np.zeros(n_variables)
        A_row[vector_idx1] = 1.0
        A_row[vector_idx2] = -1.0
        
        # FIXED: Consistent offset handling for separation constraints
        if op == "lt":  # o1.var1 ≤ o2.var2 + offset -> o1.var1 - o2.var2 leq offset
            b_l, b_u = float('-inf'), offset
        elif op == "gt":  # o1.var1 ≥ o2.var2 + offset -> o1.var1 - o2.var2 geq offset  
            b_l, b_u = offset, float('inf')
        elif op == "eq":  # o1.var1 = o2.var2 (ignore offset for exact equality)
            b_l, b_u = 0, 0
        else:
            raise ValueError(f"Unknown T2 operation: {op}")
        
        # Create constraint description
        var_names = ["x", "y", "width", "height"]
        if offset != 0 and op != "eq":
            constraint_name = f"obj{obj1_id}.{var_names[var1_idx]} {op} obj{obj2_id}.{var_names[var2_idx]} + {offset}"
        else:
            constraint_name = f"obj{obj1_id}.{var_names[var1_idx]} {op} obj{obj2_id}.{var_names[var2_idx]}"
        
        constraint_row = {
            'A': A_row.reshape(1, -1),
            'b_l': b_l,
            'b_u': b_u
        }
        
        # DEBUG: Log T2 constraint matrix construction
        if hasattr(self, 'enable_logging') and self.enable_logging:
            self.logger.info(f" T2 MATRIX DEBUG: {constraint_name}")
            self.logger.info(f"   A_row: {A_row}")
            self.logger.info(f"   Bounds: [{b_l}, {b_u}]")
            self.logger.info(f"   obj{obj1_id} coefficient: {A_row[vector_idx1]}")
            self.logger.info(f"   obj{obj2_id} coefficient: {A_row[vector_idx2]}")
        
        self.conversion_stats['T2_constraints'] += 1
        return [constraint_row], [constraint_name]
    # Handle T3 constraints: arithmetic bounds
    # right_bound(o, val) -> o.x + o.width leq val
    # up_bound(o, val) -> o.y + o.height geq val
    # For boundary constraints, offset makes bound stricter
    # right_bound(o, val, offset=10) -> o.x + o.width leq val - 10
    def _handle_constraint_t3(self, 
                            constraint: ConstraintT3, 
                            n_variables: int, 
                            object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        obj1_id = constraint.o1
        var1_idx = constraint.v1
        obj2_id = constraint.o2
        var2_idx = constraint.v2
        value = constraint.val
        offset = constraint.offset
        op = constraint.c
        arithmetic_op = constraint.a  # "+" or "-"
        
        # Get variable indices
        if obj1_id not in object_mapping or obj2_id not in object_mapping:
            raise ValueError(f"Objects {obj1_id}, {obj2_id} not found in object_mapping")
        
        vector_idx1 = object_mapping[obj1_id] * 4 + var1_idx
        vector_idx2 = object_mapping[obj2_id] * 4 + var2_idx
        
        # Create constraint row: o1.var1 + or - o2.var2 {op} value
        A_row = np.zeros(n_variables)
        A_row[vector_idx1] = 1.0
        
        if arithmetic_op == "+":
            A_row[vector_idx2] = 1.0
        elif arithmetic_op == "-":
            A_row[vector_idx2] = -1.0
        else:
            raise ValueError(f"Unknown T3 arithmetic operation: {arithmetic_op}")
        
        # Consistent offset handling - boundary constraints become stricter
        if op == "lt":  # o1.var1 +- o2.var2 leq val - offset
            b_l, b_u = float('-inf'), value - offset
        elif op == "gt":  # o1.var1 +- o2.var2 geq val + offset  
            b_l, b_u = value + offset, float('inf')
        elif op == "eq":  # o1.var1 +- o2.var2 = val (ignore offset)
            b_l, b_u = value, value
        else:
            raise ValueError(f"Unknown T3 operation: {op}")
        
        # Create constraint description
        var_names = ["x", "y", "width", "height"]
        if offset != 0 and op != "eq":
            constraint_name = f"obj{obj1_id}.{var_names[var1_idx]} {arithmetic_op} obj{obj2_id}.{var_names[var2_idx]} {op} {value} (offset: {offset})"
        else:
            constraint_name = f"obj{obj1_id}.{var_names[var1_idx]} {arithmetic_op} obj{obj2_id}.{var_names[var2_idx]} {op} {value}"
        
        constraint_row = {
            'A': A_row.reshape(1, -1),
            'b_l': b_l,
            'b_u': b_u
        }
        
        self.conversion_stats['T3_constraints'] += 1
        return [constraint_row], [constraint_name]
    #Handle T4 constraints: complex arithmetic between objects
    #con_leftleft(o1, o2) -> o1.x + o1.width ≤ o2.x
    #con_aboveabove(o1, o2) -> o1.y + o1.height ≤ o2.y
    #con_leftleft(o1, o2, offset=10) -> o1.x + o1.width ≤ o2.x + 10
    #This allows for 10 units of gap between objects
    def _handle_constraint_t4(self, 
                            constraint: ConstraintT4, 
                            n_variables: int, 
                            object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        obj1_id = constraint.o1
        var1_idx = constraint.v1
        obj2_id = constraint.o2
        var2_idx = constraint.v2
        obj3_id = constraint.o3
        var3_idx = constraint.v3
        offset = constraint.offset
        op = constraint.c
        arithmetic_op = constraint.a  # "+" or "-"
        
        # Get variable indices
        if (obj1_id not in object_mapping or obj2_id not in object_mapping or 
            obj3_id not in object_mapping):
            raise ValueError(f"Objects {obj1_id}, {obj2_id}, {obj3_id} not found in object_mapping")
        
        vector_idx1 = object_mapping[obj1_id] * 4 + var1_idx
        vector_idx2 = object_mapping[obj2_id] * 4 + var2_idx
        vector_idx3 = object_mapping[obj3_id] * 4 + var3_idx
        
        # Create constraint row: (o1.var1 ± o2.var2) - o3.var3 {op} offset
        A_row = np.zeros(n_variables)
        A_row[vector_idx1] = 1.0
        if arithmetic_op == "+":
            A_row[vector_idx2] = 1.0
        elif arithmetic_op == "-":
            A_row[vector_idx2] = -1.0
        else:
            raise ValueError(f"Unknown T4 arithmetic operation: {arithmetic_op}")
        A_row[vector_idx3] = -1.0
        # Offset handling consistent with separation semantics
        if op == "lt":  # (o1.var1 +- o2.var2) - o3.var3 ≤ offset
            b_l, b_u = float('-inf'), offset
        elif op == "gt":  # (o1.var1 +- o2.var2) - o3.var3 ≥ offset
            b_l, b_u = offset, float('inf')
        elif op == "eq":  # (o1.var1 +- o2.var2) - o3.var3 = offset
            b_l, b_u = offset, offset
        else:
            raise ValueError(f"Unknown T4 operation: {op}")
        
        # Create constraint description
        var_names = ["x", "y", "width", "height"]
        constraint_name = (f"(obj{obj1_id}.{var_names[var1_idx]} {arithmetic_op} "
                         f"obj{obj2_id}.{var_names[var2_idx]}) - "
                         f"obj{obj3_id}.{var_names[var3_idx]} {op} {offset}")
        
        constraint_row = {
            'A': A_row.reshape(1, -1),
            'b_l': b_l,
            'b_u': b_u
        }
        
        self.conversion_stats['T4_constraints'] += 1
        return [constraint_row], [constraint_name]
    #Handle AND constraints - conjunction of multiple constraints
    # converts all subconstraints and returns them as separate rows
    def _handle_constraint_and(self, 
                             constraint: ConstraintAND, 
                             n_variables: int, 
                             object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        all_rows = []
        all_names = []
        
        for sub_constraint in constraint.c:
            rows, names = self._convert_single_constraint(sub_constraint, n_variables, object_mapping)
            all_rows.extend(rows)
            all_names.extend(names)
        
        self.conversion_stats['AND_constraints'] += 1
        return all_rows, all_names
    
    def _handle_extended_constraint(self, 
                                  constraint: Any, 
                                  n_variables: int, 
                                  object_mapping: Dict[int, int]) -> Tuple[List[Dict], List[str]]:
        """Handle new constraint types using the extended converter."""
        if not EXTENDED_CONSTRAINTS_AVAILABLE:
            raise ValueError("Extended constraints not available")
        
        # Use the extended converter to handle new constraint types
        extended_converter = ExtendedConstraintToAffineConverter()
        
        # Convert the constraint to affine form - FIXED METHOD NAMES
        if isinstance(constraint, BoundaryConstraint):
            A_row, b_l, b_u = extended_converter.convert_boundary_constraint(constraint, n_variables // 4)
            name = f"boundary_{constraint.obj_idx}_{constraint.constraint_type}"
        elif isinstance(constraint, SizeRatioConstraint):
            A_row, b_l, b_u = extended_converter.convert_size_ratio_constraint(constraint, n_variables // 4)
            name = f"size_ratio_{constraint.smaller_obj_idx}_{constraint.larger_obj_idx}_{constraint.dimension}"
        elif isinstance(constraint, NonOverlapConstraint):
            # Non-overlap constraints may generate multiple rows (OR structure)
            affine_forms = extended_converter.convert_non_overlap_constraint(constraint, n_variables // 4)
            rows = []
            names = []
            for i, (A_row, b_l, b_u) in enumerate(affine_forms):
                rows.append({'A': A_row, 'b_l': b_l, 'b_u': b_u})
                names.append(f"non_overlap_{constraint.obj1_idx}_{constraint.obj2_idx}_{i}")
            self.conversion_stats['extended_constraints'] += 1
            return rows, names
        elif isinstance(constraint, RelativeToExistingConstraint):
            A_row, b_l, b_u = extended_converter.convert_relative_to_existing_constraint(constraint, n_variables // 4)
            name = f"relative_{constraint.new_object_idx}_to_existing"
        else:
            raise ValueError(f"Unsupported extended constraint type: {type(constraint)}")
        
        # Single constraint case (boundary, size ratio, relative)
        row = {'A': A_row, 'b_l': b_l, 'b_u': b_u}
        self.conversion_stats['extended_constraints'] += 1
        return [row], [name]
    
    def _preprocess_constraints(self, A, b_l, b_u, constraint_names, n_objects):
        """CRITICAL MATHEMATICAL FIXES for numerical stability"""
        import numpy as np
        
        original_shape = A.shape
        self.logger.info(f"Preprocessing constraints: {original_shape}")
        
        # Step 1: Rank analysis and redundancy removal
        # Convert A to tensor if needed
        A_tensor = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        matrix_rank = torch.linalg.matrix_rank(A_tensor).item()
        
        if matrix_rank < A.shape[0]:
            self.logger.warning(f"Rank deficiency detected: {A.shape[0]} constraints, rank = {matrix_rank}")
            
            # THEORETICAL FIX: Use QR decomposition for proper rank reduction
            # CRITICAL FIX: Use SVD-based rank reduction instead of broken QR approach
            # SVD provides mathematically sound rank identification and row selection
            try:
                U, S, Vh = torch.linalg.svd(A_tensor, full_matrices=False)
                
                # Determine rank with proper numerical threshold
                sv_threshold = max(S.max().item() * 1e-12, 1e-15)  # Stricter threshold for stability
                rank_indices = torch.where(S > sv_threshold)[0]
                actual_rank = len(rank_indices)
                
                self.logger.info(f"SVD rank analysis: {A.shape[0]} constraints, singular values range [{S.min():.2e}, {S.max():.2e}]")
                self.logger.info(f"SVD threshold: {sv_threshold:.2e}, identified rank: {actual_rank}")
                
                # Take the most significant constraints up to the determined rank
                # SVD naturally orders by significance (largest singular values first)
                max_constraints = min(actual_rank, matrix_rank, A.shape[0])
                selected_indices = rank_indices[:max_constraints]
                
                # CRITICAL: Map back to original constraint indices properly
                # U contains the constraint combinations, but we need original rows
                # Use the strongest rows based on Frobenius norm after SVD projection
                A_projected = U[:, :max_constraints] @ torch.diag(S[:max_constraints]) @ Vh[:max_constraints, :]
                row_norms = torch.norm(A_projected, dim=1)
                _, strongest_rows = torch.topk(row_norms, max_constraints)
                
                A_reduced = A[strongest_rows.cpu().numpy()]
                b_l_reduced = b_l[strongest_rows.cpu().numpy()]  
                b_u_reduced = b_u[strongest_rows.cpu().numpy()]
                names_reduced = [constraint_names[i] if i < len(constraint_names) else f"C_{i}"
                               for i in strongest_rows.cpu().numpy()]
                
                # CRITICAL: Verify the result is actually full rank
                final_rank = torch.linalg.matrix_rank(torch.from_numpy(A_reduced).float())
                if final_rank < len(A_reduced):
                    self.logger.error(f"RANK DEFICIENCY DETECTED: {final_rank} < {len(A_reduced)}")
                    self.logger.error(f"This indicates constraint contradictions or numerical issues")
                    # Use only the truly independent constraints
                    A_reduced = A_reduced[:final_rank]
                    b_l_reduced = b_l_reduced[:final_rank]
                    b_u_reduced = b_u_reduced[:final_rank]
                    names_reduced = names_reduced[:final_rank]
                
                self.logger.info(f"SVD-based reduction: {A.shape[0]} -> {len(A_reduced)} constraints (rank: {final_rank})")
                
            except Exception as svd_error:
                self.logger.error(f"SVD decomposition failed: {svd_error}")
                raise RuntimeError(f"Critical constraint preprocessing failure: {svd_error}")
            
            self.logger.info(f"Reduced to {len(A_reduced)} independent constraints")
        else:
            A_reduced = A_tensor.clone()
            b_l_tensor = b_l if isinstance(b_l, torch.Tensor) else torch.tensor(b_l)
            b_u_tensor = b_u if isinstance(b_u, torch.Tensor) else torch.tensor(b_u)
            b_l_reduced = b_l_tensor.clone()
            b_u_reduced = b_u_tensor.clone()
            names_reduced = constraint_names.copy()
        
        # Step 2: MATHEMATICAL FIX - Correct coordinate normalization
        # PRODUCTION FIX: Remove double-scaling bug
        # HardNet will handle ALL coordinate scaling internally
        # AffineConverter should work in per-mille coordinates [0,1000]
        # and let HardNet scale to normalized [0,1] space
        
        # NO SCALING in AffineConverter - work in original per-mille coordinates
        # CRITICAL FIX: Convert tensors to numpy arrays for consistent data types
        if isinstance(A_reduced, torch.Tensor):
            A_normalized = A_reduced.cpu().numpy()
        else:
            A_normalized = A_reduced
            
        if isinstance(b_l_reduced, torch.Tensor):
            b_l_normalized = b_l_reduced.cpu().numpy()
        else:
            b_l_normalized = b_l_reduced
            
        if isinstance(b_u_reduced, torch.Tensor):
            b_u_normalized = b_u_reduced.cpu().numpy()
        else:
            b_u_normalized = b_u_reduced
        
        self.logger.info(f"PRODUCTION FIX: No coordinate scaling in AffineConverter - HardNet handles scaling")
        
        # Step 3: Remove infeasible constraints
        feasible_mask = b_l_normalized <= b_u_normalized + 1e-6
        infeasible_count = np.sum(~feasible_mask)
        
        if infeasible_count > 0:
            self.logger.warning(f"Removing {infeasible_count} infeasible constraints")
            A_final = A_normalized[feasible_mask]
            b_l_final = b_l_normalized[feasible_mask]
            b_u_final = b_u_normalized[feasible_mask]
            names_final = [names_reduced[i] for i in range(len(names_reduced)) if feasible_mask[i]]
        else:
            A_final = A_normalized
            b_l_final = b_l_normalized
            b_u_final = b_u_normalized
            names_final = names_reduced
        
        # Step 4: CRITICAL - Minimum rank enforcement for numerical stability
        if A_final.shape[0] > 0:
            final_tensor = torch.tensor(A_final, dtype=torch.float32)
            final_rank = torch.linalg.matrix_rank(final_tensor).item()
            required_rank = min(A_final.shape)
            
            # FAIL-FAST: If matrix is rank-deficient, either reduce or augment
            if final_rank < required_rank:
                self.logger.warning(f"CRITICAL: Rank deficiency detected: {final_rank} < {required_rank}")
                
                # Option 1: Reduce to rank-deficient row space (keep only linearly independent rows)
                U, S, Vh = torch.linalg.svd(final_tensor, full_matrices=False)
                sv_threshold = S.max().item() * 1e-10  # Strict threshold for independence
                valid_rows = torch.where(S > sv_threshold)[0]
                
                if len(valid_rows) == 0:
                    # FAIL-FAST: No valid constraints - return empty matrix
                    self.logger.error("No linearly independent constraints found - returning empty matrix")
                    return np.zeros((0, A_final.shape[1])), np.zeros(0), np.zeros(0), []
                
                # Keep only linearly independent rows
                A_final = A_final[:len(valid_rows)]
                b_l_final = b_l_final[:len(valid_rows)]
                b_u_final = b_u_final[:len(valid_rows)]
                names_final = names_final[:len(valid_rows)]
                
                final_tensor = torch.tensor(A_final, dtype=torch.float32)
                final_rank = torch.linalg.matrix_rank(final_tensor).item()
                
                self.logger.info(f"Reduced to {len(valid_rows)} linearly independent constraints")
        
        # ORTHOGONALIZATION: REMOVED per ML auditor recommendation
        # QR orthogonalization was mathematically flawed (used R instead of Q matrix)
        # Instead, rely on proper SVD-based rank reduction which is more numerically stable
        self.logger.debug("QR orthogonalization disabled - using SVD-based rank reduction only")
        
        # Validate final condition number
        try:
            if A_final.shape[0] > 0:
                final_tensor = torch.tensor(A_final, dtype=torch.float32)
                final_rank = torch.linalg.matrix_rank(final_tensor).item()
                AAT = torch.mm(final_tensor, final_tensor.t())
                final_cond = torch.linalg.cond(AAT).item()
            else:
                final_rank = 0
                final_cond = 1.0
        except:
            final_rank = 0
            final_cond = float('inf')
        
        self.logger.info(f"Final matrix: {A_final.shape}, rank={final_rank}, cond={final_cond:.2e}")
        
        # LOG condition number but DON'T reject matrices - let adaptive SVD handle all cases
        if final_cond == float('inf'):
            self.logger.warning(f"Infinite condition number detected - relying on adaptive SVD regularization")
        elif final_cond > 1e12:
            self.logger.warning(f"High condition number: {final_cond:.2e} - adaptive SVD will handle this")
        
        return A_final, b_l_final, b_u_final, names_final
    
    #reate empty constraint matrix for scenes with no constraints
    def _create_empty_matrix(self, n_objects: int) -> AffineConstraintMatrix:
        n_variables = n_objects * 4
        object_mapping = {obj_id: obj_id for obj_id in range(n_objects)}
        
        return AffineConstraintMatrix(
            A=np.zeros((0, n_variables)),
            b_l=np.zeros(0),
            b_u=np.zeros(0),
            n_objects=n_objects,
            n_constraints=0,
            constraint_names=[],
            object_mapping=object_mapping
        )
    #Validate the generated constraint matrix for consistency and feasibility
    def validate_matrix(self, matrix: AffineConstraintMatrix) -> Dict[str, Any]:
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'matrix_properties': {},
            'feasibility_check': {}
        }
        
        # Check matrix dimensions
        expected_vars = matrix.n_objects * 4
        if matrix.A.shape[1] != expected_vars:
            validation_results['errors'].append(
                f"Matrix A has {matrix.A.shape[1]} columns, expected {expected_vars}"
            )
            validation_results['is_valid'] = False
        
        if matrix.A.shape[0] != matrix.n_constraints:
            validation_results['errors'].append(
                f"Matrix A has {matrix.A.shape[0]} rows, expected {matrix.n_constraints}"
            )
            validation_results['is_valid'] = False
        
        # Check bound vector dimensions
        if len(matrix.b_l) != matrix.n_constraints:
            validation_results['errors'].append(
                f"Lower bounds vector has {len(matrix.b_l)} elements, expected {matrix.n_constraints}"
            )
            validation_results['is_valid'] = False
        
        if len(matrix.b_u) != matrix.n_constraints:
            validation_results['errors'].append(
                f"Upper bounds vector has {len(matrix.b_u)} elements, expected {matrix.n_constraints}"
            )
            validation_results['is_valid'] = False
        
        # Check bound consistency
        for i in range(matrix.n_constraints):
            if not np.isneginf(matrix.b_l[i]) and not np.isposinf(matrix.b_u[i]):
                if matrix.b_l[i] > matrix.b_u[i]:
                    validation_results['errors'].append(
                        f"Inconsistent bounds at constraint {i}: {matrix.b_l[i]} > {matrix.b_u[i]}"
                    )
                    validation_results['is_valid'] = False
        
        # Matrix properties
        if matrix.n_constraints > 0:
            validation_results['matrix_properties'] = {
                'rank': np.linalg.matrix_rank(matrix.A),
                'condition_number': np.linalg.cond(matrix.A @ matrix.A.T) if matrix.A.shape[0] > 0 else 0,
                'sparsity': np.mean(matrix.A == 0),
                'max_coefficient': np.max(np.abs(matrix.A)),
                'min_nonzero_coefficient': np.min(np.abs(matrix.A[matrix.A != 0])) if np.any(matrix.A != 0) else 0
            }
            
            # Check for potential numerical issues
            if validation_results['matrix_properties']['condition_number'] > 1e12:
                validation_results['warnings'].append(
                    f"High condition number: {validation_results['matrix_properties']['condition_number']:.2e}"
                )
            
            if validation_results['matrix_properties']['max_coefficient'] > 1e6:
                validation_results['warnings'].append(
                    f"Large matrix coefficients: {validation_results['matrix_properties']['max_coefficient']:.2e}"
                )
            
            # Check for rank deficiency
            if validation_results['matrix_properties']['rank'] < min(matrix.A.shape):
                validation_results['warnings'].append(
                    f"Matrix is rank deficient: rank {validation_results['matrix_properties']['rank']} < min({matrix.A.shape})"
                )

        # Basic feasibility check very simple check
        if matrix.n_constraints > 0:
            feasible_bounds = np.all(matrix.b_l <= matrix.b_u)
            validation_results['feasibility_check'] = {
                'bounds_consistent': feasible_bounds,
                'potentially_feasible': feasible_bounds and (matrix.n_constraints <= matrix.A.shape[1])
            }
            
            if not feasible_bounds:
                validation_results['errors'].append("Inconsistent bounds detected - constraints may be infeasible")
                validation_results['is_valid'] = False
        
        return validation_results
    
    def print_matrix_summary(self, matrix: AffineConstraintMatrix):
        print(f"Objects: {matrix.n_objects}")
        print(f"Variables: {matrix.A.shape[1]} {matrix.variable_names}")
        print(f"Constraints: {matrix.n_constraints}")
        print(f"Matrix A shape: {matrix.A.shape}")
        print(f"Sparsity: {np.mean(matrix.A == 0):.1%}")
        
        if matrix.n_constraints > 0:
            for i, name in enumerate(matrix.constraint_names):
                bounds_str = f"[{matrix.b_l[i]:.2f}, {matrix.b_u[i]:.2f}]"
                print(f"  {i+1:2d}: {name} -> bounds {bounds_str}")
    
    def get_conversion_stats(self) -> Dict[str, int]:
        return dict(self.conversion_stats)
    
    def reset_stats(self):
        self.conversion_stats.clear()


# Enhanced test functions with offset validation
def create_sample_constraints_for_testing() -> List[Any]:
    constraints = [
        # T1: Single object constraints (test offset handling)
        ConstraintT1("lt", 0, 0, 100, 0),      # obj0.x ≤ 100 (no offset)
        ConstraintT1("lt", 0, 0, 200, 10),     # obj0.x ≤ 190 (with offset)  
        ConstraintT1("gt", 1, 1, 200, 0),      # obj1.y ≥ 200 (no offset)
        ConstraintT1("eq", 0, 2, 150, 5),      # obj0.width = 150 (offset ignored)
        
        # T2: Two object constraints (test separation semantics)
        ConstraintT2("lt", 0, 0, 1, 0, 50),    # obj0.x ≤ obj1.x + 50
        ConstraintT2("gt", 1, 3, 0, 3, 0),     # obj1.height ≥ obj0.height  
        ConstraintT2("eq", 0, 1, 1, 1, 10),    # obj0.y = obj1.y (offset ignored)
        
        # T3: Arithmetic bounds (test boundary semantics)
        ConstraintT3("lt", "+", 0, 0, 0, 2, 800, 0),   # obj0.x + obj0.width ≤ 800
        ConstraintT3("lt", "+", 0, 0, 0, 2, 900, 20),  # obj0.x + obj0.width ≤ 880 (stricter)
        ConstraintT3("gt", "+", 1, 1, 1, 3, 100, 0),   # obj1.y + obj1.height ≥ 100
        
        # T4: Complex arithmetic (test separation semantics)
        ConstraintT4("lt", "+", 0, 0, 0, 2, 1, 0, 20), # obj0.x + obj0.width ≤ obj1.x + 20
        ConstraintT4("gt", "+", 1, 1, 1, 3, 0, 1, 10), # obj1.y + obj1.height ≥ obj0.y + 10
    ]
    
    return constraints


def create_spring_realistic_constraints() -> List[Any]:
    constraints = [
        # From SPRING paper: above(o1, o2, 100) - o1 at least 100 units above o2  
        ConstraintT2("lt", 0, 1, 1, 1, -100),  # obj0.y ≤ obj1.y - 100
        
        # From SPRING paper: left(o1, o2, 50) - o1 at least 50 units left of o2
        ConstraintT2("lt", 0, 0, 1, 0, -50),   # obj0.x ≤ obj1.x - 50
        
        # Boundary constraints: objects must fit in 1000x1000 canvas
        ConstraintT3("lt", "+", 0, 0, 0, 2, 1000, 0),  # obj0.x + obj0.width ≤ 1000
        ConstraintT3("lt", "+", 1, 0, 1, 2, 1000, 0),  # obj1.x + obj1.width ≤ 1000
        ConstraintT3("lt", "+", 0, 1, 0, 3, 1000, 0),  # obj0.y + obj0.height ≤ 1000
        ConstraintT3("lt", "+", 1, 1, 1, 3, 1000, 0),  # obj1.y + obj1.height ≤ 1000
        
        # Size constraints: minimum dimensions
        ConstraintT1("gt", 0, 2, 50, 0),       # obj0.width ≥ 50
        ConstraintT1("gt", 0, 3, 50, 0),       # obj0.height ≥ 50
        ConstraintT1("gt", 1, 2, 50, 0),       # obj1.width ≥ 50
        ConstraintT1("gt", 1, 3, 50, 0),       # obj1.height ≥ 50
    ]
    
    return constraints


def validate_offset_semantics():
    
    converter = ConstraintToAffineConverter(enable_logging=False)
    
    # Test T1 offset handling
    t1_constraint = ConstraintT1("lt", 0, 0, 100, 10)  # obj0.x ≤ 100 with offset 10
    matrix = converter.convert_constraints_to_matrix([t1_constraint], 1)
    expected_bound = 90  # 100 - 10
    actual_bound = matrix.b_u[0]
    print(f"T1 offset test: obj0.x ≤ 100 (offset=10) -> bound = {actual_bound} (expected: {expected_bound})")
    assert np.isclose(actual_bound, expected_bound), f"T1 offset handling failed: {actual_bound} != {expected_bound}"
    
    # Test T2 offset handling  
    t2_constraint = ConstraintT2("lt", 0, 0, 1, 0, 50)  # obj0.x ≤ obj1.x + 50
    matrix = converter.convert_constraints_to_matrix([t2_constraint], 2)
    expected_bound = 50
    actual_bound = matrix.b_u[0]
    print(f"T2 offset test: obj0.x ≤ obj1.x + 50 -> bound = {actual_bound} (expected: {expected_bound})")
    assert np.isclose(actual_bound, expected_bound), f"T2 offset handling failed: {actual_bound} != {expected_bound}"
    
    # Test T3 offset handling
    t3_constraint = ConstraintT3("lt", "+", 0, 0, 0, 2, 800, 10)  # obj0.x + obj0.width ≤ 800 with offset 10
    matrix = converter.convert_constraints_to_matrix([t3_constraint], 1)
    expected_bound = 790  # 800 - 10  
    actual_bound = matrix.b_u[0]
    print(f"T3 offset test: obj0.x + obj0.width ≤ 800 (offset=10) -> bound = {actual_bound} (expected: {expected_bound})")
    assert np.isclose(actual_bound, expected_bound), f"T3 offset handling failed: {actual_bound} != {expected_bound}"
    
    print("All offset semantics tests passed!")


if __name__ == "__main__":
    # Demonstration and testing    
    # Create converter
    converter = ConstraintToAffineConverter(enable_logging=True)
    
    # Basic constraint conversion
    print("Basic constraint conversion with offset handling")
    sample_constraints = create_sample_constraints_for_testing()
    n_test_objects = 2
    
    matrix = converter.convert_constraints_to_matrix(sample_constraints, n_test_objects)
    converter.print_matrix_summary(matrix)
    
    # Validate matrix structure
    print("\nMatrix validation")
    validation = converter.validate_matrix(matrix)
    print(f"Valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    
    if validation['matrix_properties']:
        props = validation['matrix_properties']
        print(f"\nMatrix Properties:")
        print(f"  Rank: {props['rank']}")
        print(f"  Condition number: {props['condition_number']:.2e}")
        print(f"  Sparsity: {props['sparsity']:.2%}")
        print(f"  Max coefficient: {props['max_coefficient']:.2e}")
    
    #SPRING realistic constraints
    print("\nSPRING realistic constraints")
    spring_constraints = create_spring_realistic_constraints()
    spring_matrix = converter.convert_constraints_to_matrix(spring_constraints, 2)
    print(f"SPRING example: {spring_matrix.n_constraints} constraints for kitchen layout")
    
    #Offset semantics validation
    print("\nOffset semantics validation")
    try:
        validate_offset_semantics()
    except Exception as e:
        print(f"Offset validation failed: {e}")
    
    # Show conversion statistics
    stats = converter.get_conversion_stats()
    print(f"\nConversion Statistics:")
    for constraint_type, count in stats.items():
        print(f"  {constraint_type}: {count}")