"""
SPRING-HardNet Extended Constraint Language
==========================================

Extended constraint types for implicit constraint generation:
- BoundaryConstraint: Keep objects within canvas
- SizeRatioConstraint: Enforce realistic size relationships  
- NonOverlapConstraint: Prevent object collisions
- RelativeToExistingConstraint: Size relative to existing objects
- MinimumSpacingConstraint: Aesthetic spacing

These extend the existing constraint_language_v2.py with new constraint types
that can be converted to affine form A(x)y ≤ b(x) by the affine converter.
"""

from collections import namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

# Import existing constraint types for compatibility
try:
    from constraint_language_v2 import (
        ConstraintT1, ConstraintT2, ConstraintT3, ConstraintT4, ConstraintT5, ConstraintT6,
        ConstraintOR, ConstraintAND, ConstraintNOT
    )
except ImportError:
    # Define basic constraint types if not available
    ConstraintT1 = namedtuple("ConstraintT1", ["c", "o1", "v1", "val", "offset"])
    ConstraintT2 = namedtuple("ConstraintT2", ["c", "o1", "v1", "o2", "v2", "offset"])


# ======================================================================================
# NEW CONSTRAINT TYPES FOR IMPLICIT CONSTRAINTS
# ======================================================================================

# Boundary Constraints (objects must stay within canvas)
BoundaryConstraint = namedtuple("BoundaryConstraint", [
    "constraint_type",  # 'left', 'right', 'top', 'bottom'
    "obj_idx",          # Object index
    "canvas_size",      # Canvas width/height
    "margin"            # Safety margin
])

# Size Ratio Constraints (obj1.size <= ratio * obj2.size)
SizeRatioConstraint = namedtuple("SizeRatioConstraint", [
    "smaller_obj_idx",  # Index of smaller object
    "larger_obj_idx",   # Index of larger object  
    "dimension",        # 'width' or 'height'
    "max_ratio",        # Maximum size ratio (≤ 1.0)
    "constraint_source" # 'fixed_scale' or 'scene_adaptive'
])

# Non-Overlap Constraints (prevent object collisions)
NonOverlapConstraint = namedtuple("NonOverlapConstraint", [
    "obj1_idx",         # First object index
    "obj2_idx",         # Second object index
    "separation_axis",  # 'horizontal' or 'vertical'
    "min_margin",       # Minimum separation distance
    "spatial_relation"  # Optional: 'left', 'right', 'above', 'below'
])

# Relative to Existing Constraints (size relative to detected/background objects)
RelativeToExistingConstraint = namedtuple("RelativeToExistingConstraint", [
    "new_obj_idx",      # Index of new object being sized
    "existing_obj",     # Dict with existing object info {'bbox': [x,y,w,h], 'category': str}
    "dimension",        # 'width' or 'height'
    "max_ratio",        # Maximum size ratio relative to existing object
    "reference_type"    # 'detected' or 'background'
])

# Minimum Spacing Constraints (aesthetic spacing between objects)
MinimumSpacingConstraint = namedtuple("MinimumSpacingConstraint", [
    "obj1_idx",         # First object index
    "obj2_idx",         # Second object index 
    "min_distance",     # Minimum center-to-center distance
    "distance_type"     # 'euclidean', 'manhattan', 'bounding_box'
])

# Composite Boundary Constraint (all boundaries for one object)
CompositeBoundaryConstraint = namedtuple("CompositeBoundaryConstraint", [
    "obj_idx",          # Object index
    "canvas_width",     # Canvas dimensions
    "canvas_height",
    "margin_left",      # Individual margins for each side
    "margin_right", 
    "margin_top",
    "margin_bottom"
])


# ======================================================================================
# CONSTRAINT CREATION HELPER FUNCTIONS
# ======================================================================================

def create_boundary_constraints(obj_idx: int, 
                               canvas_width: float = 1.0, 
                               canvas_height: float = 1.0,
                               margin: float = 0.01) -> List[BoundaryConstraint]:
    """
    Create all boundary constraints for an object.
    
    Generates:
    - x >= margin (left boundary)
    - y >= margin (top boundary) 
    - x + w <= canvas_width - margin (right boundary)
    - y + h <= canvas_height - margin (bottom boundary)
    """
    
    return [
        BoundaryConstraint('left', obj_idx, canvas_width, margin),
        BoundaryConstraint('right', obj_idx, canvas_width, margin),
        BoundaryConstraint('top', obj_idx, canvas_height, margin),
        BoundaryConstraint('bottom', obj_idx, canvas_height, margin)
    ]


def create_size_ratio_constraints(smaller_obj_idx: int,
                                 larger_obj_idx: int, 
                                 width_ratio: float,
                                 height_ratio: float,
                                 constraint_source: str = 'fixed_scale') -> List[SizeRatioConstraint]:
    """
    Create size ratio constraints between two objects.
    
    Generates:
    - smaller_obj.width <= width_ratio * larger_obj.width
    - smaller_obj.height <= height_ratio * larger_obj.height
    """
    
    return [
        SizeRatioConstraint(smaller_obj_idx, larger_obj_idx, 'width', width_ratio, constraint_source),
        SizeRatioConstraint(smaller_obj_idx, larger_obj_idx, 'height', height_ratio, constraint_source)
    ]


def create_non_overlap_constraint_from_spatial_relation(obj1_idx: int,
                                                       obj2_idx: int,
                                                       spatial_relation: str,
                                                       margin: float = 0.03) -> NonOverlapConstraint:
    """
    Create non-overlap constraint from spatial relation.
    
    Args:
        spatial_relation: 'left', 'right', 'above', 'below'
        
    Returns:
        NonOverlapConstraint ensuring objects don't overlap given the spatial relation
    """
    
    if spatial_relation in ['left', 'right']:
        return NonOverlapConstraint(obj1_idx, obj2_idx, 'horizontal', margin, spatial_relation)
    elif spatial_relation in ['above', 'below']:
        return NonOverlapConstraint(obj1_idx, obj2_idx, 'vertical', margin, spatial_relation)
    else:
        raise ValueError(f"Unknown spatial relation: {spatial_relation}")


def create_relative_to_existing_constraints(new_obj_idx: int,
                                           existing_obj: Dict[str, Any],
                                           width_ratio: float,
                                           height_ratio: float,
                                           reference_type: str = 'detected') -> List[RelativeToExistingConstraint]:
    """
    Create constraints to size new object relative to existing object.
    
    Args:
        existing_obj: Dict with 'bbox' [x,y,w,h] and 'category' keys
        width_ratio: max_ratio for width constraint
        height_ratio: max_ratio for height constraint
    """
    
    return [
        RelativeToExistingConstraint(new_obj_idx, existing_obj, 'width', width_ratio, reference_type),
        RelativeToExistingConstraint(new_obj_idx, existing_obj, 'height', height_ratio, reference_type)
    ]


# ======================================================================================
# AFFINE CONSTRAINT CONVERSION HELPERS  
# ======================================================================================

class ExtendedConstraintToAffineConverter:
    """
    Converter for extended constraint types to affine matrix form.
    
    Converts constraint types to the form A(x)y ≤ b(x) where:
    - A(x): constraint coefficient matrix
    - y: layout variables [x1, y1, w1, h1, x2, y2, w2, h2, ...]
    - b(x): constraint bounds
    """
    
    def __init__(self, coordinate_system: str = "normalized"):
        """
        Args:
            coordinate_system: "normalized" [0,1] or "per_mille" [0,1000]
        """
        self.coordinate_system = coordinate_system
        self.coord_scale = 1.0 if coordinate_system == "normalized" else 1000.0
    
    def convert_boundary_constraint(self, 
                                  constraint: BoundaryConstraint, 
                                  n_objects: int) -> Tuple[np.ndarray, float, float]:
        """
        Convert boundary constraint to affine form.
        
        Returns:
            (A_row, b_lower, b_upper) for constraint A_row * y ∈ [b_lower, b_upper]
        """
        
        # Create coefficient row (4 variables per object: x, y, w, h)
        A_row = np.zeros(n_objects * 4)
        obj_idx = constraint.obj_idx
        
        if constraint.constraint_type == 'left':
            # x >= margin  →  -x <= -margin  →  A_row = [-1, 0, 0, 0, ...], b_upper = -margin
            A_row[obj_idx * 4] = -1.0  # x coefficient
            b_lower = -np.inf
            b_upper = -constraint.margin * self.coord_scale
            
        elif constraint.constraint_type == 'top': 
            # y >= margin  →  -y <= -margin
            A_row[obj_idx * 4 + 1] = -1.0  # y coefficient  
            b_lower = -np.inf
            b_upper = -constraint.margin * self.coord_scale
            
        elif constraint.constraint_type == 'right':
            # x + w <= canvas_width - margin  →  A_row = [1, 0, 1, 0, ...], b_upper = canvas_width - margin
            A_row[obj_idx * 4] = 1.0      # x coefficient
            A_row[obj_idx * 4 + 2] = 1.0  # w coefficient
            b_lower = -np.inf
            b_upper = (constraint.canvas_size - constraint.margin) * self.coord_scale
            
        elif constraint.constraint_type == 'bottom':
            # y + h <= canvas_height - margin
            A_row[obj_idx * 4 + 1] = 1.0  # y coefficient
            A_row[obj_idx * 4 + 3] = 1.0  # h coefficient
            b_lower = -np.inf  
            b_upper = (constraint.canvas_size - constraint.margin) * self.coord_scale
            
        else:
            raise ValueError(f"Unknown boundary constraint type: {constraint.constraint_type}")
        
        return A_row, b_lower, b_upper
    
    def convert_size_ratio_constraint(self,
                                    constraint: SizeRatioConstraint,
                                    n_objects: int) -> Tuple[np.ndarray, float, float]:
        """
        Convert size ratio constraint to affine form.
        
        Size constraint: smaller_obj.dim <= max_ratio * larger_obj.dim
        Rearranged: smaller_obj.dim - max_ratio * larger_obj.dim <= 0
        """
        
        A_row = np.zeros(n_objects * 4)
        smaller_idx = constraint.smaller_obj_idx
        larger_idx = constraint.larger_obj_idx
        max_ratio = constraint.max_ratio
        
        if constraint.dimension == 'width':
            # w_small - max_ratio * w_large <= 0
            A_row[smaller_idx * 4 + 2] = 1.0         # w_small coefficient
            A_row[larger_idx * 4 + 2] = -max_ratio   # w_large coefficient
        elif constraint.dimension == 'height':
            # h_small - max_ratio * h_large <= 0  
            A_row[smaller_idx * 4 + 3] = 1.0         # h_small coefficient
            A_row[larger_idx * 4 + 3] = -max_ratio   # h_large coefficient
        else:
            raise ValueError(f"Unknown dimension: {constraint.dimension}")
        
        b_lower = -np.inf
        b_upper = 0.0
        
        return A_row, b_lower, b_upper
    
    def convert_non_overlap_constraint(self,
                                     constraint: NonOverlapConstraint,
                                     n_objects: int) -> Tuple[np.ndarray, float, float]:
        """
        Convert non-overlap constraint to affine form.
        
        Horizontal non-overlap: x1 + w1 + margin <= x2
        Rearranged: x1 + w1 - x2 <= -margin
        
        Vertical non-overlap: y1 + h1 + margin <= y2
        Rearranged: y1 + h1 - y2 <= -margin
        """
        
        A_row = np.zeros(n_objects * 4)
        obj1_idx = constraint.obj1_idx
        obj2_idx = constraint.obj2_idx
        margin = constraint.min_margin * self.coord_scale
        
        if constraint.separation_axis == 'horizontal':
            # x1 + w1 - x2 <= -margin
            A_row[obj1_idx * 4] = 1.0      # x1 coefficient
            A_row[obj1_idx * 4 + 2] = 1.0  # w1 coefficient  
            A_row[obj2_idx * 4] = -1.0     # x2 coefficient
            b_upper = -margin
            
        elif constraint.separation_axis == 'vertical':
            # y1 + h1 - y2 <= -margin
            A_row[obj1_idx * 4 + 1] = 1.0  # y1 coefficient
            A_row[obj1_idx * 4 + 3] = 1.0  # h1 coefficient
            A_row[obj2_idx * 4 + 1] = -1.0 # y2 coefficient
            b_upper = -margin
            
        else:
            raise ValueError(f"Unknown separation axis: {constraint.separation_axis}")
        
        b_lower = -np.inf
        return A_row, b_lower, b_upper
    
    def convert_relative_to_existing_constraint(self,
                                              constraint: RelativeToExistingConstraint,
                                              n_objects: int) -> Tuple[np.ndarray, float, float]:
        """
        Convert relative-to-existing constraint to affine form.
        
        Constraint: new_obj.dim <= max_ratio * existing_obj.dim
        Since existing object has fixed dimensions, this becomes:
        new_obj.dim <= constant_value
        """
        
        A_row = np.zeros(n_objects * 4)
        new_obj_idx = constraint.new_obj_idx
        existing_bbox = constraint.existing_obj['bbox']  # [x, y, w, h]
        max_ratio = constraint.max_ratio
        
        if constraint.dimension == 'width':
            # w_new <= max_ratio * w_existing
            A_row[new_obj_idx * 4 + 2] = 1.0  # w_new coefficient
            existing_width = existing_bbox[2] * self.coord_scale
            b_upper = max_ratio * existing_width
            
        elif constraint.dimension == 'height':
            # h_new <= max_ratio * h_existing
            A_row[new_obj_idx * 4 + 3] = 1.0  # h_new coefficient
            existing_height = existing_bbox[3] * self.coord_scale
            b_upper = max_ratio * existing_height
            
        else:
            raise ValueError(f"Unknown dimension: {constraint.dimension}")
        
        b_lower = -np.inf
        return A_row, b_lower, b_upper


# ======================================================================================
# VALIDATION AND UTILITIES
# ======================================================================================

def validate_extended_constraint(constraint: Any) -> bool:
    """Validate an extended constraint object"""
    
    if isinstance(constraint, BoundaryConstraint):
        return (constraint.constraint_type in ['left', 'right', 'top', 'bottom'] and
                constraint.obj_idx >= 0 and
                constraint.canvas_size > 0 and
                0 <= constraint.margin < constraint.canvas_size)
    
    elif isinstance(constraint, SizeRatioConstraint):
        return (constraint.smaller_obj_idx >= 0 and
                constraint.larger_obj_idx >= 0 and
                constraint.smaller_obj_idx != constraint.larger_obj_idx and
                constraint.dimension in ['width', 'height'] and
                0 < constraint.max_ratio <= 1.0)
    
    elif isinstance(constraint, NonOverlapConstraint):
        return (constraint.obj1_idx >= 0 and
                constraint.obj2_idx >= 0 and
                constraint.obj1_idx != constraint.obj2_idx and
                constraint.separation_axis in ['horizontal', 'vertical'] and
                constraint.min_margin >= 0)
    
    elif isinstance(constraint, RelativeToExistingConstraint):
        return (constraint.new_obj_idx >= 0 and
                isinstance(constraint.existing_obj, dict) and
                'bbox' in constraint.existing_obj and
                constraint.dimension in ['width', 'height'] and
                constraint.max_ratio > 0)
    
    return False


def get_constraint_description(constraint: Any) -> str:
    """Get human-readable description of constraint"""
    
    if isinstance(constraint, BoundaryConstraint):
        return f"Object {constraint.obj_idx} {constraint.constraint_type} boundary (margin: {constraint.margin})"
    
    elif isinstance(constraint, SizeRatioConstraint):
        return f"Object {constraint.smaller_obj_idx} {constraint.dimension} ≤ {constraint.max_ratio:.2f} × Object {constraint.larger_obj_idx} {constraint.dimension}"
    
    elif isinstance(constraint, NonOverlapConstraint):
        return f"Objects {constraint.obj1_idx}, {constraint.obj2_idx} non-overlap ({constraint.separation_axis}, margin: {constraint.min_margin})"
    
    elif isinstance(constraint, RelativeToExistingConstraint):
        existing_category = constraint.existing_obj.get('category', 'unknown')
        return f"Object {constraint.new_obj_idx} {constraint.dimension} ≤ {constraint.max_ratio:.2f} × existing {existing_category}"
    
    return f"Unknown constraint type: {type(constraint)}"


# ======================================================================================
# EXAMPLE USAGE AND TESTING
# ======================================================================================

if __name__ == "__main__":
    print("🧪 Testing Extended Constraint Language")
    
    # Test boundary constraints
    print("\n📋 Boundary Constraints:")
    boundary_constraints = create_boundary_constraints(obj_idx=0, canvas_width=1.0, canvas_height=1.0, margin=0.01)
    for constraint in boundary_constraints:
        print(f"  - {get_constraint_description(constraint)}")
        print(f"    Valid: {validate_extended_constraint(constraint)}")
    
    # Test size ratio constraints
    print("\n📋 Size Ratio Constraints:")
    size_constraints = create_size_ratio_constraints(
        smaller_obj_idx=0, larger_obj_idx=1, 
        width_ratio=0.5, height_ratio=0.4, 
        constraint_source='scene_adaptive'
    )
    for constraint in size_constraints:
        print(f"  - {get_constraint_description(constraint)}")
        print(f"    Valid: {validate_extended_constraint(constraint)}")
    
    # Test non-overlap constraints
    print("\n📋 Non-Overlap Constraints:")
    non_overlap = create_non_overlap_constraint_from_spatial_relation(
        obj1_idx=0, obj2_idx=1, spatial_relation='left', margin=0.02
    )
    print(f"  - {get_constraint_description(non_overlap)}")
    print(f"    Valid: {validate_extended_constraint(non_overlap)}")
    
    # Test relative-to-existing constraints
    print("\n📋 Relative-to-Existing Constraints:")
    existing_obj = {
        'bbox': [0.2, 0.3, 0.15, 0.12],  # [x, y, w, h]
        'category': 'oven'
    }
    relative_constraints = create_relative_to_existing_constraints(
        new_obj_idx=0, existing_obj=existing_obj,
        width_ratio=0.3, height_ratio=0.4
    )
    for constraint in relative_constraints:
        print(f"  - {get_constraint_description(constraint)}")
        print(f"    Valid: {validate_extended_constraint(constraint)}")
    
    # Test affine conversion
    print("\n📋 Affine Conversion Test:")
    converter = ExtendedConstraintToAffineConverter(coordinate_system="normalized")
    
    # Convert a boundary constraint to affine form
    boundary_constraint = boundary_constraints[0]  # Left boundary
    A_row, b_lower, b_upper = converter.convert_boundary_constraint(boundary_constraint, n_objects=2)
    
    print(f"  Boundary constraint: {get_constraint_description(boundary_constraint)}")
    print(f"  Affine form: A_row = {A_row}")  
    print(f"  Bounds: [{b_lower}, {b_upper}]")
    
    print("\n✅ Extended constraint language tests completed!")