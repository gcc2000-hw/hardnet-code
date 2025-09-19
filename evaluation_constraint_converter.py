#!/usr/bin/env python3
"""
HYBRID SPRING 10K EVALUATION - CONSTRAINT CONVERTER
Convert evaluation JSON constraints to internal constraint objects.
No fallbacks, no mocks - if it fails, it fails clearly.
"""

import sys
import json
from typing import Dict, List, Any, Tuple
from constraint_language_v2 import (
    ConstraintT1, ConstraintT2, ConstraintOR,
    con_left, con_right, con_above, con_below,
    con_wider, con_taller, con_narrower, con_shorter,
    con_xeq, con_yeq, con_left_val, con_right_val
)

class EvaluationConstraintConverter:
    """Convert evaluation JSON constraints to internal constraint objects."""
    
    def __init__(self):
        self.object_name_to_idx = {}
        self.existing_objects = []  # List of existing object names from DETR detection
        self.existing_positions = {}  # {obj_name: [x, y, w, h]} for existing objects
        self.new_objects = []  # List of new object names to be generated
    
    def set_object_mapping(self, object_names: List[str]):
        """Set the mapping from object names to indices."""
        self.object_name_to_idx = {name: idx for idx, name in enumerate(object_names)}
        print(f"Object mapping: {self.object_name_to_idx}")
    
    def set_mixed_scenario(self, existing_objects: List[str], existing_positions: Dict[str, List[float]], new_objects: List[str]):
        """Set up mixed constraint scenario with existing and new objects.
        
        Args:
            existing_objects: List of existing object names from DETR detection
            existing_positions: Dict mapping existing object names to [x, y, w, h] positions
            new_objects: List of new object names to be generated
        """
        self.existing_objects = existing_objects.copy()
        self.existing_positions = existing_positions.copy()
        self.new_objects = new_objects.copy()
        
        # Create combined object mapping: [existing_0, existing_1, ..., new_0, new_1, ...]
        all_objects = existing_objects + new_objects
        self.set_object_mapping(all_objects)
        
        print(f"Mixed scenario set up:")
        print(f"  Existing objects: {existing_objects}")
        print(f"  New objects: {new_objects}")
        print(f"  Combined mapping: {self.object_name_to_idx}")
        
        # PHASE 3D: Flexible object indexing - constraints work with any object order
        # The constraint system is robust to different object orderings
        existing_indices = [self.object_name_to_idx[obj] for obj in existing_objects if obj in self.object_name_to_idx]
        new_indices = [self.object_name_to_idx[obj] for obj in new_objects if obj in self.object_name_to_idx]
        
        print(f"  ✓ FLEXIBLE INDEXING: {len(existing_objects)} existing + {len(new_objects)} new objects")
        print(f"    Existing indices: {existing_indices}")  
        print(f"    New indices: {new_indices}")
    
    def _get_furniture_typical_width(self, obj_name: str) -> float:
        """Get typical width for furniture object based on furniture priors."""
        furniture_priors = {
            'chair': {'typical_size': 0.04, 'aspect_ratio': (0.7, 1.4)},
            'table': {'typical_size': 0.08, 'aspect_ratio': (0.8, 2.5)},
            'couch': {'typical_size': 0.12, 'aspect_ratio': (1.8, 3.5)},
            'sofa': {'typical_size': 0.12, 'aspect_ratio': (1.8, 3.5)},
            'bed': {'typical_size': 0.15, 'aspect_ratio': (1.2, 2.0)},
        }
        
        default_prior = {'typical_size': 0.06, 'aspect_ratio': (1.0, 1.5)}
        prior = furniture_priors.get(obj_name.lower(), default_prior)
        
        # Calculate typical width from area and aspect ratio
        typical_area = prior['typical_size']
        min_aspect, max_aspect = prior['aspect_ratio']
        avg_aspect = (min_aspect + max_aspect) / 2  # Use average aspect ratio
        
        # area = width * height, aspect = width / height
        # So: area = width * (width / aspect) = width^2 / aspect
        # Therefore: width = sqrt(area * aspect)
        typical_width = (typical_area * avg_aspect) ** 0.5
        
        return typical_width
    
    def _get_aspect_ratio(self, obj_name: str) -> tuple:
        """Get aspect ratio range for furniture object."""
        furniture_priors = {
            'chair': (0.7, 1.4), 'table': (0.8, 2.5), 'couch': (1.8, 3.5),
            'sofa': (1.8, 3.5), 'bed': (1.2, 2.0)
        }
        return furniture_priors.get(obj_name.lower(), (1.0, 1.5))
    
    def _calculate_dynamic_separation(self, obj1_name: str, obj2_name: str, direction: str) -> float:
        """Calculate required separation between two objects to prevent overlap in MODEL COORDINATE SYSTEM."""
        
        # Get typical widths for both objects (in [0,1] normalized space)
        obj1_width = self._get_furniture_typical_width(obj1_name)
        obj2_width = self._get_furniture_typical_width(obj2_name)
        
        if direction in ['left', 'right']:
            # For horizontal separation, need: obj1_half_width + obj2_half_width + safety_margin
            required_separation_norm = (obj1_width + obj2_width) / 2 + 0.02  # In [0,1] space
        else:  # above, below
            # For vertical separation, use height-based calculation
            obj1_height = obj1_width / ((self._get_aspect_ratio(obj1_name)[0] + self._get_aspect_ratio(obj1_name)[1]) / 2)
            obj2_height = obj2_width / ((self._get_aspect_ratio(obj2_name)[0] + self._get_aspect_ratio(obj2_name)[1]) / 2)
            required_separation_norm = (obj1_height + obj2_height) / 2 + 0.02  # In [0,1] space
        
        # COORDINATE SYSTEM FIX: Remove incorrect 2.4x scaling
        # Analysis shows model operates in coordinate range similar to [0,1], not [-1.2, 1.2]
        # Training evidence: Target range [-1.0, 1.9], Predicted range [-0.6, 0.4]
        # The 2.4x scaling created impossible constraints by making separations too large
        # Keep separations in the model's natural coordinate space
        
        return required_separation_norm  # No scaling needed - use natural separation
    
    def _are_constraints_compatible(self, constraint1: Dict[str, Any], constraint2: Dict[str, Any]) -> bool:
        """Check if two constraints on the same object pair are compatible for AND grouping."""
        type1, type2 = constraint1['type'], constraint2['type']
        
        # Identical constraints are compatible (redundant but harmless)
        if type1 == type2:
            return True
        
        # Define contradictory pairs
        contradictory_pairs = {
            ('left', 'right'), ('right', 'left'),
            ('above', 'below'), ('below', 'above'),
            ('bigger', 'smaller'), ('smaller', 'bigger')
        }
        
        # Check for direct contradictions
        if (type1, type2) in contradictory_pairs:
            return False
        
        # Alignment constraints contradict with spatial constraints on same axis
        alignment_spatial_conflicts = {
            ('vertically_aligned', 'left'), ('vertically_aligned', 'right'),
            ('left', 'vertically_aligned'), ('right', 'vertically_aligned'),
            ('horizontally_aligned', 'above'), ('horizontally_aligned', 'below'),
            ('above', 'horizontally_aligned'), ('below', 'horizontally_aligned')
        }
        
        if (type1, type2) in alignment_spatial_conflicts:
            return False
        
        # All other combinations are compatible
        return True
    
    def _group_constraints_by_object_pair(self, eval_constraints: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], List[Dict]], List[Dict]]:
        """Group constraints by object pairs, return grouped and ungrouped constraints."""
        grouped = {}  # {(obj1, obj2): [constraint1, constraint2, ...]}
        ungrouped = []  # Constraints that don't have object pairs (like OR)
        
        for constraint in eval_constraints:
            constraint_type = constraint['type']
            
            # Only group constraints that have object1 and object2
            if constraint_type in ['left', 'right', 'above', 'below', 'bigger', 'smaller', 
                                 'horizontally_aligned', 'vertically_aligned']:
                obj1 = constraint['object1']
                obj2 = constraint['object2']
                pair = (obj1, obj2)
                
                if pair not in grouped:
                    grouped[pair] = []
                grouped[pair].append(constraint)
            else:
                # OR constraints and others that don't have simple object pairs
                ungrouped.append(constraint)
        
        return grouped, ungrouped
    
    def create_and_compatible_specification(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specification with AND-compatible JSON format for pipeline consumption."""
        # Make a deep copy to avoid modifying original
        import copy
        modified_spec = copy.deepcopy(specification)
        
        # Group constraints by object pairs  
        grouped_constraints, ungrouped_constraints = self._group_constraints_by_object_pair(specification['constraints'])
        
        # Get object lists for compatibility checking
        existing_objects = specification.get('existing_objects', [])
        new_objects = specification.get('objects', [])
        
        new_constraints = []
        and_groups_created = 0
        
        # Process grouped constraints
        for object_pair, constraint_group in grouped_constraints.items():
            if len(constraint_group) == 1:
                # Single constraint - add as-is
                new_constraints.append(constraint_group[0])
            else:
                # Multiple constraints - check compatibility
                all_compatible = True
                for i in range(len(constraint_group)):
                    for j in range(i + 1, len(constraint_group)):
                        if not self._are_constraints_compatible(constraint_group[i], constraint_group[j]):
                            all_compatible = False
                            print(f"WARNING: Incompatible constraints for {object_pair}: {constraint_group[i]['type']} vs {constraint_group[j]['type']}")
                            break
                    if not all_compatible:
                        break
                
                if all_compatible and len(constraint_group) > 1:
                    # Create AND JSON constraint
                    # First apply constraint flipping if needed
                    flipped_group = []
                    for constraint in constraint_group:
                        flipped_constraint = self._flip_constraint_if_needed_json(constraint, existing_objects, new_objects)
                        flipped_group.append(flipped_constraint)
                    
                    and_constraint = {
                        "type": "and", 
                        "conditions": flipped_group
                    }
                    new_constraints.append(and_constraint)
                    and_groups_created += 1
                    print(f"Created AND JSON constraint for {object_pair} with {len(constraint_group)} conditions")
                else:
                    # Incompatible - add individually (with flipping if needed)
                    for constraint in constraint_group:
                        flipped_constraint = self._flip_constraint_if_needed_json(constraint, existing_objects, new_objects)
                        new_constraints.append(flipped_constraint)
        
        # Add ungrouped constraints (with flipping if needed)
        for constraint in ungrouped_constraints:
            flipped_constraint = self._flip_constraint_if_needed_json(constraint, existing_objects, new_objects)
            new_constraints.append(flipped_constraint)
        
        # Update specification with new constraints
        modified_spec['constraints'] = new_constraints
        
        print(f"Specification modified: {len(specification['constraints'])} -> {len(new_constraints)} constraints (created {and_groups_created} AND groups)")
        return modified_spec
    
    def _flip_constraint_if_needed_json(self, constraint: Dict[str, Any], existing_objects: List[str], new_objects: List[str]) -> Dict[str, Any]:
        """JSON version of constraint flipping logic."""
        import copy
        ctype = constraint.get('type')
        obj1 = constraint.get('object1')
        obj2 = constraint.get('object2')
        
        # Only handle spatial constraints that can be flipped
        if ctype not in ['left', 'right', 'above', 'below']:
            return constraint
        
        # Check if obj1 is existing and obj2 is new (invalid - need to flip)
        if obj1 in existing_objects and obj2 in new_objects:
            # Define flip mappings
            flips = {
                'left': 'right',
                'right': 'left', 
                'above': 'below',
                'below': 'above'
            }
            
            flipped_constraint = copy.deepcopy(constraint)
            flipped_constraint['type'] = flips[ctype]
            flipped_constraint['object1'] = obj2  # Swap objects
            flipped_constraint['object2'] = obj1
            
            print(f"  Flipped JSON: {constraint} -> {flipped_constraint}")
            return flipped_constraint
        
        return constraint

    def convert_constraints(self, eval_constraints: List[Dict[str, Any]]) -> List[Any]:
        """Convert all evaluation constraints to internal format with AND grouping."""
        internal_constraints = []
        
        # Group constraints by object pairs
        grouped_constraints, ungrouped_constraints = self._group_constraints_by_object_pair(eval_constraints)
        
        # Process grouped constraints (potential AND combinations)
        and_groups_created = 0
        for object_pair, constraint_group in grouped_constraints.items():
            if len(constraint_group) == 1:
                # Single constraint for this pair - process normally
                try:
                    converted = self._convert_single_constraint(constraint_group[0])
                    if converted is not None:
                        if isinstance(converted, list):
                            internal_constraints.extend(converted)
                        else:
                            internal_constraints.append(converted)
                except Exception as e:
                    print(f"ERROR: Failed to convert constraint: {constraint_group[0]}")
                    print(f"ERROR: {e}")
                    raise
            else:
                # Multiple constraints for same object pair - check compatibility
                all_compatible = True
                for i in range(len(constraint_group)):
                    for j in range(i + 1, len(constraint_group)):
                        if not self._are_constraints_compatible(constraint_group[i], constraint_group[j]):
                            all_compatible = False
                            print(f"WARNING: Incompatible constraints for {object_pair}: {constraint_group[i]['type']} vs {constraint_group[j]['type']}")
                            break
                    if not all_compatible:
                        break
                
                if all_compatible:
                    # HARDNET FIX: Add individual constraints instead of creating ConstraintAND
                    try:
                        converted_subconstraints = []
                        for constraint in constraint_group:
                            converted = self._convert_single_constraint(constraint)
                            if converted is not None:
                                if isinstance(converted, list):
                                    converted_subconstraints.extend(converted)
                                else:
                                    converted_subconstraints.append(converted)
                        
                        # Add all individual constraints directly (no ConstraintAND wrapper)
                        internal_constraints.extend(converted_subconstraints)
                        and_groups_created += len(converted_subconstraints)
                        print(f"Added {len(converted_subconstraints)} individual constraints for {object_pair} (was {len(constraint_group)} grouped conditions)")
                    except Exception as e:
                        print(f"ERROR: Failed to process constraint group for {object_pair}: {constraint_group}")
                        print(f"ERROR: {e}")
                        raise
                else:
                    # Incompatible constraints - process individually with warning
                    print(f"WARNING: Processing incompatible constraints individually for {object_pair}")
                    for constraint in constraint_group:
                        try:
                            converted = self._convert_single_constraint(constraint)
                            if converted is not None:
                                if isinstance(converted, list):
                                    internal_constraints.extend(converted)
                                else:
                                    internal_constraints.append(converted)
                        except Exception as e:
                            print(f"ERROR: Failed to convert constraint: {constraint}")
                            print(f"ERROR: {e}")
                            raise
        
        # Process ungrouped constraints normally
        for constraint in ungrouped_constraints:
            try:
                converted = self._convert_single_constraint(constraint)
                if converted is not None:
                    if isinstance(converted, list):
                        internal_constraints.extend(converted)
                    else:
                        internal_constraints.append(converted)
                    print(f"Converted ungrouped constraint: {constraint['type']}")
            except Exception as e:
                print(f"ERROR: Failed to convert ungrouped constraint: {constraint}")
                print(f"ERROR: {e}")
                raise
        
        # DEDUPLICATION FIX: Remove mathematically equivalent T2 constraints
        # This prevents "A right of B" + "B left of A" from creating duplicate matrix rows
        deduplicated_constraints = self._deduplicate_equivalent_constraints(internal_constraints)
        print(f"DEDUPLICATION: {len(internal_constraints)} -> {len(deduplicated_constraints)} constraints (removed {len(internal_constraints) - len(deduplicated_constraints)} duplicates)")
        print(f"Total converted constraints: {len(deduplicated_constraints)} (created {and_groups_created} AND groups)")
        return deduplicated_constraints
    
    def _deduplicate_equivalent_constraints(self, constraints: List) -> List:
        """Remove mathematically equivalent T2 constraints to prevent SVD rank deficiency."""
        from constraint_language_v2 import ConstraintT2
        
        unique_constraints = []
        removed_count = 0
        
        for i, constraint in enumerate(constraints):
            is_duplicate = False
            
            # Only check T2 constraints for mathematical equivalence
            if isinstance(constraint, ConstraintT2):
                for j, existing in enumerate(unique_constraints):
                    if isinstance(existing, ConstraintT2) and self._are_t2_constraints_equivalent(constraint, existing):
                        print(f"DEDUP: Removing duplicate T2 constraint {i}: {constraint} (equivalent to constraint {j})")
                        is_duplicate = True
                        removed_count += 1
                        break
            
            if not is_duplicate:
                unique_constraints.append(constraint)
        
        return unique_constraints
    
    def _are_t2_constraints_equivalent(self, c1: Any, c2: Any) -> bool:
        """Check if two T2 constraints are mathematically equivalent."""
        # Both must be T2 constraints on same variable type
        if c1.v1 != c2.v1 or c1.v2 != c2.v2:
            return False
        
        # Check for direct equivalence: same objects, same operation, same offset
        if (c1.o1 == c2.o1 and c1.o2 == c2.o2 and c1.c == c2.c and 
            abs(c1.offset - c2.offset) < 1e-10):
            return True
        
        # Check for inverse equivalence: "A > B + offset" ≡ "B < A - offset"
        if (c1.o1 == c2.o2 and c1.o2 == c2.o1 and 
            abs(c1.offset + c2.offset) < 1e-10):
            
            # gt/lt are inverse operations
            if (c1.c == 'gt' and c2.c == 'lt') or (c1.c == 'lt' and c2.c == 'gt'):
                return True
            
            # eq/eq with flipped objects and negated offsets
            if c1.c == 'eq' and c2.c == 'eq':
                return True
        
        return False
    
    def _convert_single_constraint(self, constraint: Dict[str, Any]) -> Any:
        """Convert a single constraint to internal format."""
        constraint_type = constraint['type']
        
        if constraint_type == 'left':
            return self._convert_spatial_constraint(constraint, con_left)
        
        elif constraint_type == 'right':
            return self._convert_spatial_constraint(constraint, con_right)
        
        elif constraint_type == 'above':
            return self._convert_spatial_constraint(constraint, con_above)
        
        elif constraint_type == 'below':
            return self._convert_spatial_constraint(constraint, con_below)
        
        elif constraint_type == 'bigger':
            return self._convert_bigger_constraint(constraint)
        
        elif constraint_type == 'smaller':
            return self._convert_smaller_constraint(constraint)
        
        elif constraint_type == 'horizontally_aligned':
            return self._convert_horizontal_alignment(constraint)
        
        elif constraint_type == 'vertically_aligned':
            return self._convert_vertical_alignment(constraint)
        
        elif constraint_type == 'or':
            return self._convert_or_constraint(constraint)
        
        elif constraint_type == 'and':
            return self._convert_and_constraint(constraint)
        
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    def _convert_spatial_constraint(self, constraint: Dict[str, Any], constraint_func) -> Any:
        """Convert spatial constraint (left, right, above, below) with mixed constraint support."""
        obj1_name = constraint['object1']
        obj2_name = constraint['object2']
        constraint_type = constraint['type']
        
        if obj1_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj1_name}' not found in object mapping")
        if obj2_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj2_name}' not found in object mapping")
        
        obj1_idx = self.object_name_to_idx[obj1_name]
        obj2_idx = self.object_name_to_idx[obj2_name]
        base_offset = constraint.get('offset', 0.0)
        
        # MIXED CONSTRAINT HANDLING: Use T2 with enhanced offsets instead of T1 absolute constraints
        # This unifies constraint semantics between inference and evaluation
        
        # Determine constraint scenario
        obj1_is_existing = obj1_name in self.existing_objects
        obj2_is_existing = obj2_name in self.existing_objects
        
        if obj1_is_existing or obj2_is_existing:
            # MIXED or EXISTING-EXISTING constraint - use enhanced offset with dynamic separation
            dynamic_separation = self._calculate_dynamic_separation(obj1_name, obj2_name, constraint_type)
            
            # CRITICAL: For spatial constraints, we need to add separation to prevent overlap
            if constraint_type in ['left', 'right']:
                # For horizontal constraints: obj1 left of obj2 means obj1.x + separation < obj2.x
                # So we need negative offset for 'left' (constraint_func will handle the semantics)
                if constraint_type == 'left':
                    enhanced_offset = base_offset - dynamic_separation  # More negative = more separation
                else:  # right
                    enhanced_offset = base_offset + dynamic_separation  # More positive = more separation
            else:  # above, below
                # For vertical constraints: obj1 above obj2 means obj1.y + separation < obj2.y
                if constraint_type == 'above':
                    enhanced_offset = base_offset - dynamic_separation  # More negative = more separation
                else:  # below
                    enhanced_offset = base_offset + dynamic_separation  # More positive = more separation
            
            print(f"  MIXED {constraint_type}: {obj1_name} vs {obj2_name}, "
                  f"base_offset={base_offset:.3f}, dynamic_sep={dynamic_separation:.3f}, "
                  f"enhanced_offset={enhanced_offset:.3f}")
            
            # Use T2 constraint with enhanced offset (unified approach)
            return constraint_func(obj1_idx, obj2_idx, enhanced_offset)
        else:
            # PURE NEW-TO-NEW constraint - use standard T2 logic with dynamic separation
            dynamic_separation = self._calculate_dynamic_separation(obj1_name, obj2_name, constraint_type)
            
            if constraint_type == 'left':
                enhanced_offset = base_offset - dynamic_separation
            elif constraint_type == 'right':
                enhanced_offset = base_offset + dynamic_separation
            elif constraint_type == 'above':
                enhanced_offset = base_offset - dynamic_separation
            else:  # below
                enhanced_offset = base_offset + dynamic_separation
            
            print(f"  NEW-NEW {constraint_type}: {obj1_name} vs {obj2_name}, "
                  f"base_offset={base_offset:.3f}, dynamic_sep={dynamic_separation:.3f}, "
                  f"enhanced_offset={enhanced_offset:.3f}")
            
            return constraint_func(obj1_idx, obj2_idx, enhanced_offset)
    
    def _convert_bigger_constraint(self, constraint: Dict[str, Any]) -> List[Any]:
        """Convert bigger constraint to wider AND taller."""
        obj1_name = constraint['object1']
        obj2_name = constraint['object2']
        
        if obj1_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj1_name}' not found in object mapping")
        if obj2_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj2_name}' not found in object mapping")
        
        obj1_idx = self.object_name_to_idx[obj1_name]
        obj2_idx = self.object_name_to_idx[obj2_name]
        offset = constraint.get('offset', 0.0)
        
        return [
            con_wider(obj1_idx, obj2_idx, offset),
            con_taller(obj1_idx, obj2_idx, offset)
        ]
    
    def _convert_smaller_constraint(self, constraint: Dict[str, Any]) -> List[Any]:
        """Convert smaller constraint to narrower AND shorter."""
        obj1_name = constraint['object1']
        obj2_name = constraint['object2']
        
        if obj1_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj1_name}' not found in object mapping")
        if obj2_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj2_name}' not found in object mapping")
        
        obj1_idx = self.object_name_to_idx[obj1_name]
        obj2_idx = self.object_name_to_idx[obj2_name]
        offset = constraint.get('offset', 0.0)
        
        return [
            con_narrower(obj1_idx, obj2_idx, offset),
            con_shorter(obj1_idx, obj2_idx, offset)
        ]
    
    def _convert_horizontal_alignment(self, constraint: Dict[str, Any]) -> Any:
        """Convert horizontally_aligned to same Y coordinate."""
        obj1_name = constraint['object1']
        obj2_name = constraint['object2']
        
        if obj1_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj1_name}' not found in object mapping")
        if obj2_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj2_name}' not found in object mapping")
        
        obj1_idx = self.object_name_to_idx[obj1_name]
        obj2_idx = self.object_name_to_idx[obj2_name]
        
        return con_yeq(obj1_idx, obj2_idx)  # Same Y = horizontal alignment
    
    def _convert_vertical_alignment(self, constraint: Dict[str, Any]) -> Any:
        """Convert vertically_aligned to same X coordinate."""
        obj1_name = constraint['object1']
        obj2_name = constraint['object2']
        
        if obj1_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj1_name}' not found in object mapping")
        if obj2_name not in self.object_name_to_idx:
            raise ValueError(f"Object '{obj2_name}' not found in object mapping")
        
        obj1_idx = self.object_name_to_idx[obj1_name]
        obj2_idx = self.object_name_to_idx[obj2_name]
        
        return con_xeq(obj1_idx, obj2_idx)  # Same X = vertical alignment
    
    def _convert_or_constraint(self, constraint: Dict[str, Any]) -> Any:
        """Convert OR constraint."""
        conditions = constraint.get('conditions', [])
        if not conditions:
            raise ValueError("OR constraint has no conditions")
        
        # Convert each condition
        converted_conditions = []
        for condition in conditions:
            if condition['type'] == 'left' and 'value' in condition:
                obj_name = condition['object']
                if obj_name not in self.object_name_to_idx:
                    raise ValueError(f"Object '{obj_name}' not found in object mapping")
                obj_idx = self.object_name_to_idx[obj_name]
                value = condition['value']
                converted_conditions.append(con_left_val(obj_idx, value))
                
            elif condition['type'] == 'right' and 'value' in condition:
                obj_name = condition['object']
                if obj_name not in self.object_name_to_idx:
                    raise ValueError(f"Object '{obj_name}' not found in object mapping")
                obj_idx = self.object_name_to_idx[obj_name]
                value = condition['value']
                converted_conditions.append(con_right_val(obj_idx, value))
            
            else:
                raise ValueError(f"Unsupported OR condition: {condition}")
        
        return ConstraintOR(converted_conditions)
    
    def _convert_and_constraint(self, constraint: Dict[str, Any]) -> Any:
        """Convert AND constraint from JSON format to individual constraint objects."""
        conditions = constraint.get('conditions', [])
        if not conditions:
            raise ValueError("AND constraint has no conditions")
        
        # Convert each condition using the existing conversion logic
        converted_conditions = []
        for condition in conditions:
            converted = self._convert_single_constraint(condition)
            if converted is not None:
                if isinstance(converted, list):
                    converted_conditions.extend(converted)
                else:
                    converted_conditions.append(converted)
        
        if not converted_conditions:
            raise ValueError("AND constraint produced no valid sub-constraints")
        
        # HARDNET FIX: Return list of individual constraints instead of ConstraintAND
        # This eliminates rank deficiency issues in HardNet matrix operations
        return converted_conditions
    
    def convert_constraints_unified(self, 
                                  eval_constraints: List[Dict[str, Any]], 
                                  existing_objects: List[str] = None, 
                                  existing_positions: Dict[str, List[float]] = None, 
                                  new_objects: List[str] = None) -> List[Any]:
        """
        UNIFIED constraint conversion method for both inference and evaluation pipelines.
        
        This method replaces the constraint mismatch between inference (T1 absolute) and 
        evaluation (T2 relative) by using T2 relative constraints with enhanced offsets universally.
        
        Args:
            eval_constraints: List of constraint dictionaries in JSON format
            existing_objects: List of existing object names (from DETR detection)
            existing_positions: Dict mapping existing objects to [x, y, w, h] positions
            new_objects: List of new object names to be generated
            
        Returns:
            List of internal constraint objects (T1, T2, OR, etc.) with unified semantics
        """
        
        # Set up mixed scenario if parameters provided
        if existing_objects is not None and existing_positions is not None and new_objects is not None:
            self.set_mixed_scenario(existing_objects, existing_positions, new_objects)
            print(f"UNIFIED CONVERTER: Mixed scenario with {len(existing_objects)} existing + {len(new_objects)} new objects")
        else:
            # Pure evaluation scenario - all objects are new
            all_object_names = []
            for constraint in eval_constraints:
                for obj_key in ['object1', 'object2', 'object']:
                    if obj_key in constraint and constraint[obj_key] not in all_object_names:
                        all_object_names.append(constraint[obj_key])
            
            self.set_object_mapping(all_object_names)
            self.existing_objects = []
            self.existing_positions = {}
            self.new_objects = all_object_names
            print(f"UNIFIED CONVERTER: Pure evaluation scenario with {len(all_object_names)} new objects")
        
        # Use the enhanced convert_constraints method
        internal_constraints = self.convert_constraints(eval_constraints)
        
        print(f"UNIFIED CONVERTER: Generated {len(internal_constraints)} internal constraints")
        return internal_constraints