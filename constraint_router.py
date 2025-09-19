
#Constraint Router whcih will separate affine and non affine constraints
import logging
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
import numpy as np

# constraint types from the language file
from constraint_language_v2 import (
    ConstraintT1, ConstraintT2, ConstraintT3, ConstraintT4, ConstraintT5, ConstraintT6,
    ConstraintOR, ConstraintAND, ConstraintNOT
)

class ConstraintRouter:
    # T1 to T4 and TAND can be made affine 
    # distance constraints maybe but depends, 6, Or and Not are always non affine
    """
    Affine constraints:
    ConstraintT1- single object + value constraints
    ConstraintT2- two object comparisons  
    ConstraintT3- arithmetic bounds (x + width ≤ val)
    ConstraintT4- complex arithmetic between objects
    ConstraintAND- conjunction of affine constraints
    
    Non affine constraints 
    ConstraintT5- distance constraints (Euclidean distance)
    ConstraintT6- complex multi-object constraints
    ConstraintOR- logical disjunction (creates nonconvex regions)
    ConstraintNOT- logical negation (creates nonconvex regions)
    """
    
    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging(enable_logging)
        self.classification_stats = defaultdict(int)
        self.routing_history = []
        
    def _setup_logging(self, enable: bool) -> logging.Logger:
        """Setup logging for constraint routing analysis."""
        logger = logging.getLogger('ConstraintRouter')
        if enable and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def split_constraints(self, constraints: List[Any]) -> Tuple[List[Any], List[Any]]:
        # First expand any composite constraints
        expanded_constraints = self._expand_composite_constraints(constraints)
        
        affine_constraints = []
        non_affine_constraints = []
        
        for constraint in expanded_constraints:
            constraint_type = type(constraint).__name__
            is_affine = self._is_affine_constraint(constraint)
            
            # DEBUG: Log constraint routing decisions
            if hasattr(self, 'enable_logging') and self.enable_logging:
                self.logger.info(f" ROUTING DEBUG: {constraint_type} -> {'AFFINE' if is_affine else 'NON-AFFINE'}")
                if hasattr(constraint, 'c'):
                    self.logger.info(f"   Constraint details: c='{constraint.c}'")
            
            if is_affine:
                affine_constraints.append(constraint)
                self.classification_stats['affine'] += 1
            else:
                non_affine_constraints.append(constraint)
                self.classification_stats['non_affine'] += 1
                
        # Logging
        routing_info = {
            'total_constraints': len(expanded_constraints),
            'affine_count': len(affine_constraints),
            'non_affine_count': len(non_affine_constraints),
            'affine_percentage': len(affine_constraints) / max(len(expanded_constraints), 1) * 100
        }
        self.routing_history.append(routing_info)
        
        self.logger.debug(
            f"Routed {routing_info['affine_count']} affine, "
            f"{routing_info['non_affine_count']} non-affine "
            f"({routing_info['affine_percentage']:.1f}% affine)"
        )
        
        return affine_constraints, non_affine_constraints
    
    def route_constraints(self, constraints: List[Any]):
        """Route constraints - wrapper for split_constraints for API compatibility."""
        from collections import namedtuple
        RoutingResult = namedtuple('RoutingResult', ['affine_constraints', 'non_affine_constraints'])
        affine, non_affine = self.split_constraints(constraints)
        return RoutingResult(affine, non_affine)
    #Expand composite constraints that return lists (like cons_atop amd cons_disjoint).
    def _expand_composite_constraints(self, constraints: List[Any]) -> List[Any]:
        expanded = []
        for constraint in constraints:
            if isinstance(constraint, list):
                # Recursive expansion for nested lists
                expanded.extend(self._expand_composite_constraints(constraint))
            else:
                expanded.append(constraint)
        return expanded
    #Determine if a constraint is affine or not
    def _is_affine_constraint(self, constraint: Any) -> bool:
        # Direct affine constraint types
        if isinstance(constraint, (ConstraintT1, ConstraintT2, ConstraintT3, ConstraintT4)):
            return True
            
        # CRITICAL FIX: Extended affine constraint types
        # BoundaryConstraint and SizeRatioConstraint are mathematically affine (linear inequalities)
        # They must be routed to HardNet for 100% satisfaction guarantee
        try:
            # Check if extended constraints are available
            from constraint_language_v2_extended import BoundaryConstraint, SizeRatioConstraint, NonOverlapConstraint, RelativeToExistingConstraint
            EXTENDED_CONSTRAINTS_AVAILABLE = True
        except ImportError:
            EXTENDED_CONSTRAINTS_AVAILABLE = False
        
        if EXTENDED_CONSTRAINTS_AVAILABLE:
            # Extended affine constraint types that must go to HardNet
            if isinstance(constraint, (BoundaryConstraint, SizeRatioConstraint)):
                return True
            
            # Extended non-affine constraint types  
            if isinstance(constraint, (NonOverlapConstraint, RelativeToExistingConstraint)):
                return False
            
        # ConstraintAND is affine if all sub-constraints are affine
        if isinstance(constraint, ConstraintAND):
            return self._all_subconstraints_affine(constraint.c)
            
        # Non-affine constraint types
        if isinstance(constraint, (ConstraintT5, ConstraintT6, ConstraintOR, ConstraintNOT)):
            return False
            
        # Unknown constraint type - log warning and default to non-affine for safety
        constraint_type_name = type(constraint).__name__
        self.logger.warning(f"Unknown constraint type: {constraint_type_name}. Defaulting to non-affine.")
        return False
    #Check if all subconstraints of a composite constraint are affin
    def _all_subconstraints_affine(self, subconstraints: List[Any]) -> bool:

        expanded_subs = self._expand_composite_constraints(subconstraints)
        return all(self._is_affine_constraint(sub) for sub in expanded_subs)
    #Get the overall split percentages across all routing operations.
    def get_split_percentages(self) -> Dict[str, float]:
        total = self.classification_stats['affine'] + self.classification_stats['non_affine']
        if total == 0:
            return {'affine': 0.0, 'non_affine': 0.0}
            
        return {
            'affine': (self.classification_stats['affine'] / total) * 100,
            'non_affine': (self.classification_stats['non_affine'] / total) * 100
        }
    #Routing history
    def get_routing_history(self) -> List[Dict[str, Any]]:
        return self.routing_history.copy()
    #Reset statistics and history
    def reset_stats(self):
        self.classification_stats.clear()
        self.routing_history.clear()
        self.logger.info("Constraint router statistics reset")
    
    def create_constraint_analysis_report(self) -> str:

        percentages = self.get_split_percentages()
        total_processed = sum(self.classification_stats.values())
        
        report = f"""
Total Constraints Processed {total_processed}
Affine Constraints {self.classification_stats['affine']} ({percentages['affine']:.1f}%)
Non Affine Constraints {self.classification_stats['non_affine']} ({percentages['non_affine']:.1f}%)
Routing History ({len(self.routing_history)} operations):
"""
        
        for i, routing in enumerate(self.routing_history):
            report += f"  Operation {i+1}: {routing['affine_count']} affine, {routing['non_affine_count']} non-affine ({routing['affine_percentage']:.1f}% affine)\n"
    
    def visualize_constraint_distribution(self, save_path: str = None):
        percentages = self.get_split_percentages()
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart of overall distribution
        labels = ['Affine', 'Non-Affine']
        sizes = [percentages['affine'], percentages['non_affine']]
        colors = ['#2E8B57', '#CD5C5C']  # Sea green, Indian red
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Constraint Distribution')
        
        # Bar chart of routing history
        if self.routing_history:
            operations = range(1, len(self.routing_history) + 1)
            affine_counts = [r['affine_count'] for r in self.routing_history]
            non_affine_counts = [r['non_affine_count'] for r in self.routing_history]
            
            width = 0.35
            ax2.bar([x - width/2 for x in operations], affine_counts, width, label='Affine', color='#2E8B57')
            ax2.bar([x + width/2 for x in operations], non_affine_counts, width, label='Non-Affine', color='#CD5C5C')
            
            ax2.set_xlabel('Routing Operation')
            ax2.set_ylabel('Number of Constraints')
            ax2.set_title('Constraint Counts by Operation')
            ax2.legend()
            ax2.set_xticks(operations)
        else:
            ax2.text(0.5, 0.5, 'No routing history available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Routing History (No Data)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Constraint distribution plot saved to {save_path}")
        
        return fig


def create_sample_constraint_set() -> List[Any]:
    from constraint_language_v2 import (
        con_left, con_above, con_left_val, con_wider, 
        right_bound, con_leftleft, cons_atop, cons_disjoint,
        con_mdisteq, ConstraintOR, ConstraintAND
    )
    
    # Sample constraint set with known distribution
    constraints = [
        # Affine constraints (T1-T4)
        con_left_val(0, 100),           # T1: o0.x <= 100
        con_above(0, 1, 50),            # T2: o0.y <= o1.y - 50
        right_bound(0, 800),            # T3: o0.x + o0.width <= 800
        con_leftleft(0, 1, 20),         # T4: o0.x + o0.width <= o1.x - 20
        
        # Composite affine constraints
        cons_atop(0, 1, 10, 5),         # Multiple T2/T4 constraints
        
        # Non-affine constraints
        con_mdisteq(0, 1, 2, 100),      # T5: Distance constraint
        cons_disjoint(0, 1),            # OR constraints
        
        # AND constraint (affine if all subs are affine)
        ConstraintAND([con_wider(0, 1, 0), con_left_val(0, 200)]),
        
        # OR constraint (always non-affine)
        ConstraintOR([con_left(0, 1, 0), con_above(0, 1, 0)])
    ]
    
    return constraints


if __name__ == "__main__":
    # Demonstration and validation
    
    # Create router
    router = ConstraintRouter(enable_logging=True)
    
    # Test with sample constraints
    sample_constraints = create_sample_constraint_set()
    print(f"Testing with {len(sample_constraints)} sample constraints...")
    
    # Route constraints
    affine, non_affine = router.split_constraints(sample_constraints)
    
    # Display results
    print(f"\nRouting Results:")
    print(f"Affine constraints: {len(affine)}")
    print(f"Non-affine constraints: {len(non_affine)}")
    
    # Show detailed analysis
    print("\n" + router.create_constraint_analysis_report())
    
    # Get split percentages for validation
    percentages = router.get_split_percentages()
    print(f"\nValidation Results:")
    print(f"Affine percentage: {percentages['affine']:.1f}%")
    print(f"Target: 75-90% (Current: {'PASS' if 75 <= percentages['affine'] <= 90 else 'FAIL'})")
    
    # Create visualization
    try:
        fig = router.visualize_constraint_distribution()
        # plt.show()  # Uncomment to display plot
    except Exception as e:
        print(f"Visualization error: {e}")
    
