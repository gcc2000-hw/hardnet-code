#!/usr/bin/env python3
"""
SCENE PROMPT GENERATOR
Generate text descriptions for all scenes in the evaluation dataset.
Maps each scene ID to its corresponding text prompt based on specifications.
Optimized for fast processing of 10K+ specifications.
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

def create_text_description(specification: Dict[str, Any]) -> str:
    """
    Create text description from specification for CLIP scoring.
    Same logic as proper_evaluation_metrics.py:343-375
    """
    try:
        room_type = specification.get('room_type', 'room')
        objects = specification.get('objects', [])
        constraints = specification.get('constraints', [])
        
        # Start with room type and objects
        description = f"A {room_type} scene with "
        description += ", ".join(objects)
        
        # Add constraint descriptions for key spatial relationships
        constraint_descriptions = []
        for constraint in constraints:
            if isinstance(constraint, dict):
                if constraint.get('type') == 'left':
                    constraint_descriptions.append(f"{constraint['object1']} left of {constraint['object2']}")
                elif constraint.get('type') == 'right':
                    constraint_descriptions.append(f"{constraint['object1']} right of {constraint['object2']}")
                elif constraint.get('type') == 'above':
                    constraint_descriptions.append(f"{constraint['object1']} above {constraint['object2']}")
                elif constraint.get('type') == 'below':
                    constraint_descriptions.append(f"{constraint['object1']} below {constraint['object2']}")
        
        if constraint_descriptions:
            description += " with " + ", ".join(constraint_descriptions[:3])  # Limit to 3 constraints
        
        return description
    except Exception as e:
        # Fallback to simple description
        room_type = specification.get('room_type', 'room')
        objects = specification.get('objects', [])
        return f"A {room_type} scene with " + ", ".join(objects)

def process_specifications_batch(specifications: List[Dict[str, Any]], batch_size: int = 1000) -> Dict[str, str]:
    """Process specifications in batches for memory efficiency."""
    scene_prompts = {}
    total_specs = len(specifications)
    
    print(f"Processing {total_specs} specifications in batches of {batch_size}...")
    
    for i in range(0, total_specs, batch_size):
        batch_end = min(i + batch_size, total_specs)
        batch = specifications[i:batch_end]
        
        # Process batch
        batch_start_time = time.time()
        for spec in batch:
            scene_id = spec['id']
            prompt = create_text_description(spec)
            scene_prompts[scene_id] = prompt
        
        batch_time = time.time() - batch_start_time
        print(f"  Batch {i//batch_size + 1}: Processed {len(batch)} specs in {batch_time:.2f}s ({len(batch)/batch_time:.0f} specs/sec)")
    
    return scene_prompts

def verify_background_coverage(scene_prompts: Dict[str, str], backgrounds_dir: str) -> Dict[str, Any]:
    """Verify that all background images have corresponding prompts."""
    backgrounds_path = Path(backgrounds_dir)
    
    if not backgrounds_path.exists():
        print(f"WARNING: Backgrounds directory not found: {backgrounds_dir}")
        return {
            'background_files': 0,
            'matched_prompts': 0,
            'missing_prompts': [],
            'extra_prompts': []
        }
    
    # Get all background image files
    background_files = list(backgrounds_path.glob("*.png"))
    background_ids = {f.stem for f in background_files}  # Remove .png extension
    
    prompt_ids = set(scene_prompts.keys())
    
    missing_prompts = background_ids - prompt_ids
    extra_prompts = prompt_ids - background_ids
    
    coverage_stats = {
        'background_files': len(background_files),
        'prompt_count': len(scene_prompts),
        'matched_prompts': len(background_ids & prompt_ids),
        'missing_prompts': sorted(list(missing_prompts)),
        'extra_prompts': sorted(list(extra_prompts))
    }
    
    return coverage_stats

def main():
    """Main entry point."""
    print("=" * 60)
    print("SCENE PROMPT GENERATOR")
    print("=" * 60)
    
    # Set device for potential GPU operations (though this is mostly CPU work)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    specs_path = "/home/gaurang/hardnetnew/data/evaluation/specifications_realistic_cleaned_FIXED.json"
    backgrounds_dir = "/home/gaurang/hardnetnew/data/evaluation/backgrounds"
    output_path = "/home/gaurang/hardnetnew/scene_prompts_mapping.json"
    
    start_time = time.time()
    
    try:
        # Load specifications
        print(f"\nLoading specifications from: {specs_path}")
        load_start = time.time()
        
        with open(specs_path, 'r') as f:
            specifications = json.load(f)
        
        load_time = time.time() - load_start
        print(f"Loaded {len(specifications)} specifications in {load_time:.2f}s")
        
        # Process specifications to generate prompts
        print(f"\nGenerating text descriptions...")
        process_start = time.time()
        
        scene_prompts = process_specifications_batch(specifications, batch_size=1000)
        
        process_time = time.time() - process_start
        print(f"Generated {len(scene_prompts)} prompts in {process_time:.2f}s ({len(scene_prompts)/process_time:.0f} prompts/sec)")
        
        # Verify coverage against background images
        print(f"\nVerifying coverage against background images...")
        coverage_stats = verify_background_coverage(scene_prompts, backgrounds_dir)
        
        print(f"Coverage Statistics:")
        print(f"  Background files: {coverage_stats['background_files']}")
        print(f"  Generated prompts: {coverage_stats['prompt_count']}")
        print(f"  Matched prompts: {coverage_stats['matched_prompts']}")
        print(f"  Missing prompts: {len(coverage_stats['missing_prompts'])}")
        print(f"  Extra prompts: {len(coverage_stats['extra_prompts'])}")
        
        if coverage_stats['missing_prompts']:
            print(f"  First few missing: {coverage_stats['missing_prompts'][:5]}")
        
        if coverage_stats['extra_prompts']:
            print(f"  First few extra: {coverage_stats['extra_prompts'][:5]}")
        
        # Save results
        print(f"\nSaving scene-prompt mapping...")
        save_start = time.time()
        
        output_data = {
            'metadata': {
                'total_scenes': len(scene_prompts),
                'generation_time': process_time,
                'timestamp': time.time(),
                'coverage_stats': coverage_stats
            },
            'scene_prompts': scene_prompts
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        save_time = time.time() - save_start
        print(f"Saved mapping to: {output_path} ({save_time:.2f}s)")
        
        # Show sample prompts
        print(f"\nSample prompts:")
        sample_ids = list(scene_prompts.keys())[:5]
        for i, scene_id in enumerate(sample_ids, 1):
            print(f"  {i}. {scene_id}: {scene_prompts[scene_id]}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETED in {total_time:.2f}s")
        print(f"Generated {len(scene_prompts)} scene-prompt mappings")
        print(f"Average: {len(scene_prompts)/total_time:.0f} prompts/sec")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()