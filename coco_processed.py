"""
Take COCO interior scenes with multiple objects
Randomly select 1 to 3 objects to remove  
Use Telea inpainting to create clean backgrounds
Create training pairs inpainted_background, constraints → object_positions
Ground truth = original COCO annotations for removed objects

This creates selfsupervised learning exactly like the SPRING paper
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import shutil

# SPRING interior object categories (from paper)
INTERIOR_CATEGORIES = {
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 66: "mirror",
    67: "dining table", 68: "window", 69: "desk", 70: "toilet", 71: "door",
    72: "tv", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 83: "blender"
}

class SpringCOCOProcessor:    
    def __init__(self, 
                 coco_images_path: str = "/home/gaurang/logicgrad/data/coco/train2017",
                 coco_annotations_path: str = "/home/gaurang/logicgrad/data/coco/annotations/instances_train2017.json",
                 output_path: str = "/home/gaurang/hardnet/data/spring_training_data"):
        
        self.coco_images_path = Path(coco_images_path)
        self.coco_annotations_path = Path(coco_annotations_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "backgrounds").mkdir(exist_ok=True)  # Inpainted backgrounds
        (self.output_path / "annotations").mkdir(exist_ok=True)  # Training annotations
        (self.output_path / "original_scenes").mkdir(exist_ok=True)  # For debugging
        
        self.logger = self._setup_logging()
        
        # Load COCO annotations
        self.logger.info("Loading COCO annotations...")
        with open(self.coco_annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.logger.info(f"Loaded {len(self.coco_data['images'])} images, {len(self.coco_data['annotations'])} annotations")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('SpringCOCOProcessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def process_dataset(self, 
                       target_training_pairs: int = 10000,
                       min_objects_per_scene: int = 3,
                       objects_to_remove_range: Tuple[int, int] = (1, 3)) -> Dict:
        """
        Create training pairs with telea inpainting
        
        Args:
            target_training_pairs: Number of training pairs to create
            min_objects_per_scene: Minimum objects needed in scene
            objects_to_remove_range: Range of objects to remove per scene (min, max)
        """
        
        # Step 1: Find suitable interior scenes
        self.logger.info("Step 1: Finding suitable interior scenes...")
        interior_scenes = self._find_interior_scenes(min_objects_per_scene)
        
        self.logger.info(f"Found {len(interior_scenes)} suitable interior scenes")
        
        # Step 2: Create training pairs with inpainting
        self.logger.info(f"Step 2: Creating {target_training_pairs} training pairs with Telea inpainting...")
        training_pairs = self._create_training_pairs(
            interior_scenes, 
            target_training_pairs,
            objects_to_remove_range
        )
        
        # Step 3: Create dataset splits
        self.logger.info("Step 3: Creating dataset splits...")
        splits = self._create_splits(training_pairs)
        
        # Step 4: Save dataset info
        dataset_info = {
            'total_training_pairs': len(training_pairs),
            'interior_categories': INTERIOR_CATEGORIES,
            'splits': splits,
            'processing_stats': {
                'target_pairs': target_training_pairs,
                'min_objects_per_scene': min_objects_per_scene,
                'objects_to_remove_range': objects_to_remove_range,
                'source_coco_scenes': len(interior_scenes),
                'training_methodology': 'telea_inpainting_with_object_removal'
            }
        }
        
        # Save splits and info
        with open(self.output_path / "splits.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        with open(self.output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f" Processing complete! Created {len(training_pairs)} SPRING training pairs")
        self.logger.info(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        self.logger.info(f"   Methodology: Telea inpainting with {objects_to_remove_range[0]}-{objects_to_remove_range[1]} objects removed per scene")
        
        return dataset_info
    
    def _find_interior_scenes(self, min_objects: int) -> List[Dict]:
        """Find COCO scenes with sufficient interior objects."""
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in INTERIOR_CATEGORIES:
                image_annotations[ann['image_id']].append(ann)
        
        # Filter suitable scenes
        suitable_scenes = []
        for image_info in self.coco_data['images']:
            image_id = image_info['id']
            
            if image_id in image_annotations:
                annotations = image_annotations[image_id]
                
                # Filter valid annotations
                valid_annotations = [
                    ann for ann in annotations 
                    if not ann.get('iscrowd', False) and 
                    ann['area'] > 1500 and  # Filter small objects
                    self._is_reasonable_bbox(ann['bbox'], image_info['width'], image_info['height'])
                ]
                
                if len(valid_annotations) >= min_objects:
                    suitable_scenes.append({
                        'image_info': image_info,
                        'annotations': valid_annotations,
                        'object_count': len(valid_annotations)
                    })
        
        # Sort by object count (prefer scenes with more objects)
        suitable_scenes.sort(key=lambda x: x['object_count'], reverse=True)
        
        return suitable_scenes
    
    def _is_reasonable_bbox(self, bbox: List[float], img_width: int, img_height: int) -> bool:
        """Check if bounding box is reasonable."""
        x, y, w, h = bbox
        
        # Check bounds
        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
            return False
        
        # Check size (not too small, not too large)
        area = w * h
        img_area = img_width * img_height
        area_ratio = area / img_area
        
        if area_ratio < 0.01 or area_ratio > 0.5:  # 1% to 50% of image
            return False
        
        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else float('inf')
        if aspect_ratio < 0.1 or aspect_ratio > 10:  # Reasonable aspect ratios
            return False
        
        return True
    
    def _create_training_pairs(self, 
                              interior_scenes: List[Dict], 
                              target_pairs: int,
                              objects_to_remove_range: Tuple[int, int]) -> List[Dict]:
        """Create training pairs using Telea inpainting."""
        
        training_pairs = []
        min_remove, max_remove = objects_to_remove_range
        
        # Create training pairs by sampling from scenes
        scene_cycle = 0
        
        # Simple progress logging without progress bars
        while len(training_pairs) < target_pairs and scene_cycle < target_pairs * 2:  # Prevent infinite loops
            if len(training_pairs) % 100 == 0:  # Log every 100 pairs
                print(f"Created {len(training_pairs)}/{target_pairs} training pairs")
                
                # Select a random scene
                scene = random.choice(interior_scenes)
                
                try:
                    # Determine how many objects to remove
                    available_objects = len(scene['annotations'])
                    max_removable = min(max_remove, available_objects - 1)  # Keep at least 1 object
                    
                    if max_removable < min_remove:
                        scene_cycle += 1
                        continue
                    
                    n_remove = random.randint(min_remove, max_removable)
                    
                    # Create training pair
                    training_pair = self._create_single_training_pair(scene, n_remove)
                    
                    if training_pair:
                        training_pairs.append(training_pair)
                        # Progress updated via periodic logging above
                
                except Exception as e:
                    self.logger.debug(f"Failed to create training pair from scene {scene['image_info']['id']}: {e}")
                
                scene_cycle += 1
        
        self.logger.info(f"Created {len(training_pairs)} training pairs from {len(interior_scenes)} scenes")
        return training_pairs
    
    def _create_single_training_pair(self, scene: Dict, n_remove: int) -> Optional[Dict]:
        """Create a single training pair with Telea inpainting."""
        
        image_info = scene['image_info']
        annotations = scene['annotations']
        
        # Load original image
        image_path = self.coco_images_path / image_info['file_name']
        if not image_path.exists():
            return None
        
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            return None
        
        # Resize to SPRING standard (512x512)
        original_image = cv2.resize(original_image, (512, 512))
        orig_height, orig_width = original_image.shape[:2]
        
        # Randomly select objects to remove
        objects_to_remove = random.sample(annotations, n_remove)
        remaining_objects = [ann for ann in annotations if ann not in objects_to_remove]
        
        # Create mask for objects to remove
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        removed_objects_info = []
        
        for ann in objects_to_remove:
            # Convert bbox to image coordinates
            bbox = ann['bbox']
            x = int((bbox[0] / image_info['width']) * orig_width)
            y = int((bbox[1] / image_info['height']) * orig_height)
            w = int((bbox[2] / image_info['width']) * orig_width)
            h = int((bbox[3] / image_info['height']) * orig_height)
            
            # Ensure bounds
            x = max(0, min(x, orig_width - 1))
            y = max(0, min(y, orig_height - 1))
            w = max(1, min(w, orig_width - x))
            h = max(1, min(h, orig_height - y))
            
            # Add to mask with some padding for better inpainting
            padding = 5
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(orig_width - x_pad, w + 2 * padding)
            h_pad = min(orig_height - y_pad, h + 2 * padding)
            
            mask[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad] = 255
            
            # Store object info in per-mille coordinates (SPRING standard)
            x_pm = int((x / orig_width) * 1000)
            y_pm = int((y / orig_height) * 1000)
            w_pm = int((w / orig_width) * 1000)
            h_pm = int((h / orig_height) * 1000)
            
            removed_objects_info.append({
                'bbox': [x_pm, y_pm, w_pm, h_pm],
                'category': INTERIOR_CATEGORIES[ann['category_id']],
                'category_id': ann['category_id'],
                'original_annotation': ann
            })
        
        # Apply Telea inpainting
        inpainted_image = cv2.inpaint(original_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Create unique ID for this training pair
        pair_id = f"spring_pair_{image_info['id']:012d}_{random.randint(1000, 9999)}"
        
        # Save inpainted background
        background_path = self.output_path / "backgrounds" / f"{pair_id}.jpg"
        cv2.imwrite(str(background_path), inpainted_image)
        
        # Save original scene for debugging
        original_path = self.output_path / "original_scenes" / f"{pair_id}_original.jpg"
        cv2.imwrite(str(original_path), original_image)
        
        # Create remaining objects info (objects that stay in the background)
        remaining_objects_info = []
        for ann in remaining_objects:
            bbox = ann['bbox']
            x_pm = int((bbox[0] / image_info['width']) * 1000)
            y_pm = int((bbox[1] / image_info['height']) * 1000)
            w_pm = int((bbox[2] / image_info['width']) * 1000)
            h_pm = int((bbox[3] / image_info['height']) * 1000)
            
            remaining_objects_info.append({
                'bbox': [x_pm, y_pm, w_pm, h_pm],
                'category': INTERIOR_CATEGORIES[ann['category_id']],
                'category_id': ann['category_id']
            })
        
        # Create training annotation
        training_annotation = {
            'pair_id': pair_id,
            'original_coco_id': image_info['id'],
            'background_image_path': str(background_path),
            'original_image_path': str(original_path),
            
            # INPUT: What the model sees
            'input': {
                'background_objects': remaining_objects_info,  # Objects still in background
                'constraints': self._generate_constraints(removed_objects_info),  # Spatial constraints
                'scene_context': {
                    'total_objects_in_original': len(annotations),
                    'objects_removed': len(objects_to_remove),
                    'objects_remaining': len(remaining_objects)
                }
            },
            
            # OUTPUT: What the model should predict
            'target': {
                'objects_to_place': removed_objects_info,  # Ground truth positions
                'n_objects': len(removed_objects_info)
            },
            
            # Metadata
            'image_size': [512, 512],
            'coordinate_system': 'per_mille',
            'processing_info': {
                'inpainting_method': 'telea',
                'mask_padding': 5,
                'removed_categories': [obj['category'] for obj in removed_objects_info]
            }
        }
        
        # Save training annotation
        annotation_path = self.output_path / "annotations" / f"{pair_id}.json"
        with open(annotation_path, 'w') as f:
            json.dump(training_annotation, f, indent=2)
        
        return {
            'pair_id': pair_id,
            'annotation_path': str(annotation_path),
            'background_path': str(background_path),
            'n_objects_to_place': len(removed_objects_info),
            'categories': [obj['category'] for obj in removed_objects_info]
        }
    
    def _generate_constraints(self, removed_objects: List[Dict]) -> List[Dict]:
        """
        REVOLUTIONARY: Generate layout-aware spatial constraints from GROUND TRUTH positions.
        
        This creates constraints that are SATISFIABLE by the target layout, eliminating
        the fundamental data corruption issue that was breaking training.
        """
        
        if not removed_objects:
            return []
        
        # Convert to tensor format for constraint generation
        import torch
        layout_tensor = torch.zeros((len(removed_objects), 4))
        categories = []
        
        for i, obj in enumerate(removed_objects):
            bbox = obj['bbox']  # In per-mille format (0-1000)
            # Normalize to 0-1 range for constraint generator
            x_norm = bbox[0] / 1000.0
            y_norm = bbox[1] / 1000.0
            w_norm = bbox[2] / 1000.0
            h_norm = bbox[3] / 1000.0
            layout_tensor[i] = torch.tensor([x_norm, y_norm, w_norm, h_norm], dtype=torch.float32)
            categories.append(obj['category'])
        
        # Import our layout-aware constraint generator
        try:
            from constraint_gen import ConstraintGenerator, ConstraintGenerationConfig, ConstraintDifficulty
            
            # Create constraint generator with SPRING configuration
            # Note: ConstraintGenerationConfig uses normalized coordinates (0-1)
            # but our layout_tensor is in per-mille (0-1000), so we need to normalize
            config = ConstraintGenerationConfig(
                canvas_width=1.0,  # Normalized coordinate system (0-1)
                canvas_height=1.0,
                min_object_size=0.03,  # 3% minimum size
                max_object_size=0.4,   # 40% maximum size
                constraints_per_scene=(1, 3),  # Generate 1-3 constraints
                constraint_difficulty=ConstraintDifficulty.BEGINNER
            )
            
            constraint_gen = ConstraintGenerator(config)
            
            # Generate layout-aware constraints from ACTUAL target positions
            constraint_sets = constraint_gen.generate_constraints_for_layout(
                layout=layout_tensor,
                n_objects=len(removed_objects),
                categories=categories
            )
            
            # Convert to the dictionary format expected by the training pipeline
            formatted_constraints = []
            
            for constraint in constraint_sets:
                if hasattr(constraint, 'c') and hasattr(constraint, 'o1'):
                    # Handle different constraint types
                    constraint_dict = {
                        'type': 'spatial_relation',
                        'constraint_object': constraint,
                        'description': f"Layout-aware constraint generated from ground truth positions"
                    }
                    formatted_constraints.append(constraint_dict)
            
            # Always add boundary constraints for safety
            for i, obj in enumerate(removed_objects):
                formatted_constraints.append({
                    'type': 'boundary',
                    'object_index': i,
                    'constraint': f"{obj['category']} should be within canvas bounds",
                    'parameters': {
                        'min_x': 50, 'max_x': 950,  # Per-mille coordinates
                        'min_y': 50, 'max_y': 950,
                        'min_w': 30, 'max_w': 300,
                        'min_h': 30, 'max_h': 300
                    }
                })
            
            self.logger.debug(f"Generated {len(formatted_constraints)} layout-aware constraints from {len(removed_objects)} ground truth objects")
            return formatted_constraints
            
        except ImportError:
            # Fallback to simple constraints if constraint generator not available
            self.logger.warning("Layout-aware constraint generator not available, falling back to basic constraints")
            
            constraints = []
            
            for i, obj in enumerate(removed_objects):
                constraints.append({
                    'type': 'boundary',
                    'object_index': i,
                    'constraint': f"{obj['category']} should be within canvas bounds",
                    'parameters': {
                        'min_x': 50, 'max_x': 950,
                        'min_y': 50, 'max_y': 950,
                        'min_w': 30, 'max_w': 300,
                        'min_h': 30, 'max_h': 300
                    }
                })
            
            # Add basic non-overlapping constraints
            if len(removed_objects) > 1:
                for i in range(len(removed_objects)):
                    for j in range(i + 1, len(removed_objects)):
                        constraints.append({
                            'type': 'non_overlapping',
                            'object_indices': [i, j],
                            'constraint': f"{removed_objects[i]['category']} and {removed_objects[j]['category']} should not overlap",
                            'parameters': {'min_distance': 20}
                        })
            
            return constraints
    
    def _create_splits(self, training_pairs: List[Dict]) -> Dict[str, List[str]]:
        """Create train/val/test splits."""
        
        pair_ids = [pair['pair_id'] for pair in training_pairs]
        random.shuffle(pair_ids)
        
        total = len(pair_ids)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        splits = {
            'train': pair_ids[:train_end],
            'val': pair_ids[train_end:val_end],
            'test': pair_ids[val_end:]
        }
        
        return splits


def main():
    """Main processing function."""
    print("=== SPRING-STYLE COCO PROCESSOR WITH TELEA INPAINTING ===\n")
    
    processor = SpringCOCOProcessor()
    
    # Process dataset with SPRING methodology
    dataset_info = processor.process_dataset(
        target_training_pairs=10000,  # Create 10k training pairs
        min_objects_per_scene=3,      # Need at least 3 objects (remove 1-2, keep 1+)
        objects_to_remove_range=(1, 3)  # Remove 1-3 objects per scene
    )
    
    print(f"\n SPRING training dataset created!")
    print(f" Total training pairs: {dataset_info['total_training_pairs']}")
    print(f" Output directory: {processor.output_path}")
    print(f" Training methodology: Telea inpainting with object removal")
    print(f" Ready for SPRING two-stage training!")
    
    return dataset_info


if __name__ == "__main__":
    main()