#!/usr/bin/env python3
"""
FAST COCO Processor - Optimized for speed during training data generation

Key optimizations:
1. Disable mathematical constraint validation (major bottleneck)
2. Minimal constraint generation (1 constraint per pair max)
3. Batch processing optimizations
4. Progress tracking every 50 pairs

This creates training data quickly while maintaining correctness.
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
import torch

# SPRING interior object categories (from paper)
INTERIOR_CATEGORIES = {
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 66: "mirror",
    67: "dining table", 68: "window", 69: "desk", 70: "toilet", 71: "door",
    72: "tv", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 83: "blender"
}

class FastCOCOProcessor:
    """FAST: SPRING-style COCO processor optimized for speed."""
    
    def __init__(self, 
                 coco_images_path: str = "/home/gaurang/logicgrad/data/coco/train2017",
                 coco_annotations_path: str = "/home/gaurang/logicgrad/data/coco/annotations/instances_train2017.json",
                 output_path: str = "/home/gaurang/hardnet/data/spring_training_data"):
        
        self.coco_images_path = Path(coco_images_path)
        self.coco_annotations_path = Path(coco_annotations_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "backgrounds").mkdir(exist_ok=True)
        (self.output_path / "annotations").mkdir(exist_ok=True)
        (self.output_path / "original_scenes").mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Load COCO annotations
        self.logger.info("Loading COCO annotations...")
        with open(self.coco_annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.logger.info(f"Loaded {len(self.coco_data['images'])} images, {len(self.coco_data['annotations'])} annotations")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('FastCOCOProcessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - FAST - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
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
    
    def _generate_fast_constraints(self, removed_objects: List[Dict]) -> List[Dict]:
        """
        FAST: Generate minimal constraints for speed.
        
        Only generates 1 simple constraint per pair to avoid bottleneck.
        """
        
        if not removed_objects or len(removed_objects) < 2:
            # Single object - just boundary constraint
            if removed_objects:
                return [{
                    'type': 'boundary',
                    'object_index': 0,
                    'constraint': f"{removed_objects[0]['category']} should be within canvas bounds",
                    'parameters': {
                        'min_x': 50, 'max_x': 950,
                        'min_y': 50, 'max_y': 950,
                        'min_w': 30, 'max_w': 300,
                        'min_h': 30, 'max_h': 300
                    }
                }]
            return []
        
        # Multiple objects - create 1 simple spatial constraint
        obj1 = removed_objects[0]
        obj2 = removed_objects[1]
        
        # Simple left/right constraint based on X positions
        x1 = obj1['bbox'][0]  # X position in per-mille
        x2 = obj2['bbox'][0]
        
        # Import constraint language for single constraint
        from constraint_language_v2 import con_left, con_right
        
        if x1 < x2:
            # obj1 is left of obj2
            constraint = con_left(0, 1, 50)  # 50 per-mille offset
            description = f"{obj1['category']} is left of {obj2['category']}"
        else:
            # obj2 is left of obj1
            constraint = con_right(0, 1, 50)  # 50 per-mille offset
            description = f"{obj1['category']} is right of {obj2['category']}"
        
        # Return single constraint + boundary constraints
        constraints = [
            {
                'type': 'spatial_relation',
                'constraint_object': constraint,
                'description': description
            }
        ]
        
        # Add boundary constraints
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
        
        return constraints
    
    def _create_single_training_pair(self, scene: Dict, n_remove: int) -> Optional[Dict]:
        """Create a single training pair with fast processing."""
        
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
        
        # FAST: Generate minimal constraints
        constraints = self._generate_fast_constraints(removed_objects_info)
        
        # Create training annotation
        training_annotation = {
            'pair_id': pair_id,
            'original_coco_id': image_info['id'],
            'background_image_path': str(background_path),
            'original_image_path': str(original_path),
            
            # INPUT: What the model sees
            'input': {
                'background_objects': remaining_objects_info,
                'constraints': constraints,
                'scene_context': {
                    'total_objects_in_original': len(annotations),
                    'objects_removed': len(objects_to_remove),
                    'objects_remaining': len(remaining_objects)
                }
            },
            
            # OUTPUT: What the model should predict
            'target': {
                'objects_to_place': removed_objects_info,
                'n_objects': len(removed_objects_info)
            },
            
            'image_size': [512, 512],
            'coordinate_system': 'per_mille',
            'processing_info': {
                'inpainting_method': 'telea',
                'mask_padding': 5,
                'removed_categories': [obj['category'] for obj in removed_objects_info],
                'generation_method': 'fast_minimal_constraints'
            }
        }
        
        # Save annotation
        annotation_path = self.output_path / "annotations" / f"{pair_id}.json"
        with open(annotation_path, 'w') as f:
            json.dump(training_annotation, f, indent=2)
        
        return training_annotation
    
    def process_dataset(self, 
                       target_training_pairs: int = 10000,
                       min_objects_per_scene: int = 3,
                       objects_to_remove_range: Tuple[int, int] = (1, 3)) -> Dict:
        """
        Create SPRING-style training pairs FAST.
        """
        
        # Step 1: Find suitable interior scenes
        self.logger.info("Step 1: Finding suitable interior scenes...")
        interior_scenes = self._find_interior_scenes(min_objects_per_scene)
        
        self.logger.info(f"Found {len(interior_scenes)} suitable interior scenes")
        
        # Step 2: Create training pairs with fast processing
        self.logger.info(f"Step 2: Creating {target_training_pairs} training pairs FAST...")
        training_pairs = []
        min_remove, max_remove = objects_to_remove_range
        
        scene_cycle = 0
        
        # Fast progress logging
        while len(training_pairs) < target_training_pairs and scene_cycle < target_training_pairs * 2:
            if len(training_pairs) % 50 == 0:  # Log every 50 pairs (faster updates)
                self.logger.info(f"FAST: Created {len(training_pairs)}/{target_training_pairs} training pairs")
            
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
            
            except Exception as e:
                self.logger.debug(f"Failed to create training pair from scene {scene['image_info']['id']}: {e}")
            
            scene_cycle += 1
        
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
                'training_methodology': 'fast_telea_inpainting_minimal_constraints'
            }
        }
        
        # Save splits and info
        with open(self.output_path / "splits.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        with open(self.output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f" FAST Processing complete! Created {len(training_pairs)} SPRING training pairs")
        self.logger.info(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        self.logger.info(f"   Methodology: FAST Telea inpainting with minimal constraint generation")
        
        return dataset_info
    
    def _create_splits(self, training_pairs: List[Dict]) -> Dict:
        """Create train/val/test splits."""
        random.shuffle(training_pairs)
        
        total = len(training_pairs)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        splits = {
            'train': [pair['pair_id'] for pair in training_pairs[:train_size]],
            'val': [pair['pair_id'] for pair in training_pairs[train_size:train_size + val_size]],
            'test': [pair['pair_id'] for pair in training_pairs[train_size + val_size:]]
        }
        
        return splits

def main():
    """Main processing function."""
    print("=== FAST SPRING-STYLE COCO PROCESSOR ===\n")
    
    processor = FastCOCOProcessor()
    
    # Process dataset with FAST methodology
    dataset_info = processor.process_dataset(
        target_training_pairs=10000,  # Full 10k dataset
        min_objects_per_scene=3,
        objects_to_remove_range=(1, 3)
    )
    
    print(f"\n FAST SPRING training dataset created!")
    print(f" Total training pairs: {dataset_info['total_training_pairs']}")
    print(f" Output directory: {processor.output_path}")
    print(f" Training methodology: FAST Telea inpainting with minimal constraints")
    print(f"⚡ Optimized for speed - ready for SPRING two-stage training!")
    
    return dataset_info

if __name__ == "__main__":
    main()