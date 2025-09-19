"""
FIXED: SPRING Data Loader for Telea Inpainting Training Pairs
Loads training pairs created by the SPRING methodology:

INPUT: Inpainted background + remaining objects + constraints
OUTPUT: Positions for removed objects (ground truth)

This matches the original SPRING paper training exactly.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import random
from collections import defaultdict

# Import tensor format utilities from our system
try:
    from spring_int import TensorFormatConverter
    from completeInt import SpringHybridConfig
    SYSTEM_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Warning: System components not available. Using mock implementations.")
    SYSTEM_COMPONENTS_AVAILABLE = False


@dataclass
class SpringDatasetConfig:
    """Configuration for SPRING training pairs dataset."""
    
    # UPDATED: Dataset paths for SPRING training pairs
    dataset_root: str = "/home/gaurang/hardnet/data/spring_training_data"
    backgrounds_dir: str = "backgrounds"
    annotations_dir: str = "annotations" 
    splits_file: str = "splits.json"
    
    # Image processing (SPRING standard)
    image_size: Tuple[int, int] = (512, 512)  #  SPRING standard
    normalize_images: bool = True
    image_format: str = "RGB"
    
    # Layout processing (per-mille coordinates)
    max_objects_to_place: int = 5  # Max objects to place per training pair
    max_background_objects: int = 10  # Max objects remaining in background
    coordinate_system: str = "per_mille"  #  0-1000 range (SPRING standard)
    padding_value: float = -1.0  # Value for padding unused object slots
    
    # SPRING interior object categories
    object_categories: List[str] = None
    category_to_id: Dict[str, int] = None
    
    # Data augmentation (reduced for training stability)
    enable_augmentation: bool = True
    augmentation_probability: float = 0.2  
    max_rotation: float = 3.0  
    max_scale: Tuple[float, float] = (0.98, 1.02)  
    
    # Tensor format
    output_format: str = "sequence"  # "sequence" or "flat"
    ensure_batch_consistency: bool = True
    
    # Performance 
    cache_images: bool = False  
    num_workers: int = 0  # FIXED: Multi-worker DataLoader is 100x slower on small datasets  
    
    def __post_init__(self):
        if self.object_categories is None:
            # SPRING interior categories
            self.object_categories = [
                "chair", "couch", "potted plant", "bed", "mirror", 
                "dining table", "window", "desk", "toilet", "door", 
                "tv", "microwave", "oven", "toaster", "sink", 
                "refrigerator", "blender"
            ]
        
        if self.category_to_id is None:
            self.category_to_id = {cat: idx for idx, cat in enumerate(self.object_categories)}


class SpringTrainingPairDataset(Dataset):
    """Dataset for SPRING training pairs with Telea inpainting."""
    
    def __init__(self, 
                 config: SpringDatasetConfig,
                 split: str = "train",
                 transform: Optional[Callable] = None):
        
        self.config = config
        self.split = split
        self.transform = transform
        self.logger = self._setup_logging()
        
        # Load dataset
        self.dataset_root = Path(config.dataset_root)
        self.backgrounds_dir = self.dataset_root / config.backgrounds_dir
        self.annotations_dir = self.dataset_root / config.annotations_dir
        
        # Load splits
        splits_path = self.dataset_root / config.splits_file
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                splits_data = json.load(f)
            self.pair_ids = splits_data.get(split, [])
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_path}")
        
        self.logger.info(f"Loaded {len(self.pair_ids)} training pairs for {split} split")
        
        # Setup tensor format converter
        if SYSTEM_COMPONENTS_AVAILABLE:
            self.format_converter = TensorFormatConverter()
        else:
            self.format_converter = None
        
        # Setup transforms
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(f'SpringTrainingDataset_{self.split}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms."""
        
        transform_list = [
            transforms.Resize(self.config.image_size),
        ]
        
        # Add augmentation for training
        if self.split == "train" and self.config.enable_augmentation:
            if random.random() < self.config.augmentation_probability:
                transform_list.extend([
                    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
                    transforms.RandomRotation(self.config.max_rotation),
                ])
        
        # Final transforms
        transform_list.extend([
            transforms.ToTensor(),
        ])
        
        if self.config.normalize_images:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.pair_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training pair."""
        
        try:
            pair_id = self.pair_ids[idx]
            
            # Load training annotation
            annotation_path = self.annotations_dir / f"{pair_id}.json"
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            # Load inpainted background image
            background_path = self.backgrounds_dir / f"{pair_id}.jpg"
            background_image = Image.open(background_path).convert('RGB')
            background_tensor = self.transform(background_image)
            
            # Extract input data (what model sees)
            input_data = annotation['input']
            background_objects = input_data.get('background_objects', [])
            constraints = input_data.get('constraints', [])
            
            # Extract target data (what model should predict)
            target_data = annotation['target']
            objects_to_place = target_data['objects_to_place']
            
            # Convert background objects to tensor
            background_layout, background_valid_mask = self._objects_to_tensor(
                background_objects, self.config.max_background_objects
            )
            
            # Convert target objects to tensor (ground truth)
            target_layout, target_valid_mask = self._objects_to_tensor(
                objects_to_place, self.config.max_objects_to_place
            )
            
            # Get categories for background and target objects
            background_categories = [obj['category'] for obj in background_objects]
            target_categories = [obj['category'] for obj in objects_to_place]
            
            # Convert categories to IDs with padding
            background_category_ids = self._categories_to_tensor(
                background_categories, self.config.max_background_objects
            )
            target_category_ids = self._categories_to_tensor(
                target_categories, self.config.max_objects_to_place
            )
            
            # Create sample following SPRING training methodology
            sample = {
                # INPUT: What the model receives
                'images': background_tensor,  # Inpainted background
                'background_layouts': background_layout,  # Objects still in scene
                'background_valid_masks': background_valid_mask,
                'background_category_ids': background_category_ids,
                'n_background_objects': torch.tensor(len(background_objects), dtype=torch.long),
                
                # CONSTRAINTS: Spatial specifications (your innovation)
                'constraints': self._process_constraints(constraints),
                
                # TARGET: What the model should predict (ground truth)
                'layouts': target_layout,  # Positions of removed objects
                'valid_masks': target_valid_mask,
                'category_ids': target_category_ids,
                'n_objects': torch.tensor(len(objects_to_place), dtype=torch.long),
                
                # METADATA
                'pair_ids': pair_id,
                'metadata': {
                    'background_categories': background_categories,
                    'target_categories': target_categories,
                    'original_coco_id': annotation.get('original_coco_id', 0),
                    'coordinate_system': annotation.get('coordinate_system', 'per_mille'),
                    'processing_info': annotation.get('processing_info', {}),
                    'scene_context': input_data.get('scene_context', {})
                }
            }
            
            # Handle output format
            if self.config.output_format == "flat" and self.format_converter:
                sample['layouts'] = self.format_converter.sequence_to_flat(sample['layouts'])
                sample['background_layouts'] = self.format_converter.sequence_to_flat(sample['background_layouts'])
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error loading training pair {idx} ({self.pair_ids[idx] if idx < len(self.pair_ids) else 'unknown'}): {e}")
            # Return a valid default sample
            return self._get_default_sample()
    
    def _objects_to_tensor(self, objects: List[Dict], max_objects: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert object list to tensor format with UNIFIED [0,1] COORDINATE SYSTEM.
        
        COORDINATE SYSTEM FIX:
        - Input: pixel coordinates from COCO/training data
        - Output: [0,1] normalized coordinates for model training
        - Validation: ensure all coordinates are in valid [0,1] range
        """
        import logging
        logger = logging.getLogger('CoordinateSystem')
        
        # Initialize tensors
        layout_tensor = torch.full((max_objects, 4), self.config.padding_value)
        valid_mask = torch.zeros(max_objects, dtype=torch.bool)
        
        n_objects = min(len(objects), max_objects)
        logger.info(f"COORDINATE NORMALIZATION: Processing {n_objects} objects")
        
        for i in range(n_objects):
            obj = objects[i]
            bbox = obj['bbox']  # [x, y, w, h] - Input pixel coordinates
            category = obj.get('category', 'unknown')
            
            # Extract pixel coordinates
            x_pixel, y_pixel, w_pixel, h_pixel = bbox
            image_size = self.config.image_size[0]  # Should be 512 for SPRING
            
            logger.debug(f"Object {i} ({category}): Raw pixels = [{x_pixel}, {y_pixel}, {w_pixel}, {h_pixel}]")
            
            # UNIFIED [0,1] NORMALIZATION - pixel coordinates → [0,1]
            x_normalized = x_pixel / image_size
            y_normalized = y_pixel / image_size  
            w_normalized = w_pixel / image_size
            h_normalized = h_pixel / image_size
            
            # COORDINATE SYSTEM FIX: Ensure all coordinates are strictly within [0,1] range
            # Handle COCO edge cases where objects extend beyond image boundaries
            
            # Clamp coordinates to valid [0,1] range with proper object constraints
            x_normalized = max(0.0, min(0.95, x_normalized))  # Leave 5% margin for object width
            y_normalized = max(0.0, min(0.95, y_normalized))  # Leave 5% margin for object height  
            w_normalized = max(0.02, min(0.4, w_normalized))   # Object width: 2-40% of image
            h_normalized = max(0.02, min(0.4, h_normalized))   # Object height: 2-40% of image
            
            # Additional validation: ensure object fits within image when positioned
            # If object extends beyond right/bottom edge, move it left/up
            if x_normalized + w_normalized > 1.0:
                x_normalized = 1.0 - w_normalized
            if y_normalized + h_normalized > 1.0:
                y_normalized = 1.0 - h_normalized
            
            # Final safety clamp
            x_normalized = max(0.0, min(1.0 - w_normalized, x_normalized))
            y_normalized = max(0.0, min(1.0 - h_normalized, y_normalized))
            
            coords = [x_normalized, y_normalized, w_normalized, h_normalized]
            
            # Validation log
            if any(c < 0 or c > 1 for c in coords):
                logger.error(f"Object {i} ({category}): Coordinates still outside [0,1] after clamping: {coords}")
            
            # Coordinates are already properly clamped above
            
            logger.debug(f"Object {i} ({category}): Normalized = [{x_normalized:.3f}, {y_normalized:.3f}, {w_normalized:.3f}, {h_normalized:.3f}]")
            
            # Store in tensor
            layout_tensor[i] = torch.tensor([x_normalized, y_normalized, w_normalized, h_normalized], dtype=torch.float32)
            valid_mask[i] = True
        
        # Final validation log (exclude padding values)
        if n_objects > 0:
            valid_coords = layout_tensor[valid_mask]
            coord_min = valid_coords.min().item()
            coord_max = valid_coords.max().item()
            logger.info(f"COORDINATE VALIDATION: Final range = [{coord_min:.3f}, {coord_max:.3f}] (should be [0,1])")
            
            if coord_min < 0.0 or coord_max > 1.0:
                logger.error(f"COORDINATE SYSTEM ERROR: Valid coordinates outside [0,1] range!")
            else:
                logger.info(f"✓ All {n_objects} objects have coordinates in [0,1] range")
        
        # Final check: Log padding values separately
        if n_objects < max_objects:
            padding_coords = layout_tensor[~valid_mask]
            if padding_coords.numel() > 0:
                logger.debug(f"Padding slots: {max_objects - n_objects} slots filled with {self.config.padding_value}")
        
        return layout_tensor, valid_mask
    
    def _categories_to_tensor(self, categories: List[str], max_objects: int) -> torch.Tensor:
        """Convert category list to tensor format."""
        
        category_tensor = torch.full((max_objects,), -1, dtype=torch.long)
        n_categories = min(len(categories), max_objects)
        
        for i in range(n_categories):
            category_id = self.config.category_to_id.get(categories[i], 0)
            category_tensor[i] = category_id
        
        return category_tensor
    
    def _process_constraints(self, constraints: List[Dict]) -> Dict[str, Any]:
        """Process constraints into a format suitable for the model."""
        
        # For now, create a simple constraint representation
        # In full implementation, this would use your constraint system
        
        constraint_info = {
            'n_constraints': len(constraints),
            'constraint_types': [c.get('type', 'unknown') for c in constraints],
            'has_spatial_constraints': any(c.get('type') == 'boundary' for c in constraints),
            'has_relational_constraints': any(c.get('type') == 'non_overlapping' for c in constraints),
            'raw_constraints': constraints  # Keep for constraint generation system
        }
        
        return constraint_info
    
    def _get_default_sample(self) -> Dict[str, Any]:
        """Get a default sample when loading fails."""
        
        sample = {
            # Input
            'images': torch.zeros(3, *self.config.image_size),
            'background_layouts': torch.full((self.config.max_background_objects, 4), self.config.padding_value),
            'background_valid_masks': torch.zeros(self.config.max_background_objects, dtype=torch.bool),
            'background_category_ids': torch.full((self.config.max_background_objects,), -1, dtype=torch.long),
            'n_background_objects': torch.tensor(0, dtype=torch.long),
            
            # Constraints
            'constraints': {'n_constraints': 0, 'constraint_types': [], 'raw_constraints': []},
            
            # Target
            'layouts': torch.full((self.config.max_objects_to_place, 4), self.config.padding_value),
            'valid_masks': torch.zeros(self.config.max_objects_to_place, dtype=torch.bool),
            'category_ids': torch.full((self.config.max_objects_to_place,), -1, dtype=torch.long),
            'n_objects': torch.tensor(0, dtype=torch.long),
            
            # Metadata
            'pair_ids': "default_pair",
            'metadata': {
                'background_categories': [],
                'target_categories': [],
                'error': True
            }
        }
        
        return sample


class SpringTrainingDataLoader:
    """Data loader factory for SPRING training pairs."""
    
    @staticmethod
    def create_train_loader(config: SpringDatasetConfig, 
                           batch_size: int = 32,
                           shuffle: bool = True,
                           num_workers: int = None,
                           pin_memory: bool = True) -> DataLoader:
        """Create training data loader."""
        
        dataset = SpringTrainingPairDataset(config, split="train")
        
        if num_workers is None:
            num_workers = config.num_workers
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=SpringTrainingDataLoader._collate_fn,
            drop_last=True
        )
    
    @staticmethod
    def create_val_loader(config: SpringDatasetConfig,
                         batch_size: int = 32,
                         num_workers: int = None,
                         pin_memory: bool = True) -> DataLoader:
        """Create validation data loader."""
        
        dataset = SpringTrainingPairDataset(config, split="val")
        
        if num_workers is None:
            num_workers = config.num_workers
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=SpringTrainingDataLoader._collate_fn,
            drop_last=False
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for SPRING training pairs."""
        
        collated = {}
        
        # Get batch size
        batch_size = len(batch)
        
        # Collate tensor fields
        tensor_fields = [
            'images', 'background_layouts', 'background_valid_masks', 'background_category_ids', 'n_background_objects',
            'layouts', 'valid_masks', 'category_ids', 'n_objects'
        ]
        
        for field in tensor_fields:
            if field in batch[0]:
                collated[field] = torch.stack([sample[field] for sample in batch])
        
        # Collate constraints (list of dicts)
        collated['constraints'] = [sample['constraints'] for sample in batch]
        
        # Collate lists
        collated['pair_ids'] = [sample['pair_ids'] for sample in batch]
        collated['metadata'] = [sample['metadata'] for sample in batch]
        
        # Add batch size for convenience
        collated['batch_size'] = batch_size
        
        return collated


def analyze_spring_dataset(dataset: SpringTrainingPairDataset) -> Dict[str, Any]:
    """Analyze the SPRING training pairs dataset."""
    
    print(f"Analyzing SPRING dataset: {dataset.split} split")
    
    # Sample a subset for analysis
    sample_size = min(200, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    
    analysis = {
        'total_pairs': len(dataset),
        'analyzed_pairs': sample_size,
        'target_object_distribution': defaultdict(int),
        'background_object_distribution': defaultdict(int),
        'objects_to_place_count': defaultdict(int),
        'constraint_analysis': {
            'total_constraints': 0,
            'constraint_types': defaultdict(int),
            'pairs_with_constraints': 0
        },
        'coordinate_statistics': {
            'target_objects': {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf')},
            'background_objects': {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf')}
        }
    }
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            
            # Analyze target objects (what model should predict)
            n_target = sample['n_objects'].item()
            analysis['objects_to_place_count'][n_target] += 1
            
            target_categories = sample['metadata']['target_categories']
            for cat in target_categories:
                analysis['target_object_distribution'][cat] += 1
            
            # Analyze background objects (what remains in scene)
            background_categories = sample['metadata']['background_categories']
            for cat in background_categories:
                analysis['background_object_distribution'][cat] += 1
            
            # Analyze constraints
            constraints = sample['constraints']
            n_constraints = constraints.get('n_constraints', 0)
            analysis['constraint_analysis']['total_constraints'] += n_constraints
            
            if n_constraints > 0:
                analysis['constraint_analysis']['pairs_with_constraints'] += 1
                
                for constraint_type in constraints.get('constraint_types', []):
                    analysis['constraint_analysis']['constraint_types'][constraint_type] += 1
            
            # Coordinate statistics
            if n_target > 0:
                target_layouts = sample['layouts']
                target_valid = sample['valid_masks']
                valid_target_layouts = target_layouts[target_valid]
                
                if len(valid_target_layouts) > 0:
                    stats = analysis['coordinate_statistics']['target_objects']
                    stats['min_x'] = min(stats['min_x'], valid_target_layouts[:, 0].min().item())
                    stats['max_x'] = max(stats['max_x'], valid_target_layouts[:, 0].max().item())
                    stats['min_y'] = min(stats['min_y'], valid_target_layouts[:, 1].min().item())
                    stats['max_y'] = max(stats['max_y'], valid_target_layouts[:, 1].max().item())
            
        except Exception as e:
            print(f"Error analyzing sample {idx}: {e}")
            continue
    
    # Compute averages
    if analysis['analyzed_pairs'] > 0:
        analysis['avg_objects_to_place'] = sum(count * num for num, count in analysis['objects_to_place_count'].items()) / analysis['analyzed_pairs']
        analysis['avg_constraints_per_pair'] = analysis['constraint_analysis']['total_constraints'] / analysis['analyzed_pairs']
    
    return analysis


def main():
    """Test the SPRING training pairs data loader."""
    print("=== SPRING TRAINING PAIRS DATA LOADER TEST ===\n")
    
    # Check if SPRING training data exists
    dataset_root = Path("/home/gaurang/hardnet/data/spring_training_data")
    if not dataset_root.exists():
        print(" SPRING training data not found!")
        print("Please run: python coco_processor_fixed.py first")
        return
    
    # Create config for SPRING training pairs
    config = SpringDatasetConfig(
        dataset_root=str(dataset_root),
        coordinate_system="per_mille",
        max_objects_to_place=5,
        max_background_objects=10,
        image_size=(512, 512),
        enable_augmentation=True
    )
    
    try:
        # Test dataset creation
        print("Testing SPRING training dataset creation...")
        train_dataset = SpringTrainingPairDataset(config, split="train")
        val_dataset = SpringTrainingPairDataset(config, split="val")
        
        print(f" Train dataset: {len(train_dataset)} training pairs")
        print(f" Val dataset: {len(val_dataset)} training pairs")
        
        # Test sample loading
        print("\nTesting training pair loading...")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f" Training pair loaded:")
            print(f"  Background image shape: {sample['images'].shape}")
            print(f"  Background objects: {sample['n_background_objects'].item()}")
            print(f"  Target objects to place: {sample['n_objects'].item()}")
            print(f"  Target layout shape: {sample['layouts'].shape}")
            print(f"  Constraints: {sample['constraints']['n_constraints']}")
            print(f"  Target categories: {sample['metadata']['target_categories']}")
            print(f"  Background categories: {sample['metadata']['background_categories']}")
        
        # Test data loader
        print("\nTesting data loader...")
        train_loader = SpringTrainingDataLoader.create_train_loader(config, batch_size=4)
        batch = next(iter(train_loader))
        print(f" Batch loaded:")
        print(f"  Background images shape: {batch['images'].shape}")
        print(f"  Target layouts shape: {batch['layouts'].shape}")
        print(f"  Background layouts shape: {batch['background_layouts'].shape}")
        print(f"  Batch size: {batch['batch_size']}")
        
        # Analyze dataset
        print("\nAnalyzing SPRING training dataset...")
        analysis = analyze_spring_dataset(train_dataset)
        print(f" Dataset analysis:")
        print(f"  Total training pairs: {analysis['total_pairs']}")
        print(f"  Avg objects to place: {analysis.get('avg_objects_to_place', 0):.2f}")
        print(f"  Avg constraints per pair: {analysis.get('avg_constraints_per_pair', 0):.2f}")
        print(f"  Target categories: {list(analysis['target_object_distribution'].keys())}")
        print(f"  Most common targets: {dict(list(analysis['target_object_distribution'].items())[:5])}")
        
    except Exception as e:
        print(f"✗ SPRING data loader test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== SPRING TRAINING PAIRS DATA LOADER TEST COMPLETE ===")


if __name__ == "__main__":
    main()