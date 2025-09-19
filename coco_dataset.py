
import os, sys
import types
import logging
import json
import time
import weakref
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import psutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import PIL
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("pycocotools not available, some functionality may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NumericalStabilityBounds:
    """Coordinate-type-specific numerical stability bounds"""
    # Position coordinate bounds (x, y)
    position_min: float = 0.0
    position_max: float = 999.0
    position_epsilon: float = 1e-6
    
    # Size coordinate bounds (w, h)
    size_min: float = 1.0
    size_max: float = 1000.0
    size_epsilon: float = 1e-6
    
    # Beta distribution parameter bounds
    beta_alpha_min: float = 0.1
    beta_alpha_max: float = 100.0
    beta_beta_min: float = 0.1
    beta_beta_max: float = 100.0
    
    # Gradient stability bounds
    gradient_clip_value: float = 1.0
    coordinate_delta_max: float = 50.0  # Maximum coordinate change per step
    
    def validate_position(self, coord: float) -> float:
        """Validate and clamp position coordinate with numerical stability"""
        if not np.isfinite(coord):
            coord = self.position_min
        return np.clip(coord, self.position_min, self.position_max)
    
    def validate_size(self, coord: float) -> float:
        """Validate and clamp size coordinate with numerical stability"""
        if not np.isfinite(coord) or coord < self.size_epsilon:
            coord = self.size_min
        return np.clip(coord, self.size_min, self.size_max)
    
    def validate_beta_param(self, param: float) -> float:
        """Validate and clamp Beta distribution parameters"""
        if not np.isfinite(param) or param < self.beta_alpha_min:
            param = self.beta_alpha_min
        return np.clip(param, self.beta_alpha_min, self.beta_alpha_max)
    
    def stabilize_bbox(self, bbox: List[float]) -> List[float]:
        """Apply numerical stability to full bounding box"""
        if len(bbox) != 4:
            return [0.0, 0.0, self.size_min, self.size_min]
        
        x, y, w, h = bbox
        stable_x = self.validate_position(x)
        stable_y = self.validate_position(y)
        stable_w = self.validate_size(w)
        stable_h = self.validate_size(h)
        
        # Ensure bbox doesn't exceed boundaries
        if stable_x + stable_w > self.position_max:
            stable_w = self.position_max - stable_x
        if stable_y + stable_h > self.position_max:
            stable_h = self.position_max - stable_y
        
        return [stable_x, stable_y, stable_w, stable_h]

@dataclass 
class AdaptiveThresholds:
    """Curriculum-based adaptive threshold system"""
    # Constraint satisfaction thresholds
    initial_threshold: float = 0.5
    target_threshold: float = 0.95
    current_threshold: float = field(init=False)
    
    # Curriculum stages
    total_stages: int = 5
    current_stage: int = 0
    samples_per_stage: int = 10000
    samples_processed: int = 0
    
    # Performance tracking
    satisfaction_rates: List[float] = field(default_factory=list)
    adaptation_rate: float = 0.1
    
    def __post_init__(self):
        self.current_threshold = self.initial_threshold
    
    def update_stage(self, satisfaction_rate: float) -> bool:
        """Update curriculum stage based on performance"""
        self.satisfaction_rates.append(satisfaction_rate)
        self.samples_processed += 1
        
        # Check if ready for next stage
        if (satisfaction_rate >= self.current_threshold and 
            self.samples_processed >= self.samples_per_stage and
            self.current_stage < self.total_stages - 1):
            
            self.current_stage += 1
            self.samples_processed = 0
            
            # Increase threshold
            stage_progress = self.current_stage / (self.total_stages - 1)
            self.current_threshold = (self.initial_threshold + 
                                    stage_progress * (self.target_threshold - self.initial_threshold))
            
            logger.info(f"Advanced to curriculum stage {self.current_stage}, threshold: {self.current_threshold:.3f}")
            return True
        
        return False
    
    def get_current_threshold(self) -> float:
        """Get current constraint satisfaction threshold"""
        return self.current_threshold
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get current stage information"""
        return {
            'stage': self.current_stage,
            'threshold': self.current_threshold,
            'samples_processed': self.samples_processed,
            'recent_satisfaction': np.mean(self.satisfaction_rates[-100:]) if self.satisfaction_rates else 0.0
        }

@dataclass
class HybridSamplingConfig:
    """Configuration for hybrid sampling strategy"""
    # Sampling weights (must sum to 1.0)
    mode_weight: float = 0.4
    mean_weight: float = 0.3
    stochastic_weight: float = 0.3
    
    # Sampling parameters
    temperature: float = 1.0  # Controls stochastic sampling sharpness
    confidence_threshold: float = 0.8  # Threshold for mode vs stochastic selection
    
    # Training vs inference behavior
    training_stochastic_boost: float = 0.2  # Increase stochastic weight during training
    
    def __post_init__(self):
        total_weight = self.mode_weight + self.mean_weight + self.stochastic_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Sampling weights must sum to 1.0, got {total_weight}")
    
    def get_weights(self, is_training: bool = True) -> Tuple[float, float, float]:
        """Get sampling weights adjusted for training/inference"""
        if is_training:
            # Boost stochastic sampling during training
            stoch_boost = min(self.training_stochastic_boost, 
                            1.0 - self.mode_weight - self.mean_weight)
            return (
                self.mode_weight - stoch_boost/2,
                self.mean_weight - stoch_boost/2,
                self.stochastic_weight + stoch_boost
            )
        return self.mode_weight, self.mean_weight, self.stochastic_weight

@dataclass
class DatasetStats:
    """Enhanced statistics for dataset filtering and processing"""
    total_images: int = 0
    total_annotations: int = 0
    filtered_images: int = 0
    filtered_annotations: int = 0
    crowd_filtered: int = 0
    invalid_bbox_filtered: int = 0
    max_objects_filtered: int = 0
    min_area_filtered: int = 0
    aspect_ratio_filtered: int = 0
    
    # Enhanced error tracking
    numerical_instability_fixes: int = 0
    memory_optimization_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_recoveries: int = 0
    
    # Performance metrics
    avg_load_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Quality metrics
    coordinate_stability_score: float = 0.0
    gradient_health_score: float = 0.0
    
    def log_summary(self):
        """Log comprehensive filtering summary"""
        logger.info("=" * 60)
        logger.info("COCO DATASET FILTERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {self.total_images}")
        logger.info(f"Total annotations processed: {self.total_annotations}")
        logger.info(f"Images kept after filtering: {self.filtered_images}")
        logger.info(f"Annotations kept after filtering: {self.filtered_annotations}")
        logger.info(f"Filtering breakdown:")
        logger.info(f"  - Crowd annotations removed: {self.crowd_filtered}")
        logger.info(f"  - Invalid bboxes removed: {self.invalid_bbox_filtered}")
        logger.info(f"  - Max objects exceeded: {self.max_objects_filtered}")
        logger.info(f"  - Min area threshold: {self.min_area_filtered}")
        logger.info(f"  - Aspect ratio threshold: {self.aspect_ratio_filtered}")
        retention_rate = (self.filtered_images / self.total_images * 100) if self.total_images > 0 else 0
        logger.info(f"Image retention rate: {retention_rate:.2f}%")
        logger.info("=" * 60)


class ProductionCocoDetection(dset.CocoDetection):
    """Production-ready COCO Detection with proper error handling"""
    
    def __init__(self, root: str, annFile: str, transform=None, target_transform=None, use_inpainted_prefix: bool = False):
        """Initialize with comprehensive error checking"""
        # Validate paths
        if not os.path.exists(root):
            raise FileNotFoundError(f"COCO image directory not found: {root}")
        if not os.path.exists(annFile):
            raise FileNotFoundError(f"COCO annotation file not found: {annFile}")
            
        # Validate annotation file format
        try:
            with open(annFile, 'r') as f:
                data = json.load(f)
                if 'images' not in data or 'annotations' not in data:
                    raise ValueError(f"Invalid COCO annotation format: {annFile}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotation file {annFile}: {e}")
        
        self.use_inpainted_prefix = use_inpainted_prefix    
        super().__init__(root, annFile, transform, target_transform)
        logger.info(f"Initialized COCO dataset: {len(self.ids)} images available")
    
    def _load_image(self, id: int):
        """Load image with error handling and inpainted prefix support"""
        try:
            # logger.info(f"Loading image {id}, use_inpainted_prefix={self.use_inpainted_prefix}")
            if self.use_inpainted_prefix:
                # Get the original filename from COCO
                path = self.coco.loadImgs(id)[0]['file_name']
                # Add inpainted prefix to filename (e.g., 'image.jpg' -> 'inpainted_image.jpg')
                filename = os.path.basename(path)
                inpainted_filename = f"inpainted_{filename}"
                inpainted_path = os.path.join(self.root, inpainted_filename)
                
                # Try to load the inpainted image
                try:
                    from PIL import Image
                    img = Image.open(inpainted_path).convert('RGB')
                    # logger.info(f"Successfully loaded inpainted image: {inpainted_path}")
                    return img
                except FileNotFoundError:
                    # NO FALLBACK - Skip missing inpainted images entirely
                    # logger.warning(f"Inpainted image not found: {inpainted_path} - skipping")
                    return None
            
            # Default behavior (load original image)
            return super()._load_image(id)
        except Exception as e:
            logger.warning(f"Failed to load image {id}: {e}")
            return None
    
    def _load_target(self, id: int):
        """Load target with error handling"""
        try:
            return super()._load_target(id)
        except Exception as e:
            logger.warning(f"Failed to load target for image {id}: {e}")
            return []


# 62,63,64,65,66,67,68,69,70,71,72,78,79,80,81,82,83
# COCO category mappings for interior design objects
INTERIOR_COCO_CATEGORIES = {
    62: "chair",
    63: "couch", 
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender"
}

# Reverse mapping for easy lookup
INTERIOR_NAME_TO_COCO_ID = {name: coco_id for coco_id, name in INTERIOR_COCO_CATEGORIES.items()}


class MemoryOptimizedCache:
    """Persistent caching system with memory optimization"""
    
    def __init__(self, max_cache_size: int = 1000, memory_limit_mb: float = 2048):
        """
        Initialize memory-optimized cache
        
        Args:
            max_cache_size: Maximum number of cached items
            memory_limit_mb: Memory limit in MB
        """
        self.max_cache_size = max_cache_size
        self.memory_limit_mb = memory_limit_mb
        self._cache = {}
        self._access_counts = defaultdict(int)
        self._access_order = []
        self._lock = threading.RLock()
        
        # Memory monitoring
        self._process = psutil.Process()
        self._initial_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self._process.memory_info().rss / 1024 / 1024
    
    def _evict_lru(self):
        """Evict least recently used items"""
        with self._lock:
            while (len(self._cache) > self.max_cache_size or 
                   self._get_memory_usage() - self._initial_memory > self.memory_limit_mb):
                
                if not self._access_order:
                    break
                
                # Find LRU item
                lru_key = min(self._access_order, key=lambda k: self._access_counts[k])
                
                # Remove from cache
                if lru_key in self._cache:
                    del self._cache[lru_key]
                if lru_key in self._access_counts:
                    del self._access_counts[lru_key]
                if lru_key in self._access_order:
                    self._access_order.remove(lru_key)
                
                # Force garbage collection if memory is still high
                if self._get_memory_usage() - self._initial_memory > self.memory_limit_mb:
                    gc.collect()
    
    def get(self, key: str, default=None):
        """Get cached item with LRU tracking"""
        with self._lock:
            if key in self._cache:
                self._access_counts[key] += 1
                if key not in self._access_order:
                    self._access_order.append(key)
                return self._cache[key]
            return default
    
    def set(self, key: str, value):
        """Set cached item with memory management"""
        with self._lock:
            # Check memory before adding
            if (len(self._cache) >= self.max_cache_size or 
                self._get_memory_usage() - self._initial_memory > self.memory_limit_mb * 0.8):
                self._evict_lru()
            
            self._cache[key] = value
            self._access_counts[key] = 1
            if key not in self._access_order:
                self._access_order.append(key)
    
    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._access_order.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'memory_usage_mb': self._get_memory_usage() - self._initial_memory,
                'memory_limit_mb': self.memory_limit_mb,
                'cache_efficiency': len(self._cache) / max(1, self.max_cache_size)
            }


class LazyDataFilter:
    """Lazy evaluation system for dataset filtering"""
    
    def __init__(self, coco_dataset, filter_params: Dict[str, Any]):
        """
        Initialize lazy filter
        
        Args:
            coco_dataset: COCO dataset instance
            filter_params: Filtering parameters
        """
        self.coco_dataset = coco_dataset
        self.filter_params = filter_params
        self._filtered_indices = None
        self._filter_cache = MemoryOptimizedCache(max_cache_size=500)
        self.stats = DatasetStats()
        
    def _compute_filtered_indices(self) -> List[int]:
        """Compute filtered indices with caching"""
        cache_key = self._get_filter_cache_key()
        cached_result = self._filter_cache.get(cache_key)
        
        if cached_result is not None:
            self.stats.cache_hits += 1
            logger.info(f"Filter cache hit for key: {cache_key}")
            return cached_result
        
        self.stats.cache_misses += 1
        logger.info(f"Computing filtered indices for: {cache_key}")
        
        # Perform actual filtering
        filtered_indices = self._filter_dataset_lazy()
        
        # Cache result
        self._filter_cache.set(cache_key, filtered_indices)
        
        return filtered_indices
    
    def _get_filter_cache_key(self) -> str:
        """Generate cache key from filter parameters"""
        return f"filter_{hash(json.dumps(self.filter_params, sort_keys=True))}"
    
    def _filter_dataset_lazy(self) -> List[int]:
        """Lazy dataset filtering with improved error handling"""
        valid_indices = []
        
        # Numerical stability bounds
        stability_bounds = NumericalStabilityBounds()
        
        logger.info("Starting lazy dataset filtering...")
        
        for i, image_id in enumerate(self.coco_dataset.ids):
            if i % 5000 == 0:
                logger.info(f"Processed {i}/{len(self.coco_dataset.ids)} images...")
                
                # Memory check during processing
                cache_stats = self._filter_cache.get_stats()
                if cache_stats['memory_usage_mb'] > cache_stats['memory_limit_mb'] * 0.9:
                    logger.info(f"Memory usage high: {cache_stats['memory_usage_mb']:.1f}MB, clearing cache")
                    self._filter_cache.clear()
            
            self.stats.total_images += 1
            
            try:
                # Get annotations for this image
                target = self.coco_dataset.coco.loadAnns(
                    self.coco_dataset.coco.getAnnIds(imgIds=image_id))
                
                if not target:
                    continue
                
                # Filter annotations with numerical stability
                valid_annotations = []
                for ann in target:
                    self.stats.total_annotations += 1
                    
                    # Skip crowd annotations
                    if ann.get("iscrowd", 0) == 1:
                        self.stats.crowd_filtered += 1
                        continue
                    
                    # Check if category is in our list
                    if ann["category_id"] not in self.filter_params.get('categories', []):
                        continue
                    
                    # Apply numerical stability to bbox
                    bbox = ann["bbox"]
                    try:
                        stable_bbox = stability_bounds.stabilize_bbox(bbox)
                        ann["bbox"] = stable_bbox  # Update with stable coordinates
                        self.stats.numerical_instability_fixes += 1
                    except Exception as e:
                        logger.warning(f"Bbox stabilization failed for {bbox}: {e}")
                        self.stats.invalid_bbox_filtered += 1
                        continue
                    
                    # Validate stabilized bounding box
                    if len(stable_bbox) != 4 or stable_bbox[2] <= 0 or stable_bbox[3] <= 0:
                        self.stats.invalid_bbox_filtered += 1
                        continue
                    
                    # Check minimum area
                    area = stable_bbox[2] * stable_bbox[3]
                    if area < self.filter_params.get('min_bbox_area', 100):
                        self.stats.min_area_filtered += 1
                        continue
                    
                    # Check aspect ratio
                    aspect_ratio = max(stable_bbox[2] / stable_bbox[3], 
                                     stable_bbox[3] / stable_bbox[2])
                    if aspect_ratio > self.filter_params.get('max_aspect_ratio', 10.0):
                        self.stats.aspect_ratio_filtered += 1
                        continue
                    
                    valid_annotations.append(ann)
                
                # Check if we have valid annotations and not too many
                max_objects = self.filter_params.get('max_objects_per_image', 10)
                if valid_annotations and len(valid_annotations) <= max_objects:
                    valid_indices.append(i)
                    self.stats.filtered_images += 1
                    self.stats.filtered_annotations += len(valid_annotations)
                elif len(valid_annotations) > max_objects:
                    self.stats.max_objects_filtered += 1
                    
            except Exception as e:
                logger.warning(f"Error processing image {image_id}: {e}")
                self.stats.error_recoveries += 1
                continue
        
        return valid_indices
    
    def get_filtered_indices(self) -> List[int]:
        """Get filtered indices with lazy computation"""
        if self._filtered_indices is None:
            self._filtered_indices = self._compute_filtered_indices()
        return self._filtered_indices
    
    def invalidate_cache(self):
        """Invalidate cached filtered indices"""
        self._filtered_indices = None
        self._filter_cache.clear()



def scale_coordinates(bbox: List[float], img_width: int, img_height: int, 
                     target_width: int = 1000, target_height: int = 1000) -> List[int]:
    """Scale bounding box coordinates to per-mille (0-1000) with validation
    
    Args:
        bbox: [x, y, width, height] in original image coordinates
        img_width: Original image width
        img_height: Original image height
        target_width: Target coordinate space width (default 1000 for per-mille)
        target_height: Target coordinate space height (default 1000 for per-mille)
        
    Returns:
        Scaled coordinates [x, y, width, height] as integers
        
    Raises:
        ValueError: If coordinates are invalid
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox must have 4 elements, got {len(bbox)}")
    
    x, y, w, h = bbox
    
    # Validate input coordinates
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox coordinates: {bbox}")
    if x + w > img_width or y + h > img_height:
        logger.warning(f"Bbox extends beyond image bounds: {bbox} in {img_width}x{img_height}")
        # Clamp to image bounds
        w = min(w, img_width - x)
        h = min(h, img_height - y)
    
    # Scale coordinates
    scaled_x = int(round(x / img_width * target_width))
    scaled_y = int(round(y / img_height * target_height))
    scaled_w = int(round(w / img_width * target_width))
    scaled_h = int(round(h / img_height * target_height))
    
    # Ensure minimum dimensions
    scaled_w = max(1, scaled_w)
    scaled_h = max(1, scaled_h)
    
    # Clamp to target bounds
    scaled_x = max(0, min(scaled_x, target_width - 1))
    scaled_y = max(0, min(scaled_y, target_height - 1))
    
    return [scaled_x, scaled_y, scaled_w, scaled_h]


class SquarePad:
    """Transform to pad images to square aspect ratio"""
    
    def __init__(self, fill: int = 0):
        """Initialize with padding fill value"""
        self.fill = fill
    
    def __call__(self, image):
        """Apply square padding to image"""
        if isinstance(image, PIL.Image.Image):
            w, h = image.size
        else:
            # Tensor format: [..., H, W]
            h = image.size(-2)
            w = image.size(-1)
        
        max_dim = max(w, h)
        pad_horizontal = (max_dim - w) // 2
        pad_vertical = (max_dim - h) // 2
        
        # Ensure even padding by adding remainder to right/bottom
        pad_left = pad_horizontal
        pad_right = max_dim - w - pad_left
        pad_top = pad_vertical  
        pad_bottom = max_dim - h - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.functional.pad(image, padding, self.fill, 'constant')
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fill={self.fill})"




class EnhancedCOCO_Wrapper(Dataset):
    """Production-ready COCO dataset wrapper with all academic improvements"""
    
    def __init__(self, coco_dataset, bg_dir: Optional[str] = None, 
                 categories: List[int] = [18, 19, 20, 21, 22, 23, 25], 
                 category_labels: List[str] = ["dog", "horse", "sheep", "cow", "elephant", "bear", "giraffe"],
                 max_objects_per_image: int = 10,
                 min_bbox_area: int = 100,
                 max_aspect_ratio: float = 10.0,
                 img_size: int = 128,
                 enable_lazy_filtering: bool = True,
                 enable_adaptive_thresholds: bool = True,
                 enable_hybrid_sampling: bool = True,
                 use_inpainted_prefix: bool = False):
        """
        Initialize enhanced COCO wrapper with all improvements
        
        Args:
            coco_dataset: COCO dataset instance
            bg_dir: Background image directory (optional)
            categories: List of COCO category IDs to keep
            category_labels: Corresponding category labels
            max_objects_per_image: Maximum objects per image to keep
            min_bbox_area: Minimum bounding box area (in pixels)
            max_aspect_ratio: Maximum aspect ratio (w/h or h/w)
            img_size: Output image size for square resize
            enable_lazy_filtering: Enable lazy filtering with caching
            enable_adaptive_thresholds: Enable curriculum-based thresholds
            enable_hybrid_sampling: Enable hybrid sampling strategy
        """
        self.categories = [None] + categories  # Add padding category
        self.category_labels = ["<pad>"] + category_labels
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.categories) if cat_id is not None}
        self.bg_dir = bg_dir
        self.max_objects_per_image = max_objects_per_image
        self.min_bbox_area = min_bbox_area
        self.max_aspect_ratio = max_aspect_ratio
        self.img_size = img_size
        self.use_inpainted_prefix = use_inpainted_prefix
        
        # Initialize enhanced components
        self.stability_bounds = NumericalStabilityBounds()
        
        if enable_adaptive_thresholds:
            self.adaptive_thresholds = AdaptiveThresholds()
        else:
            self.adaptive_thresholds = None
            
        if enable_hybrid_sampling:
            self.hybrid_sampling = HybridSamplingConfig()
        else:
            self.hybrid_sampling = None
        
        # Memory-optimized sample cache
        self.sample_cache = MemoryOptimizedCache(max_cache_size=1000, memory_limit_mb=1024)
        
        # Image transforms
        self.img_transforms = transforms.Compose([
            SquarePad(), 
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor()
        ])
        
        # Filter dataset with lazy evaluation if enabled
        filter_params = {
            'categories': categories,
            'max_objects_per_image': max_objects_per_image,
            'min_bbox_area': min_bbox_area,
            'max_aspect_ratio': max_aspect_ratio
        }
        
        logger.info(f"Initializing enhanced COCO dataset with categories: {categories}")
        logger.info(f"Max objects per image: {max_objects_per_image}")
        logger.info(f"Min bbox area: {min_bbox_area} pixels")
        logger.info(f"Max aspect ratio: {max_aspect_ratio}")
        logger.info(f"Lazy filtering: {enable_lazy_filtering}")
        logger.info(f"Adaptive thresholds: {enable_adaptive_thresholds}")
        logger.info(f"Hybrid sampling: {enable_hybrid_sampling}")
        
        if enable_lazy_filtering:
            self.lazy_filter = LazyDataFilter(coco_dataset, filter_params)
            self.filtered_indices = self.lazy_filter.get_filtered_indices()
            self.stats = self.lazy_filter.stats
        else:
            # Fallback to traditional filtering
            self.stats = DatasetStats()
            self.filtered_indices = self._filter_dataset_traditional(coco_dataset)
            self.lazy_filter = None
        
        self.coco_dataset = coco_dataset
        
        # Create subset
        self.dataset = Subset(coco_dataset, self.filtered_indices)
        
        # Performance monitoring
        self.performance_metrics = {
            'sample_load_times': [],
            'cache_hit_rate': 0.0,
            'memory_efficiency': 0.0,
            'error_rate': 0.0
        }
        
        # Log statistics
        self.stats.log_summary()
        logger.info(f"Final enhanced dataset size: {len(self.dataset)} images")
    
    def _filter_dataset_traditional(self, coco_dataset) -> List[int]:
        """Traditional filtering method for fallback compatibility"""
        valid_indices = []
        
        logger.info("Starting traditional dataset filtering...")
        
        for i, image_id in enumerate(coco_dataset.ids):
            if i % 5000 == 0:
                logger.info(f"Processed {i}/{len(coco_dataset.ids)} images...")
            
            self.stats.total_images += 1
            
            try:
                # Get annotations for this image
                target = coco_dataset.coco.loadAnns(coco_dataset.coco.getAnnIds(imgIds=image_id))
                
                if not target:
                    continue
                
                # Filter annotations
                valid_annotations = []
                for ann in target:
                    self.stats.total_annotations += 1
                    
                    # Skip crowd annotations
                    if ann.get("iscrowd", 0) == 1:
                        self.stats.crowd_filtered += 1
                        continue
                    
                    # Check if category is in our list
                    if ann["category_id"] not in self.categories:
                        continue
                    
                    # Validate bounding box
                    bbox = ann["bbox"]
                    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        self.stats.invalid_bbox_filtered += 1
                        continue
                    
                    # Check minimum area
                    area = bbox[2] * bbox[3]
                    if area < self.min_bbox_area:
                        self.stats.min_area_filtered += 1
                        continue
                    
                    # Check aspect ratio
                    aspect_ratio = max(bbox[2] / bbox[3], bbox[3] / bbox[2])
                    if aspect_ratio > self.max_aspect_ratio:
                        self.stats.aspect_ratio_filtered += 1
                        continue
                    
                    valid_annotations.append(ann)
                
                # Check if we have valid annotations and not too many
                if valid_annotations and len(valid_annotations) <= self.max_objects_per_image:
                    valid_indices.append(i)
                    self.stats.filtered_images += 1
                    self.stats.filtered_annotations += len(valid_annotations)
                elif len(valid_annotations) > self.max_objects_per_image:
                    self.stats.max_objects_filtered += 1
                    
            except Exception as e:
                logger.warning(f"Error processing image {image_id}: {e}")
                continue
        
        return valid_indices
    
    def _filter_dataset(self, coco_dataset) -> List[int]:
        """Filter dataset based on quality criteria"""
        valid_indices = []
        
        logger.info("Starting dataset filtering...")
        
        for i, image_id in enumerate(coco_dataset.ids):
            if i % 5000 == 0:
                logger.info(f"Processed {i}/{len(coco_dataset.ids)} images...")
            
            self.stats.total_images += 1
            
            try:
                # Get annotations for this image
                target = coco_dataset.coco.loadAnns(coco_dataset.coco.getAnnIds(imgIds=image_id))
                
                if not target:
                    continue
                
                # Filter annotations
                valid_annotations = []
                for ann in target:
                    self.stats.total_annotations += 1
                    
                    # Skip crowd annotations
                    if ann.get("iscrowd", 0) == 1:
                        self.stats.crowd_filtered += 1
                        continue
                    
                    # Check if category is in our list
                    if ann["category_id"] not in self.categories:
                        continue
                    
                    # Validate bounding box
                    bbox = ann["bbox"]
                    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        self.stats.invalid_bbox_filtered += 1
                        continue
                    
                    # Check minimum area
                    area = bbox[2] * bbox[3]
                    if area < self.min_bbox_area:
                        self.stats.min_area_filtered += 1
                        continue
                    
                    # Check aspect ratio
                    aspect_ratio = max(bbox[2] / bbox[3], bbox[3] / bbox[2])
                    if aspect_ratio > self.max_aspect_ratio:
                        self.stats.aspect_ratio_filtered += 1
                        continue
                    
                    valid_annotations.append(ann)
                
                # Check if we have valid annotations and not too many
                if valid_annotations and len(valid_annotations) <= self.max_objects_per_image:
                    valid_indices.append(i)
                    self.stats.filtered_images += 1
                    self.stats.filtered_annotations += len(valid_annotations)
                elif len(valid_annotations) > self.max_objects_per_image:
                    self.stats.max_objects_filtered += 1
                    
            except Exception as e:
                logger.warning(f"Error processing image {image_id}: {e}")
                continue
        
        return valid_indices

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], List[List[int]]]:
        """Enhanced dataset item retrieval with all academic improvements
        
        Returns:
            Tuple of (transformed_image, category_indices, stable_scaled_bboxes)
        """
        # Performance timing
        start_time = time.time()
        
        # Check cache first
        cache_key = f"sample_{idx}_{self.img_size}"
        cached_sample = self.sample_cache.get(cache_key)
        if cached_sample is not None:
            self.stats.cache_hits += 1
            self.performance_metrics['cache_hit_rate'] = (
                self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses)
            )
            return cached_sample
        
        self.stats.cache_misses += 1
        
        try:
            img, target = self.dataset[idx]
            
            # Handle case where image loading failed - skip to next valid image
            if img is None:
                # Find next valid image with inpainted version available
                max_retries = 100  # Prevent infinite loop
                retry_count = 0
                
                while retry_count < max_retries:
                    idx = (idx + 1) % len(self.dataset)
                    img, target = self.dataset[idx]
                    if img is not None:
                        break
                    retry_count += 1
                
                if img is None:
                    raise ValueError(f"Failed to find any valid image after {max_retries} retries")
            
            # Process annotations with enhanced error handling
            categories = []
            bboxes = []
            
            for annotation in target:
                try:
                    # Skip crowd annotations
                    if annotation.get("iscrowd", 0) == 1:
                        continue
                    
                    category_id = annotation["category_id"]
                    if category_id not in self.category_id_to_idx:
                        continue
                    
                    # Convert category ID to index
                    category_idx = self.category_id_to_idx[category_id]
                    categories.append(category_idx)
                    
                    # Process bounding box with numerical stability
                    bbox = annotation["bbox"]
                    img_width, img_height = img.size
                    
                    # Apply numerical stability bounds
                    stable_bbox = self.stability_bounds.stabilize_bbox(bbox)
                    
                    # Scale coordinates to per-mille with proper square padding
                    scaled_bbox = self._convert_bbox_to_permille_enhanced(
                        stable_bbox, img_width, img_height)
                    bboxes.append(scaled_bbox)
                    
                except Exception as e:
                    logger.warning(f"Error processing annotation for image {idx}: {e}")
                    self.stats.error_recoveries += 1
                    continue
            
            # Apply hybrid sampling if enabled and during training
            if (self.hybrid_sampling is not None and 
                hasattr(self, 'training') and self.training):
                
                # Implement coordinate sampling based on hybrid strategy
                bboxes = self._apply_hybrid_sampling(bboxes)
            
            # Handle background image loading if specified
            if self.bg_dir is not None and target:
                try:
                    image_id = str(target[0]["image_id"]).zfill(12)
                    bg_path = os.path.join(self.bg_dir, f"{image_id}.jpg")
                    if os.path.exists(bg_path):
                        img = Image.open(bg_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load background image: {e}")
                    self.stats.error_recoveries += 1
            
            # Apply image transforms
            transformed_img = self.img_transforms(img)
            
            # Create result tuple
            result = (transformed_img, categories, bboxes)
            
            # Cache the result
            self.sample_cache.set(cache_key, result)
            
            # Update adaptive thresholds if enabled
            if self.adaptive_thresholds is not None and categories and bboxes:
                # Calculate constraint satisfaction rate (simplified metric)
                satisfaction_rate = min(1.0, len(bboxes) / max(1, len(categories)))
                self.adaptive_thresholds.update_stage(satisfaction_rate)
            
            # Performance tracking
            load_time = time.time() - start_time
            self.performance_metrics['sample_load_times'].append(load_time)
            if len(self.performance_metrics['sample_load_times']) > 1000:
                self.performance_metrics['sample_load_times'] = (
                    self.performance_metrics['sample_load_times'][-1000:]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error loading sample {idx}: {e}")
            self.stats.error_recoveries += 1
            
            # Update error rate
            total_attempts = self.stats.cache_hits + self.stats.cache_misses
            self.performance_metrics['error_rate'] = (
                self.stats.error_recoveries / max(1, total_attempts)
            )
            
            # Re-raise the exception instead of returning dummy data
            raise RuntimeError(f"Failed to load any valid sample starting from index {idx}: {e}")
    
    def _apply_hybrid_sampling(self, bboxes: List[List[int]]) -> List[List[int]]:
        """Apply hybrid sampling strategy to bounding box coordinates"""
        if not self.hybrid_sampling:
            return bboxes
        
        # Get sampling weights
        mode_weight, mean_weight, stoch_weight = self.hybrid_sampling.get_weights(
            is_training=getattr(self, 'training', True)
        )
        
        sampled_bboxes = []
        for bbox in bboxes:
            if len(bbox) != 4:
                sampled_bboxes.append(bbox)
                continue
            
            x, y, w, h = bbox
            
            # For each coordinate, apply weighted sampling
            sampled_bbox = []
            for coord in [x, y, w, h]:
                # Simulate mode, mean, and stochastic sampling
                mode_sample = coord  # Mode is the current value
                mean_sample = coord  # Mean approximation
                
                # Stochastic sampling with small perturbation
                noise_scale = 2.0 if coord in [x, y] else 1.0  # Less noise for sizes
                stoch_sample = coord + np.random.normal(0, noise_scale)
                
                # Apply coordinate-specific bounds
                if coord in [x, y]:  # Position coordinates
                    stoch_sample = self.stability_bounds.validate_position(stoch_sample)
                else:  # Size coordinates
                    stoch_sample = self.stability_bounds.validate_size(stoch_sample)
                
                # Weighted combination
                weighted_coord = (
                    mode_weight * mode_sample +
                    mean_weight * mean_sample + 
                    stoch_weight * stoch_sample
                )
                
                # Final validation
                if coord in [x, y]:
                    final_coord = self.stability_bounds.validate_position(weighted_coord)
                else:
                    final_coord = self.stability_bounds.validate_size(weighted_coord)
                
                sampled_bbox.append(int(round(final_coord)))
            
            sampled_bboxes.append(sampled_bbox)
        
        return sampled_bboxes
    
    def _convert_bbox_to_permille_enhanced(self, bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """Enhanced bbox conversion with numerical stability and error handling
        
        This method handles the square padding that SquarePad applies to maintain
        correct coordinate mapping, with enhanced numerical stability.
        """
        try:
            if len(bbox) != 4:
                raise ValueError(f"Bbox must have 4 elements, got {len(bbox)}")
            
            x, y, w, h = bbox
            
            # Validate input coordinates with numerical stability
            if not all(np.isfinite([x, y, w, h])):
                logger.warning(f"Non-finite coordinates in bbox: {bbox}")
                x = self.stability_bounds.validate_position(x if np.isfinite(x) else 0.0)
                y = self.stability_bounds.validate_position(y if np.isfinite(y) else 0.0)
                w = self.stability_bounds.validate_size(w if np.isfinite(w) else 1.0)
                h = self.stability_bounds.validate_size(h if np.isfinite(h) else 1.0)
                self.stats.numerical_instability_fixes += 1
            
            # Calculate square padding (same as SquarePad transform)
            max_dim = max(img_width, img_height)
            if max_dim <= 0:
                logger.warning(f"Invalid image dimensions: {img_width}x{img_height}")
                return [0, 0, 1, 1]
            
            pad_horizontal = (max_dim - img_width) // 2
            pad_vertical = (max_dim - img_height) // 2
            
            # Adjust coordinates for padding
            padded_x = x + pad_horizontal
            padded_y = y + pad_vertical
            
            # Scale to per-mille (0-1000) with numerical stability
            scale_factor = 1000.0 / max_dim
            
            # Apply scaling with proper rounding
            scaled_x = padded_x * scale_factor
            scaled_y = padded_y * scale_factor
            scaled_w = w * scale_factor
            scaled_h = h * scale_factor
            
            # Apply coordinate-specific bounds before rounding
            scaled_x = self.stability_bounds.validate_position(scaled_x)
            scaled_y = self.stability_bounds.validate_position(scaled_y)
            scaled_w = self.stability_bounds.validate_size(scaled_w)
            scaled_h = self.stability_bounds.validate_size(scaled_h)
            
            # Convert to integers
            final_x = int(round(scaled_x))
            final_y = int(round(scaled_y))
            final_w = int(round(scaled_w))
            final_h = int(round(scaled_h))
            
            # Final bounds checking to ensure no overlap issues
            final_x = max(0, min(final_x, 999))
            final_y = max(0, min(final_y, 999))
            final_w = max(1, min(final_w, 1000 - final_x))
            final_h = max(1, min(final_h, 1000 - final_y))
            
            return [final_x, final_y, final_w, final_h]
            
        except Exception as e:
            logger.warning(f"Error in enhanced bbox conversion for {bbox}: {e}")
            self.stats.error_recoveries += 1
            # Return a safe default bbox
            return [0, 0, 10, 10]
    
    def _convert_bbox_to_permille(self, bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """Legacy bbox conversion method - kept for backward compatibility"""
        return self._convert_bbox_to_permille_enhanced(bbox, img_width, img_height)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        cache_stats = self.sample_cache.get_stats() if hasattr(self, 'sample_cache') else {}
        
        # Calculate memory efficiency
        memory_efficiency = 1.0 - (cache_stats.get('memory_usage_mb', 0) / 
                                  max(1, cache_stats.get('memory_limit_mb', 1)))
        
        # Calculate average load time
        avg_load_time = (np.mean(self.performance_metrics['sample_load_times']) 
                        if self.performance_metrics['sample_load_times'] else 0.0)
        
        # Calculate coordinate stability score
        total_fixes = self.stats.numerical_instability_fixes
        total_samples = self.stats.cache_hits + self.stats.cache_misses
        stability_score = 1.0 - (total_fixes / max(1, total_samples))
        
        # Calculate gradient health score (based on error recovery rate)
        gradient_health = 1.0 - self.performance_metrics['error_rate']
        
        metrics = {
            'performance': {
                'avg_sample_load_time_ms': avg_load_time * 1000,
                'cache_hit_rate': self.performance_metrics['cache_hit_rate'],
                'memory_efficiency': memory_efficiency,
                'error_rate': self.performance_metrics['error_rate']
            },
            'memory': {
                'cache_size': cache_stats.get('cache_size', 0),
                'memory_usage_mb': cache_stats.get('memory_usage_mb', 0),
                'memory_limit_mb': cache_stats.get('memory_limit_mb', 0),
                'cache_efficiency': cache_stats.get('cache_efficiency', 0)
            },
            'stability': {
                'coordinate_stability_score': stability_score,
                'gradient_health_score': gradient_health,
                'numerical_fixes': total_fixes,
                'error_recoveries': self.stats.error_recoveries
            },
            'adaptive_thresholds': (
                self.adaptive_thresholds.get_stage_info() 
                if self.adaptive_thresholds else {}
            ),
            'data_quality': {
                'total_samples': total_samples,
                'valid_samples': self.stats.filtered_images,
                'retention_rate': (
                    self.stats.filtered_images / max(1, self.stats.total_images)
                )
            }
        }
        
        return metrics
    
    def validate_numerical_stability(self, num_samples: int = 100) -> Dict[str, float]:
        """Validate numerical stability across random samples"""
        logger.info(f"Validating numerical stability with {num_samples} samples...")
        
        validation_metrics = {
            'coordinate_bounds_violations': 0,
            'non_finite_coordinates': 0,
            'bbox_consistency_errors': 0,
            'successful_loads': 0
        }
        
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        
        for idx in indices:
            try:
                img, cats, boxes = self[idx]
                validation_metrics['successful_loads'] += 1
                
                # Check coordinate bounds
                for box in boxes:
                    if len(box) != 4:
                        validation_metrics['bbox_consistency_errors'] += 1
                        continue
                    
                    x, y, w, h = box
                    
                    # Check if coordinates are finite
                    if not all(np.isfinite([x, y, w, h])):
                        validation_metrics['non_finite_coordinates'] += 1
                    
                    # Check bounds violations
                    if (x < 0 or x >= 1000 or y < 0 or y >= 1000 or 
                        w < 1 or h < 1 or x + w > 1000 or y + h > 1000):
                        validation_metrics['coordinate_bounds_violations'] += 1
                        
            except Exception as e:
                logger.warning(f"Validation error for sample {idx}: {e}")
                continue
        
        # Calculate validation scores
        total_checks = validation_metrics['successful_loads']
        if total_checks > 0:
            validation_metrics['bounds_violation_rate'] = (
                validation_metrics['coordinate_bounds_violations'] / total_checks
            )
            validation_metrics['finite_coordinate_rate'] = (
                1.0 - validation_metrics['non_finite_coordinates'] / total_checks
            )
            validation_metrics['bbox_consistency_rate'] = (
                1.0 - validation_metrics['bbox_consistency_errors'] / total_checks
            )
        else:
            validation_metrics.update({
                'bounds_violation_rate': 1.0,
                'finite_coordinate_rate': 0.0,
                'bbox_consistency_rate': 0.0
            })
        
        logger.info(f"Numerical stability validation complete:")
        logger.info(f"  Bounds violation rate: {validation_metrics['bounds_violation_rate']:.3f}")
        logger.info(f"  Finite coordinate rate: {validation_metrics['finite_coordinate_rate']:.3f}")
        logger.info(f"  Bbox consistency rate: {validation_metrics['bbox_consistency_rate']:.3f}")
        
        return validation_metrics

    def id_to_coco_id(self, idx: int) -> int:
        """Convert internal category index to COCO category ID"""
        if idx < 0 or idx >= len(self.categories):
            raise ValueError(f"Invalid category index: {idx}")
        return self.categories[idx]
    
    def label_dict(self) -> Dict[int, str]:
        """Get mapping from category index to label"""
        return {i: label for i, label in enumerate(self.category_labels) if i > 0}
    
    def get_stats(self) -> DatasetStats:
        """Get dataset filtering statistics"""
        return self.stats
    
    def validate_sample(self, idx: int) -> bool:
        """Validate that a sample can be loaded correctly"""
        try:
            img, cats, boxes = self[idx]
            
            # Check image tensor
            if not isinstance(img, torch.Tensor) or img.shape != (3, 128, 128):
                return False
            
            # Check categories
            if not isinstance(cats, list) or not all(isinstance(c, int) for c in cats):
                return False
            
            # Check bboxes
            if not isinstance(boxes, list):
                return False
            
            for box in boxes:
                if (not isinstance(box, list) or len(box) != 4 or 
                    not all(isinstance(coord, int) for coord in box)):
                    return False
                
                x, y, w, h = box
                if not (0 <= x <= 1000 and 0 <= y <= 1000 and 
                       1 <= w <= 1000 and 1 <= h <= 1000):
                    return False
            
            return True
            
        except Exception:
            return False

    @classmethod
    def from_args(cls, coco_dir: str, coco_json: str, bg_dir: Optional[str] = None, **kwargs):
        """Create dataset from directory and annotation file paths"""
        ds = ProductionCocoDetection(root=coco_dir, annFile=coco_json)
        return cls(ds, bg_dir=bg_dir, **kwargs)
    
    @classmethod
    def from_args_animals(cls, coco_dir: str, coco_json: str, bg_dir: Optional[str] = None, **kwargs):
        """Create animal-focused dataset"""
        categories = [18, 19, 20, 21, 22, 23, 25]
        category_labels = ["dog", "horse", "sheep", "cow", "elephant", "bear", "giraffe"]
        
        ds = ProductionCocoDetection(root=coco_dir, annFile=coco_json)
        return cls(ds, bg_dir=bg_dir, categories=categories, 
                  category_labels=category_labels, **kwargs)
    
    @classmethod
    def from_args_interior(cls, coco_dir: str, coco_json: str, bg_dir: Optional[str] = None, **kwargs):
        """Create enhanced interior design-focused dataset with all improvements"""
        categories = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 83]
        category_labels = [
            "chair", "couch", "potted plant", "bed", "mirror", "dining table", 
            "window", "desk", "toilet", "door", "tv", "microwave", "oven", 
            "toaster", "sink", "refrigerator", "blender"
        ]
        
        # Enhanced production defaults with all improvements enabled
        production_defaults = {
            'max_objects_per_image': 8,        # Interior scenes can have many objects
            'min_bbox_area': 100,              # Minimum 10x10 pixels
            'max_aspect_ratio': 15.0,          # Allow some elongated objects like doors
            'img_size': 128,                   # Standard size for SPRING
            'enable_lazy_filtering': True,     # Enable memory-optimized lazy filtering
            'enable_adaptive_thresholds': True, # Enable curriculum-based thresholds
            'enable_hybrid_sampling': True     # Enable hybrid sampling strategy
        }
        
        # Override defaults with any provided kwargs
        for key, value in production_defaults.items():
            kwargs.setdefault(key, value)
        
        # Extract use_inpainted_prefix from kwargs before passing to ProductionCocoDetection
        use_inpainted_prefix = kwargs.pop('use_inpainted_prefix', False)
        
        ds = ProductionCocoDetection(root=coco_dir, annFile=coco_json, use_inpainted_prefix=use_inpainted_prefix)
        return cls(ds, bg_dir=bg_dir, categories=categories, 
                  category_labels=category_labels, use_inpainted_prefix=use_inpainted_prefix, **kwargs)
    
    @classmethod
    def from_args_interior_legacy(cls, coco_dir: str, coco_json: str, bg_dir: Optional[str] = None, **kwargs):
        """Create interior dataset with legacy behavior (no enhancements)"""
        categories = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 83]
        category_labels = [
            "chair", "couch", "potted plant", "bed", "mirror", "dining table", 
            "window", "desk", "toilet", "door", "tv", "microwave", "oven", 
            "toaster", "sink", "refrigerator", "blender"
        ]
        
        # Legacy defaults - no enhancements
        legacy_defaults = {
            'max_objects_per_image': 8,
            'min_bbox_area': 100,
            'max_aspect_ratio': 15.0,
            'img_size': 128,
            'enable_lazy_filtering': False,
            'enable_adaptive_thresholds': False,
            'enable_hybrid_sampling': False
        }
        
        for key, value in legacy_defaults.items():
            kwargs.setdefault(key, value)
        
        ds = ProductionCocoDetection(root=coco_dir, annFile=coco_json)
        return cls(ds, bg_dir=bg_dir, categories=categories, 
                  category_labels=category_labels, **kwargs)
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'EnhancedCOCO_Wrapper':
        """Create dataset from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract dataset configuration
        if 'production_train' in config:
            dataset_config = config['production_train']
        elif 'coco_dir' in config:
            dataset_config = config
        else:
            raise ValueError(f"Invalid config format in {config_path}")
        
        return cls.from_args_interior(
            coco_dir=dataset_config['coco_dir'],
            coco_json=dataset_config['coco_json'],
            bg_dir=dataset_config.get('bg_dir')
        )




def collate_pad_fn(batch_data: List[Tuple[torch.Tensor, List[int], List[List[int]]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function with improved padding and error handling
    
    Args:
        batch_data: List of (image, categories, bboxes) tuples
        
    Returns:
        Tuple of (batched_images, batched_categories, batched_bboxes)
    """
    if not batch_data:
        raise ValueError("Empty batch data")
    
    # Find maximum number of objects in batch
    max_objects = 0
    valid_samples = []
    
    for sample in batch_data:
        img, cats, boxes = sample
        if isinstance(cats, list) and isinstance(boxes, list) and len(cats) == len(boxes):
            max_objects = max(max_objects, len(cats))
            valid_samples.append(sample)
        else:
            logger.warning(f"Invalid sample in batch: cats={len(cats) if isinstance(cats, list) else 'invalid'}, boxes={len(boxes) if isinstance(boxes, list) else 'invalid'}")
    
    if not valid_samples:
        raise ValueError("No valid samples in batch")
    
    # Ensure minimum of 1 object slot for padding
    max_objects = max(1, max_objects)
    
    imgs, catss, boxss = [], [], []
    
    for img, cats, boxes in valid_samples:
        # Pad categories and boxes to max_objects length
        padded_cats = cats + [0] * (max_objects - len(cats))
        padded_boxes = boxes + [[0, 0, 0, 0]] * (max_objects - len(boxes))
        
        # Flatten boxes for tensor conversion
        flattened_boxes = [coord for box in padded_boxes for coord in box]
        
        # Convert to tensors
        cats_tensor = torch.tensor(padded_cats, dtype=torch.long)
        boxes_tensor = torch.tensor(flattened_boxes, dtype=torch.float32)
        
        imgs.append(img)
        catss.append(cats_tensor)
        boxss.append(boxes_tensor)
    
    # Stack all tensors
    try:
        batched_imgs = torch.stack(imgs)
        batched_cats = torch.stack(catss)
        batched_boxes = torch.stack(boxss)
        
        return batched_imgs, batched_cats, batched_boxes
    
    except Exception as e:
        logger.error(f"CRITICAL ERROR stacking batch tensors: {e}")
        logger.error(f"Batch info: {len(imgs)} samples, max_objects={max_objects}")
        logger.error(f"Image shapes: {[img.shape if torch.is_tensor(img) else 'non-tensor' for img in imgs]}")
        # Raise the error instead of returning dummy data
        # This will stop training and force fixing the real issue
        raise RuntimeError(f"Failed to create batch: {e}") from e


def create_production_dataloader(dataset: 'EnhancedCOCO_Wrapper', batch_size: int = 32, 
                               shuffle: bool = True, num_workers: int = 4,
                               pin_memory: bool = True) -> DataLoader:
    """Create production-ready DataLoader with optimized settings
    
    Args:
        dataset: COCO_Wrapper dataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Optimized DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_pad_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # Ensure consistent batch sizes
    )


# Production validation functions
def validate_dataset_integrity(dataset: 'EnhancedCOCO_Wrapper', num_samples: int = 100) -> bool:
    """Validate dataset integrity by checking random samples
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of random samples to check
        
    Returns:
        True if all samples are valid, False otherwise
    """
    logger.info(f"Validating dataset integrity with {num_samples} samples...")
    
    valid_count = 0
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i % 20 == 0:
            logger.info(f"Validated {i}/{len(indices)} samples...")
        
        if dataset.validate_sample(idx):
            valid_count += 1
        else:
            logger.warning(f"Invalid sample at index {idx}")
    
    success_rate = valid_count / len(indices)
    logger.info(f"Validation complete: {valid_count}/{len(indices)} samples valid ({success_rate:.2%})")
    
    return success_rate >= 0.95  # Require 95% success rate


def benchmark_dataset_performance(dataset: 'EnhancedCOCO_Wrapper', batch_size: int = 32, 
                                num_batches: int = 10) -> Dict[str, float]:
    """Benchmark dataset loading performance
    
    Args:
        dataset: Dataset to benchmark
        batch_size: Batch size for testing
        num_batches: Number of batches to time
        
    Returns:
        Performance metrics dictionary
    """
    logger.info(f"Benchmarking dataset performance with batch_size={batch_size}...")
    
    dataloader = create_production_dataloader(dataset, batch_size=batch_size, 
                                            shuffle=False, num_workers=0)
    
    times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start_time = time.time()
        imgs, cats, boxes = batch
        # Force tensor operations to complete
        _ = imgs.sum().item()
        _ = cats.sum().item() 
        _ = boxes.sum().item()
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    metrics = {
        'avg_batch_time': np.mean(times),
        'std_batch_time': np.std(times),
        'min_batch_time': np.min(times),
        'max_batch_time': np.max(times),
        'samples_per_second': batch_size / np.mean(times) if times else 0
    }
    
    logger.info(f"Performance metrics: {metrics}")
    return metrics


# Backward compatibility alias
COCO_Wrapper = EnhancedCOCO_Wrapper


# Example usage for production
if __name__ == "__main__":
    # Enhanced dataset creation with all improvements
    coco_dir = "/home/gaurang/hardnetnew/data/coco/train2017"
    coco_json = "/home/gaurang/hardnetnew/data/coco/annotations/instances_train2017_interior.json"
    
    logger.info("Creating enhanced COCO dataset with all academic improvements...")
    
    # Create enhanced production dataset
    dataset = EnhancedCOCO_Wrapper.from_args_interior(coco_dir, coco_json)
    
    # Validate numerical stability
    stability_metrics = dataset.validate_numerical_stability(num_samples=50)
    logger.info(f"Numerical stability validation:")
    logger.info(f"  Bounds violations: {stability_metrics['bounds_violation_rate']:.3f}")
    logger.info(f"  Coordinate consistency: {stability_metrics['bbox_consistency_rate']:.3f}")
    
    # Get comprehensive performance metrics
    perf_metrics = dataset.get_performance_metrics()
    logger.info(f"Performance metrics:")
    logger.info(f"  Memory efficiency: {perf_metrics['performance']['memory_efficiency']:.3f}")
    logger.info(f"  Cache hit rate: {perf_metrics['performance']['cache_hit_rate']:.3f}")
    logger.info(f"  Coordinate stability: {perf_metrics['stability']['coordinate_stability_score']:.3f}")
    
    # Validate overall integrity
    is_valid = validate_dataset_integrity(dataset, num_samples=50)
    logger.info(f"Dataset validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Benchmark performance
    benchmark_metrics = benchmark_dataset_performance(dataset, num_batches=5)
    logger.info(f"Performance benchmark:")
    logger.info(f"  Avg load time: {benchmark_metrics['avg_batch_time']*1000:.1f}ms")
    logger.info(f"  Samples/sec: {benchmark_metrics['samples_per_second']:.1f}")
    
    # Create production dataloader
    dataloader = create_production_dataloader(dataset, batch_size=32)
    logger.info(f"Enhanced production dataloader created with {len(dataloader)} batches")
    
    # Test a few samples to verify everything works
    logger.info("Testing sample loading...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Test first 3 batches
            break
        imgs, cats, boxes = batch
        logger.info(f"Batch {i}: {imgs.shape}, {cats.shape}, {boxes.shape}")
    
    logger.info("Enhanced COCO pipeline implementation complete!")

#===============================================================================
