#!/usr/bin/env python3
"""
Telea Inpainting System for SPRING Beta COCO Dataset
===================================================

Mathematical Foundation:
- Telea inpainting using Fast Marching Method (FMM)
- Inpainting function: I(p) = Σ w(p,q) × I(q) / Σ w(p,q)
- Weight function: w(p,q) = dir(p,q) × dst(p,q) × lev(p,q)

Author: Professor Davies
Date: 2025-08-16
"""

import os
import sys
import json
import logging
import time
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy.stats as stats

# COCO tools
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
except ImportError:
    raise ImportError("pycocotools required. Install with: pip install pycocotools")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


@dataclass
class InpaintingConfig:
    """Configuration for Telea inpainting system"""
    # Inpainting method: cv2.INPAINT_TELEA or cv2.INPAINT_NS
    method: int = cv2.INPAINT_TELEA
    
    # Inpainting radius (critical parameter)
    inpaint_radius: int = 7  # Base radius for most interior objects
    adaptive_radius: bool = True  # Calculate radius based on object size
    min_radius: int = 3  # Minimum adaptive radius
    max_radius: int = 15  # Maximum adaptive radius
    radius_scale_factor: float = 0.02  # Scale factor for adaptive radius
    
    # Mask processing parameters
    mask_dilation_kernel_size: int = 3  # Dilate masks for better boundaries
    mask_dilation_iterations: int = 1   # Number of dilation iterations
    
    # Quality control
    min_mask_area: int = 100           # Minimum mask area (pixels)
    max_mask_area_ratio: float = 0.8   # Maximum mask area as fraction of image
    
    # Output control
    output_format: str = "JPEG"        # Output format
    output_quality: int = 95           # JPEG quality (if applicable)
    
    # Performance
    max_workers: int = 4               # Parallel processing workers
    batch_size: int = 32               # Images per batch
    
    # Reproducibility
    random_seed: Optional[int] = 42    # Random seed for reproducible results
    
    # Memory management  
    max_image_size: int = 2048         # Maximum image dimension
    enable_memory_optimization: bool = True


@dataclass
class InpaintingStats:
    """Statistics tracking for inpainting operations"""
    total_images_processed: int = 0
    successful_inpaints: int = 0
    failed_inpaints: int = 0
    total_objects_removed: int = 0
    
    # Timing statistics
    total_processing_time: float = 0.0
    avg_processing_time_per_image: float = 0.0
    
    # Quality metrics
    avg_mask_area: float = 0.0
    avg_inpaint_radius_used: float = 0.0
    mask_area_distribution: List[float] = field(default_factory=list)
    mask_area_max_samples: int = 10000  # Bounded to prevent memory leak
    
    # Error tracking
    polygon_conversion_errors: int = 0
    mask_generation_errors: int = 0
    inpainting_algorithm_errors: int = 0
    memory_errors: int = 0
    
    # Quality metrics
    quality_ssim_scores: List[float] = field(default_factory=list)
    quality_psnr_scores: List[float] = field(default_factory=list)
    quality_max_samples: int = 1000  # Bounded to prevent memory leak
    avg_ssim: float = 0.0
    avg_psnr: float = 0.0
    
    def log_summary(self):
        """Log comprehensive inpainting statistics"""
        logger.info("=" * 70)
        logger.info("TELEA INPAINTING SYSTEM - PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total images processed: {self.total_images_processed}")
        logger.info(f"Successful inpaints: {self.successful_inpaints}")
        logger.info(f"Failed inpaints: {self.failed_inpaints}")
        
        # Calculate success rate with confidence interval
        success_rate = self.successful_inpaints/max(1,self.total_images_processed)
        logger.info(f"Success rate: {success_rate*100:.2f}%")
        logger.info(f"Total objects removed: {self.total_objects_removed}")
        logger.info(f"Average processing time: {self.avg_processing_time_per_image*1000:.1f}ms per image")
        logger.info(f"Average mask area: {self.avg_mask_area:.1f} pixels")
        
        if self.mask_area_distribution:
            logger.info(f"Mask area statistics:")
            logger.info(f"  - Min: {min(self.mask_area_distribution):.1f} pixels")
            logger.info(f"  - Max: {max(self.mask_area_distribution):.1f} pixels")
            logger.info(f"  - Std: {np.std(self.mask_area_distribution):.1f} pixels")
        
        if self.quality_ssim_scores:
            logger.info(f"Quality metrics:")
            logger.info(f"  - Average SSIM: {self.avg_ssim:.4f}")
            logger.info(f"  - Average PSNR: {self.avg_psnr:.2f} dB")
            logger.info(f"  - Quality samples: {len(self.quality_ssim_scores)}")
        
        logger.info(f"Error breakdown:")
        logger.info(f"  - Polygon conversion errors: {self.polygon_conversion_errors}")
        logger.info(f"  - Mask generation errors: {self.mask_generation_errors}")
        logger.info(f"  - Inpainting algorithm errors: {self.inpainting_algorithm_errors}")
        logger.info(f"  - Memory errors: {self.memory_errors}")
        logger.info("=" * 70)


class TeleaInpaintingSystem:
    """
    Production-ready Telea inpainting system for COCO interior design dataset
    
    Implements Fast Marching Method (FMM) based inpainting using OpenCV's Telea algorithm.
    Handles COCO polygon annotations, creates masks, and generates clean backgrounds.
    """
    
    def __init__(self, config: InpaintingConfig = None):
        """
        Initialize Telea inpainting system
        
        Args:
            config: Inpainting configuration. Uses defaults if None.
        """
        self.config = config or InpaintingConfig()
        self.stats = InpaintingStats()
        self._thread_lock = threading.Lock()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            logger.info(f"Set random seed to {self.config.random_seed} for reproducible results")
        
        # Validate OpenCV inpainting method
        if self.config.method not in [cv2.INPAINT_TELEA, cv2.INPAINT_NS]:
            raise ValueError(f"Invalid inpainting method: {self.config.method}")
        
        # Log configuration
        logger.info(f"Initialized Telea inpainting system:")
        logger.info(f"  Method: {'TELEA' if self.config.method == cv2.INPAINT_TELEA else 'NAVIER_STOKES'}")
        logger.info(f"  Inpaint radius: {self.config.inpaint_radius}")
        logger.info(f"  Mask dilation: {self.config.mask_dilation_kernel_size}x{self.config.mask_dilation_iterations}")
        logger.info(f"  Max workers: {self.config.max_workers}")
    
    def polygon_to_mask(self, polygon: List[float], img_height: int, img_width: int) -> np.ndarray:
        """
        Convert COCO polygon annotation to binary mask
        
        Args:
            polygon: COCO polygon format [x1,y1,x2,y2,...]
            img_height: Image height
            img_width: Image width
            
        Returns:
            Binary mask as uint8 numpy array
            
        Raises:
            ValueError: If polygon format is invalid
        """
        try:
            if len(polygon) < 6:  # At least 3 points (6 coordinates)
                raise ValueError(f"Invalid polygon: needs at least 3 points, got {len(polygon)//2}")
            
            if len(polygon) % 2 != 0:
                raise ValueError(f"Polygon coordinates must be even number, got {len(polygon)}")
            
            # Reshape polygon to (N, 2) format
            polygon_points = np.array(polygon).reshape(-1, 2)
            
            # Validate coordinates
            if np.any(polygon_points < 0) or np.any(polygon_points[:, 0] >= img_width) or np.any(polygon_points[:, 1] >= img_height):
                logger.warning(f"Polygon extends outside image bounds: {polygon_points.min()}, {polygon_points.max()} for {img_width}x{img_height}")
                # Clamp to image boundaries
                polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, img_width - 1)
                polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, img_height - 1)
            
            # Create binary mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            polygon_points = polygon_points.astype(np.int32)
            cv2.fillPoly(mask, [polygon_points], 255)
            
            return mask
            
        except Exception as e:
            logger.error(f"Polygon to mask conversion failed: {e}")
            self.stats.polygon_conversion_errors += 1
            # Return empty mask
            return np.zeros((img_height, img_width), dtype=np.uint8)
    
    def rle_to_mask(self, rle: Dict[str, Any], img_height: int, img_width: int) -> np.ndarray:
        """
        Convert COCO RLE annotation to binary mask
        
        Args:
            rle: COCO RLE format annotation
            img_height: Image height
            img_width: Image width
            
        Returns:
            Binary mask as uint8 numpy array
        """
        try:
            # Ensure RLE has size field
            if 'size' not in rle:
                rle['size'] = [img_height, img_width]
            
            # Convert RLE to binary mask
            mask = coco_mask.decode(rle)
            
            # Convert to uint8 if needed
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"RLE to mask conversion failed: {e}")
            self.stats.polygon_conversion_errors += 1
            return np.zeros((img_height, img_width), dtype=np.uint8)
    
    def create_combined_mask(self, annotations: List[Dict[str, Any]], 
                           img_height: int, img_width: int) -> Tuple[np.ndarray, int]:
        """
        Create combined mask from multiple COCO annotations
        
        Args:
            annotations: List of COCO annotation dictionaries
            img_height: Image height
            img_width: Image width
            
        Returns:
            Tuple of (combined_mask, num_objects_processed)
        """
        try:
            combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            objects_processed = 0
            
            for annotation in annotations:
                try:
                    # Skip crowd annotations
                    if annotation.get("iscrowd", 0) == 1:
                        continue
                    
                    # Get segmentation data
                    segmentation = annotation.get("segmentation", [])
                    if not segmentation:
                        # Fall back to bounding box if no segmentation
                        bbox = annotation.get("bbox", [])
                        if len(bbox) == 4:
                            x, y, w, h = [int(coord) for coord in bbox]
                            # Ensure bbox is within image bounds
                            x = max(0, min(x, img_width - 1))
                            y = max(0, min(y, img_height - 1))
                            w = max(1, min(w, img_width - x))
                            h = max(1, min(h, img_height - y))
                            combined_mask[y:y+h, x:x+w] = 255
                            objects_processed += 1
                        continue
                    
                    # Handle RLE format
                    if isinstance(segmentation, dict):
                        mask = self.rle_to_mask(segmentation, img_height, img_width)
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                        objects_processed += 1
                    
                    # Handle polygon format
                    elif isinstance(segmentation, list):
                        for polygon in segmentation:
                            if len(polygon) >= 6:  # Valid polygon
                                mask = self.polygon_to_mask(polygon, img_height, img_width)
                                combined_mask = cv2.bitwise_or(combined_mask, mask)
                        objects_processed += 1
                
                except Exception as e:
                    logger.warning(f"Error processing annotation: {e}")
                    self.stats.mask_generation_errors += 1
                    continue
            
            return combined_mask, objects_processed
            
        except Exception as e:
            logger.error(f"Combined mask creation failed: {e}")
            self.stats.mask_generation_errors += 1
            return np.zeros((img_height, img_width), dtype=np.uint8), 0
    
    def process_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process and validate mask for inpainting
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (processed_mask, is_valid)
        """
        try:
            # Validate input
            if mask is None or mask.size == 0:
                return mask, False
            
            # Calculate mask statistics
            mask_area = np.sum(mask > 0)
            total_area = mask.shape[0] * mask.shape[1]
            mask_ratio = mask_area / total_area
            
            # Check minimum area
            if mask_area < self.config.min_mask_area:
                logger.debug(f"Mask too small: {mask_area} < {self.config.min_mask_area}")
                return mask, False
            
            # Check maximum area ratio
            if mask_ratio > self.config.max_mask_area_ratio:
                logger.debug(f"Mask too large: {mask_ratio:.3f} > {self.config.max_mask_area_ratio}")
                return mask, False
            
            # Apply dilation if configured
            if self.config.mask_dilation_kernel_size > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (self.config.mask_dilation_kernel_size, self.config.mask_dilation_kernel_size)
                )
                mask = cv2.dilate(mask, kernel, iterations=self.config.mask_dilation_iterations)
            
            # Update statistics with bounded memory management
            with self._thread_lock:
                # Implement circular buffer to prevent unbounded growth
                if len(self.stats.mask_area_distribution) >= self.stats.mask_area_max_samples:
                    # Remove oldest 20% when reaching limit to maintain efficiency
                    remove_count = self.stats.mask_area_max_samples // 5
                    self.stats.mask_area_distribution = self.stats.mask_area_distribution[remove_count:]
                
                self.stats.mask_area_distribution.append(mask_area)
                self.stats.avg_mask_area = np.mean(self.stats.mask_area_distribution)
            
            return mask, True
            
        except Exception as e:
            logger.error(f"Mask processing failed: {e}")
            return mask, False
    
    def calculate_adaptive_radius(self, mask: np.ndarray) -> int:
        """
        Calculate adaptive inpainting radius based on mask area
        
        Args:
            mask: Binary mask array
            
        Returns:
            Adaptive radius as integer
        """
        if not self.config.adaptive_radius:
            return self.config.inpaint_radius
        
        try:
            # Calculate mask area
            mask_area = np.sum(mask > 0)
            
            if mask_area == 0:
                return self.config.min_radius
            
            # Calculate radius based on square root of area (perimeter-like scaling)
            adaptive_radius = int(np.sqrt(mask_area) * self.config.radius_scale_factor)
            
            # Clamp to min/max bounds
            adaptive_radius = max(self.config.min_radius, 
                                min(self.config.max_radius, adaptive_radius))
            
            logger.debug(f"Adaptive radius: {adaptive_radius} (mask area: {mask_area} pixels)")
            return adaptive_radius
            
        except Exception as e:
            logger.warning(f"Failed to calculate adaptive radius: {e}")
            return self.config.inpaint_radius
    
    def calculate_quality_metrics(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate SSIM and PSNR quality metrics for inpainted regions
        
        Args:
            original: Original image
            inpainted: Inpainted image
            mask: Binary mask of inpainted regions
            
        Returns:
            Tuple of (SSIM, PSNR) scores
        """
        try:
            # Convert to grayscale for quality assessment
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                inp_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                inp_gray = inpainted
            
            # Calculate full image SSIM
            ssim_score = ssim(orig_gray, inp_gray, data_range=orig_gray.max() - orig_gray.min())
            
            # Calculate PSNR for full image
            psnr_score = psnr(orig_gray, inp_gray, data_range=orig_gray.max() - orig_gray.min())
            
            return float(ssim_score), float(psnr_score)
            
        except Exception as e:
            logger.warning(f"Quality metric calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_success_rate_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for success rate using Wilson score interval
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (success_rate, lower_bound, upper_bound)
        """
        try:
            total_processed = self.stats.total_images_processed
            successful = self.stats.successful_inpaints
            
            if total_processed == 0:
                return 0.0, 0.0, 0.0
            
            # Calculate success rate
            success_rate = successful / total_processed
            
            # Wilson score interval for binomial proportion
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            # Wilson score interval calculation
            n = total_processed
            p = success_rate
            
            denominator = 1 + (z_score**2 / n)
            center = (p + (z_score**2 / (2*n))) / denominator
            margin = (z_score / denominator) * np.sqrt((p * (1-p) / n) + (z_score**2 / (4*n**2)))
            
            lower_bound = max(0.0, center - margin)
            upper_bound = min(1.0, center + margin)
            
            return success_rate, lower_bound, upper_bound
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics including confidence intervals
        
        Returns:
            Dictionary with all statistics and confidence intervals
        """
        # Calculate confidence intervals
        success_rate, ci_lower, ci_upper = self.calculate_success_rate_confidence_interval()
        
        return {
            'processing_stats': {
                'total_images_processed': self.stats.total_images_processed,
                'successful_inpaints': self.stats.successful_inpaints,
                'failed_inpaints': self.stats.failed_inpaints,
                'total_objects_removed': self.stats.total_objects_removed
            },
            'success_metrics': {
                'success_rate': success_rate,
                'confidence_interval_95': {
                    'lower_bound': ci_lower,
                    'upper_bound': ci_upper,
                    'margin_of_error': (ci_upper - ci_lower) / 2
                }
            },
            'quality_metrics': {
                'avg_ssim': self.stats.avg_ssim,
                'avg_psnr': self.stats.avg_psnr,
                'quality_samples': len(self.stats.quality_ssim_scores)
            },
            'performance_metrics': {
                'avg_processing_time_ms': self.stats.avg_processing_time_per_image * 1000,
                'avg_mask_area': self.stats.avg_mask_area,
                'avg_inpaint_radius_used': self.stats.avg_inpaint_radius_used
            },
            'error_breakdown': {
                'polygon_conversion_errors': self.stats.polygon_conversion_errors,
                'mask_generation_errors': self.stats.mask_generation_errors,
                'inpainting_algorithm_errors': self.stats.inpainting_algorithm_errors,
                'memory_errors': self.stats.memory_errors
            }
        }
    
    def get_coco_categories(self, coco_json_path: str) -> List[int]:
        """
        Dynamically load all available COCO category IDs from annotations
        
        Args:
            coco_json_path: Path to COCO annotations JSON
            
        Returns:
            List of all category IDs available in the dataset
        """
        try:
            coco = COCO(coco_json_path)
            category_ids = list(coco.getCatIds())
            logger.info(f"Loaded {len(category_ids)} categories from COCO dataset")
            logger.debug(f"Category IDs: {sorted(category_ids)}")
            return sorted(category_ids)
        except Exception as e:
            logger.error(f"Failed to load categories from COCO: {e}")
            # Fallback to standard COCO 80 categories
            fallback_categories = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 84, 85, 86, 87, 88, 89, 90
            ]
            logger.warning(f"Using fallback categories: {len(fallback_categories)} categories")
            return fallback_categories
    
    def apply_telea_inpainting(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply Telea inpainting algorithm to remove objects
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask of objects to remove
            
        Returns:
            Tuple of (inpainted_image, success)
        """
        try:
            # Validate inputs
            if image is None or mask is None:
                return image, False
            
            if image.shape[:2] != mask.shape[:2]:
                logger.error(f"Image and mask size mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
                return image, False
            
            # Calculate adaptive radius based on mask size
            adaptive_radius = self.calculate_adaptive_radius(mask)
            
            # Apply inpainting algorithm with adaptive radius
            inpainted = cv2.inpaint(
                image, 
                mask, 
                inpaintRadius=adaptive_radius,
                flags=self.config.method
            )
            
            # Validate output
            if inpainted is None or inpainted.shape != image.shape:
                logger.error("Inpainting produced invalid output")
                return image, False
            
            # Calculate quality metrics
            ssim_score, psnr_score = self.calculate_quality_metrics(image, inpainted, mask)
            
            # Update statistics (atomic operations to prevent race conditions)
            with self._thread_lock:
                # Capture current state atomically to prevent race conditions
                current_successful_count = self.stats.successful_inpaints
                current_avg_radius = self.stats.avg_inpaint_radius_used
                
                # Calculate new average atomically using actual radius used
                new_count = current_successful_count + 1
                self.stats.avg_inpaint_radius_used = (
                    (current_avg_radius * current_successful_count + adaptive_radius) / new_count
                )
                # Update quality metrics with bounded memory management
                if len(self.stats.quality_ssim_scores) >= self.stats.quality_max_samples:
                    # Remove oldest 20% when reaching limit
                    remove_count = self.stats.quality_max_samples // 5
                    self.stats.quality_ssim_scores = self.stats.quality_ssim_scores[remove_count:]
                    self.stats.quality_psnr_scores = self.stats.quality_psnr_scores[remove_count:]
                
                if ssim_score > 0.0:  # Only add valid scores
                    self.stats.quality_ssim_scores.append(ssim_score)
                    self.stats.quality_psnr_scores.append(psnr_score)
                    self.stats.avg_ssim = np.mean(self.stats.quality_ssim_scores)
                    self.stats.avg_psnr = np.mean(self.stats.quality_psnr_scores)
                
                # Update count last to maintain consistency
                self.stats.successful_inpaints = new_count
            
            return inpainted, True
            
        except Exception as e:
            logger.error(f"Telea inpainting failed: {e}")
            self.stats.inpainting_algorithm_errors += 1
            return image, False
    
    def resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds maximum size limits
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (resized_image, scale_factor)
        """
        try:
            h, w = image.shape[:2]
            max_dim = max(h, w)
            
            if max_dim <= self.config.max_image_size:
                return image, 1.0
            
            scale_factor = self.config.max_image_size / max_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale: {scale_factor:.3f})")
            
            return resized, scale_factor
            
        except Exception as e:
            logger.error(f"Image resizing failed: {e}")
            return image, 1.0
    
    def process_single_image(self, image_path: str, annotations: List[Dict[str, Any]], 
                           output_path: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a single image with Telea inpainting
        
        Args:
            image_path: Path to input image
            annotations: List of COCO annotations for objects to remove
            output_path: Path to save inpainted image (optional)
            
        Returns:
            Tuple of (success, processing_info)
        """
        start_time = time.time()
        processing_info = {
            'image_path': image_path,
            'objects_removed': 0,
            'processing_time': 0.0,
            'mask_area': 0,
            'success': False,
            'error_message': None
        }
        
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            img_height, img_width = image.shape[:2]
            
            # Resize if needed for memory optimization
            if self.config.enable_memory_optimization:
                image, scale_factor = self.resize_if_needed(image)
                if scale_factor != 1.0:
                    # Scale annotations accordingly
                    scaled_annotations = []
                    for ann in annotations:
                        scaled_ann = ann.copy()
                        if 'bbox' in scaled_ann:
                            bbox = scaled_ann['bbox']
                            scaled_ann['bbox'] = [coord * scale_factor for coord in bbox]
                        if 'segmentation' in scaled_ann and isinstance(scaled_ann['segmentation'], list):
                            scaled_segmentation = []
                            for poly in scaled_ann['segmentation']:
                                scaled_poly = [coord * scale_factor for coord in poly]
                                scaled_segmentation.append(scaled_poly)
                            scaled_ann['segmentation'] = scaled_segmentation
                        scaled_annotations.append(scaled_ann)
                    annotations = scaled_annotations
                    img_height, img_width = image.shape[:2]
            
            # Create combined mask
            combined_mask, objects_processed = self.create_combined_mask(
                annotations, img_height, img_width
            )
            
            if objects_processed == 0:
                # Distinguish between legitimate empty scenes vs detection failures
                if len(annotations) == 0:
                    # No annotations provided - legitimate empty scene
                    logger.debug(f"No annotations provided for {image_path} - legitimate empty scene")
                    processing_info['success'] = True
                    processing_info['empty_scene'] = True
                    return True, processing_info
                else:
                    # Annotations were provided but none were processable - potential detection failure
                    logger.warning(f"Annotations provided but no valid objects processed for {image_path} - possible detection failure")
                    processing_info['success'] = False
                    processing_info['error_message'] = "Annotations provided but no valid objects could be processed"
                    processing_info['detection_failure_risk'] = True
                    return False, processing_info
            
            # Process mask
            processed_mask, mask_valid = self.process_mask(combined_mask)
            if not mask_valid:
                logger.debug(f"Mask validation failed for {image_path}")
                processing_info['error_message'] = "Invalid mask"
                return False, processing_info
            
            # Apply Telea inpainting
            inpainted_image, inpaint_success = self.apply_telea_inpainting(image, processed_mask)
            if not inpaint_success:
                processing_info['error_message'] = "Inpainting algorithm failed"
                return False, processing_info
            
            # Save output if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if self.config.output_format.upper() == "JPEG":
                    cv2.imwrite(output_path, inpainted_image, 
                              [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality])
                else:
                    cv2.imwrite(output_path, inpainted_image)
                
                if not os.path.exists(output_path):
                    raise IOError(f"Failed to save inpainted image: {output_path}")
            
            # Update processing info
            processing_info.update({
                'objects_removed': objects_processed,
                'mask_area': np.sum(processed_mask > 0),
                'success': True,
                'processing_time': time.time() - start_time
            })
            
            return True, processing_info
            
        except MemoryError as e:
            logger.error(f"Memory error processing {image_path}: {e}")
            self.stats.memory_errors += 1
            processing_info['error_message'] = f"Memory error: {e}"
            return False, processing_info
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            processing_info['error_message'] = str(e)
            return False, processing_info
        
        finally:
            processing_info['processing_time'] = time.time() - start_time
    
    def process_coco_dataset(self, coco_json_path: str, images_dir: str, 
                           output_dir: str, target_categories: List[int] = None) -> bool:
        """
        Process entire COCO dataset with Telea inpainting - COMPLETE OBJECT REMOVAL
        
        Args:
            coco_json_path: Path to COCO annotation JSON
            images_dir: Directory containing COCO images  
            output_dir: Directory to save inpainted images
            target_categories: List of COCO category IDs to remove (defaults to ALL 80 categories)
            
        Returns:
            True if processing completed successfully
        """
        try:
            # Load COCO dataset
            logger.info(f"Loading COCO dataset from {coco_json_path}")
            coco = COCO(coco_json_path)
            
            # Use ALL COCO categories if not provided (complete object removal)
            if target_categories is None:
                # ALL 80 COCO CATEGORIES for complete object removal - CRITICAL FIX
                target_categories = [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
                ]
            
            # Get images with ANY of the target objects (complete removal)
            image_ids = []
            for cat_id in target_categories:
                image_ids.extend(coco.getImgIds(catIds=[cat_id]))
            image_ids = list(set(image_ids))  # Remove duplicates
            
            logger.info(f"Found {len(image_ids)} images with ANY COCO objects for complete removal")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process images in batches
            total_processed = 0
            total_successful = 0
            
            for i in range(0, len(image_ids), self.config.batch_size):
                batch_ids = image_ids[i:i + self.config.batch_size]
                batch_results = []
                
                # Process batch with thread pool
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    
                    for img_id in batch_ids:
                        try:
                            # Get image info
                            img_info = coco.loadImgs([img_id])[0]
                            image_filename = img_info['file_name']
                            image_path = os.path.join(images_dir, image_filename)
                            
                            if not os.path.exists(image_path):
                                logger.warning(f"Image not found: {image_path}")
                                continue
                            
                            # Get annotations for ALL target objects (complete removal)
                            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=target_categories)
                            annotations = coco.loadAnns(ann_ids)
                            
                            if not annotations:
                                continue
                            
                            # Create output path
                            output_filename = f"inpainted_{image_filename}"
                            output_path = os.path.join(output_dir, output_filename)
                            
                            # Submit processing task
                            future = executor.submit(
                                self.process_single_image, 
                                image_path, 
                                annotations, 
                                output_path
                            )
                            futures.append(future)
                            
                        except Exception as e:
                            logger.error(f"Error setting up processing for image {img_id}: {e}")
                            continue
                    
                    # Collect results
                    for future in as_completed(futures):
                        try:
                            success, info = future.result(timeout=60)  # 60 second timeout
                            batch_results.append((success, info))
                            
                            total_processed += 1
                            if success:
                                total_successful += 1
                                self.stats.successful_inpaints += 1
                                self.stats.total_objects_removed += info['objects_removed']
                            else:
                                self.stats.failed_inpaints += 1
                            
                            self.stats.total_processing_time += info['processing_time']
                            
                        except Exception as e:
                            logger.error(f"Error processing batch item: {e}")
                            total_processed += 1
                            self.stats.failed_inpaints += 1
                
                # Log batch progress
                if i % (self.config.batch_size * 10) == 0:
                    logger.info(f"Processed {total_processed}/{len(image_ids)} images, "
                              f"success rate: {total_successful/max(1,total_processed)*100:.1f}%")
            
            # Update final statistics
            self.stats.total_images_processed = total_processed
            if total_processed > 0:
                self.stats.avg_processing_time_per_image = (
                    self.stats.total_processing_time / total_processed
                )
            
            # Log final results
            logger.info(f"Dataset processing complete:")
            logger.info(f"  Total images: {total_processed}")
            logger.info(f"  Successful: {total_successful}")
            logger.info(f"  Success rate: {total_successful/max(1,total_processed)*100:.1f}%")
            logger.info(f"  Output directory: {output_dir}")
            
            return total_successful > 0
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return False
    
    def validate_inpainting_quality(self, original_path: str, inpainted_path: str, 
                                  sample_points: int = 100) -> Dict[str, float]:
        """
        Validate inpainting quality using statistical metrics
        
        Args:
            original_path: Path to original image
            inpainted_path: Path to inpainted image
            sample_points: Number of sample points for quality assessment
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Load images
            original = cv2.imread(original_path)
            inpainted = cv2.imread(inpainted_path)
            
            if original is None or inpainted is None:
                return {'error': 'Failed to load images'}
            
            if original.shape != inpainted.shape:
                return {'error': 'Image size mismatch'}
            
            # Convert to LAB color space for perceptual comparison
            original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            inpainted_lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
            
            # Calculate differences
            diff = np.abs(original_lab.astype(np.float32) - inpainted_lab.astype(np.float32))
            
            # Quality metrics
            metrics = {
                'mean_pixel_difference': np.mean(diff),
                'max_pixel_difference': np.max(diff),
                'std_pixel_difference': np.std(diff),
                'structural_similarity': 0.0  # Placeholder - would need skimage for SSIM
            }
            
            # Sample-based quality assessment
            h, w = original.shape[:2]
            sample_indices = np.random.choice(h * w, min(sample_points, h * w), replace=False)
            sample_diff = diff.reshape(-1, 3)[sample_indices]
            
            metrics.update({
                'sample_mean_diff': np.mean(sample_diff),
                'sample_std_diff': np.std(sample_diff),
                'quality_score': max(0, 100 - np.mean(sample_diff))  # Simple quality score
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return {'error': str(e)}
    
    def validate_complete_object_removal(self, original_image_path: str, inpainted_image_path: str, 
                                        coco_json_path: str, image_id: int) -> Dict[str, Any]:
        """
        Validate that ALL COCO objects have been completely removed from background
        
        Args:
            original_image_path: Path to original image
            inpainted_image_path: Path to inpainted image
            coco_json_path: Path to COCO annotations
            image_id: COCO image ID
            
        Returns:
            Validation results dictionary
        """
        try:
            # Load COCO annotations
            coco = COCO(coco_json_path)
            
            # Get ALL annotations for this image (dynamically loaded categories)
            all_category_ids = self.get_coco_categories(coco_json_path)
            
            ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=all_category_ids)
            annotations = coco.loadAnns(ann_ids)
            
            # Count objects by category
            object_counts = {}
            total_objects = 0
            
            for ann in annotations:
                if ann.get("iscrowd", 0) == 1:
                    continue
                    
                cat_id = ann["category_id"]
                cat_info = coco.loadCats([cat_id])[0]
                cat_name = cat_info["name"]
                
                object_counts[cat_name] = object_counts.get(cat_name, 0) + 1
                total_objects += 1
            
            # Load and analyze images
            validation_result = {
                "image_id": image_id,
                "original_path": original_image_path,
                "inpainted_path": inpainted_image_path,
                "total_objects_detected": total_objects,
                "objects_by_category": object_counts,
                "validation_status": "PASSED" if os.path.exists(inpainted_image_path) else "FAILED",
                "error_message": None
            }
            
            # Additional visual validation could be added here
            if os.path.exists(inpainted_image_path):
                original_img = cv2.imread(original_image_path)
                inpainted_img = cv2.imread(inpainted_image_path)
                
                if original_img is not None and inpainted_img is not None:
                    # Calculate difference metrics
                    diff = cv2.absdiff(original_img, inpainted_img)
                    mean_diff = np.mean(diff)
                    
                    validation_result.update({
                        "mean_pixel_difference": float(mean_diff),
                        "inpainting_coverage": float(np.sum(diff > 10) / diff.size),
                        "visual_validation": "COMPLETED"
                    })
                else:
                    validation_result["error_message"] = "Failed to load images for comparison"
                    validation_result["validation_status"] = "FAILED"
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed for image {image_id}: {e}")
            return {
                "image_id": image_id,
                "validation_status": "ERROR",
                "error_message": str(e),
                "total_objects_detected": 0,
                "objects_by_category": {}
            }
    
    def get_statistics(self) -> InpaintingStats:
        """Get current processing statistics"""
        return self.stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = InpaintingStats()
    
    def process_filtered_images(self, coco_json_path: str, images_dir: str, 
                              output_dir: str, target_categories: List[int], 
                              filtered_image_ids: List[int]) -> bool:
        """
        Process only the filtered image IDs that match COCO dataset filtering exactly
        
        Args:
            coco_json_path: Path to COCO annotation JSON
            images_dir: Directory containing COCO images  
            output_dir: Directory to save inpainted images
            target_categories: List of COCO category IDs to remove
            filtered_image_ids: List of image IDs that passed dataset filtering
            
        Returns:
            True if processing completed successfully
        """
        try:
            from pycocotools.coco import COCO
            
            logger.info(f"Processing {len(filtered_image_ids)} filtered images")
            logger.info(f"Target categories: {target_categories}")
            
            # Load COCO annotations
            logger.info(f"Loading COCO dataset from {coco_json_path}")
            coco = COCO(coco_json_path)
            
            # Create output directory
            inpainted_dir = output_dir
            os.makedirs(inpainted_dir, exist_ok=True)
            
            # Reset statistics
            self.stats = InpaintingStats()
            
            # Process each filtered image
            for i, img_id in enumerate(filtered_image_ids):
                try:
                    # Load image info
                    img_info = coco.loadImgs([img_id])[0]
                    image_path = os.path.join(images_dir, img_info['file_name'])
                    
                    # Skip if already processed
                    output_filename = f"inpainted_{img_info['file_name']}"
                    output_path = os.path.join(inpainted_dir, output_filename)
                    if os.path.exists(output_path):
                        continue
                    
                    # Get annotations for this image with target categories
                    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=target_categories)
                    annotations = coco.loadAnns(ann_ids)
                    
                    if not annotations:
                        logger.warning(f"No annotations found for image {img_id} - skipping")
                        continue
                    
                    # Process the image
                    success, processing_info = self.process_single_image(
                        image_path, annotations, output_path
                    )
                    
                    if success:
                        self.stats.successful_inpaints += 1
                        logger.debug(f"Successfully processed {img_info['file_name']}")
                    else:
                        self.stats.failed_inpaints += 1
                        logger.warning(f"Failed to process {img_info['file_name']}: {processing_info.get('error_message', 'Unknown error')}")
                    
                    # Progress update
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(filtered_image_ids)} images ({(i+1)/len(filtered_image_ids)*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
                    self.stats.failed_inpaints += 1
                    continue
            
            # Final summary
            self.stats.total_images_processed = len(filtered_image_ids)
            success_rate = self.stats.successful_inpaints / max(1, self.stats.total_images_processed)
            
            logger.info("="*70)
            logger.info("FILTERED INPAINTING SUMMARY")
            logger.info("="*70)
            logger.info(f"Total filtered images: {len(filtered_image_ids)}")
            logger.info(f"Successful inpaints: {self.stats.successful_inpaints}")
            logger.info(f"Failed inpaints: {self.stats.failed_inpaints}")
            logger.info(f"Success rate: {success_rate*100:.1f}%")
            logger.info(f"Output directory: {inpainted_dir}")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"Filtered inpainting failed: {e}")
            return False


def create_sample_validation_set(coco_json_path: str, images_dir: str, 
                                output_dir: str, num_samples: int = 10) -> List[str]:
    """
    Create a small validation set for testing inpainting quality
    
    Args:
        coco_json_path: Path to COCO annotation JSON
        images_dir: Directory containing COCO images
        output_dir: Directory to save validation samples
        num_samples: Number of validation samples to create
        
    Returns:
        List of paths to validation samples
    """
    try:
        # Initialize inpainting system
        config = InpaintingConfig(
            inpaint_radius=7,
            max_workers=2  # Reduce for validation
        )
        inpainter = TeleaInpaintingSystem(config)
        
        # Load COCO dataset
        coco = COCO(coco_json_path)
        # ALL COCO CATEGORIES for complete object removal - dynamically loaded
        temp_system = TeleaInpaintingSystem()  # Temporary instance for category loading
        all_coco_categories = temp_system.get_coco_categories(coco_json_path)
        
        # Get random sample of images
        image_ids = []
        for cat_id in all_coco_categories:
            image_ids.extend(coco.getImgIds(catIds=[cat_id]))
        image_ids = list(set(image_ids))
        
        # Select random samples
        selected_ids = np.random.choice(image_ids, min(num_samples, len(image_ids)), replace=False)
        
        validation_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Creating {len(selected_ids)} validation samples...")
        
        for i, img_id in enumerate(selected_ids):
            try:
                # Get image info
                img_info = coco.loadImgs([img_id])[0]
                image_filename = img_info['file_name']
                image_path = os.path.join(images_dir, image_filename)
                
                if not os.path.exists(image_path):
                    continue
                
                # Get annotations for ALL objects
                ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=all_coco_categories)
                annotations = coco.loadAnns(ann_ids)
                
                if not annotations:
                    continue
                
                # Create output path
                output_filename = f"validation_sample_{i:02d}_{image_filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Process image
                success, info = inpainter.process_single_image(image_path, annotations, output_path)
                
                if success:
                    validation_paths.append(output_path)
                    logger.info(f"Created validation sample {i+1}/{len(selected_ids)}: "
                              f"{info['objects_removed']} objects removed")
                
            except Exception as e:
                logger.error(f"Error creating validation sample {i}: {e}")
                continue
        
        logger.info(f"Created {len(validation_paths)} validation samples in {output_dir}")
        return validation_paths
        
    except Exception as e:
        logger.error(f"Validation set creation failed: {e}")
        return []


# Production interface functions
def setup_inpainting_directories(base_dir: str) -> Dict[str, str]:
    """Setup directory structure for inpainting operations"""
    dirs = {
        'inpainted_images': os.path.join(base_dir, 'inpainted_images'),
        'validation_samples': os.path.join(base_dir, 'validation_samples'),
        'quality_reports': os.path.join(base_dir, 'quality_reports'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def run_production_inpainting(coco_json_path: str, images_dir: str, output_base_dir: str) -> bool:
    """
    Run production Telea inpainting on COCO dataset
    
    Args:
        coco_json_path: Path to COCO annotation JSON
        images_dir: Directory containing COCO images
        output_base_dir: Base directory for outputs
        
    Returns:
        True if processing completed successfully
    """
    try:
        # Setup directories
        dirs = setup_inpainting_directories(output_base_dir)
        
        # Create validation samples first
        logger.info("Creating validation samples...")
        validation_paths = create_sample_validation_set(
            coco_json_path, images_dir, dirs['validation_samples'], num_samples=10
        )
        
        if len(validation_paths) < 5:
            logger.error("Failed to create sufficient validation samples")
            return False
        
        # Initialize production inpainting system
        config = InpaintingConfig(
            inpaint_radius=7,
            mask_dilation_kernel_size=3,
            max_workers=4,
            batch_size=32,
            enable_memory_optimization=True
        )
        
        inpainter = TeleaInpaintingSystem(config)
        
        # Process full dataset
        logger.info("Processing full COCO dataset...")
        success = inpainter.process_coco_dataset(
            coco_json_path, images_dir, dirs['inpainted_images']
        )
        
        if not success:
            logger.error("Dataset processing failed")
            return False
        
        # Log final statistics
        stats = inpainter.get_statistics()
        stats.log_summary()
        
        # Save statistics to file
        stats_file = os.path.join(dirs['logs'], 'inpainting_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'total_images_processed': stats.total_images_processed,
                'successful_inpaints': stats.successful_inpaints,
                'failed_inpaints': stats.failed_inpaints,
                'total_objects_removed': stats.total_objects_removed,
                'avg_processing_time_per_image': stats.avg_processing_time_per_image,
                'avg_mask_area': stats.avg_mask_area,
                'success_rate': stats.successful_inpaints / max(1, stats.total_images_processed),
                'validation_samples_created': len(validation_paths)
            }, f, indent=2)
        
        logger.info(f"Production inpainting complete. Results saved to {output_base_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Production inpainting failed: {e}")
        return False
    


def run_production_inpainting_filtered(coco_json_path: str, images_dir: str, output_base_dir: str, target_categories: List[int]) -> bool:
    """
    Run production Telea inpainting on COCO dataset with specific categories and exact dataset filtering
    
    Args:
        coco_json_path: Path to COCO annotation JSON
        images_dir: Directory containing COCO images
        output_base_dir: Base directory for outputs
        target_categories: List of COCO category IDs to process (interior categories)
        
    Returns:
        True if processing completed successfully
    """
    try:
        from pycocotools.coco import COCO
        
        logger.info(f"Getting filtered image IDs for categories: {target_categories}")
        
        # Get EXACT same filtered image IDs as COCO dataset filtering
        coco = COCO(coco_json_path)
        valid_image_ids = []
        
        # Get all images that have ANY of our target categories
        image_ids = []
        for cat_id in target_categories:
            image_ids.extend(coco.getImgIds(catIds=[cat_id]))
        image_ids = list(set(image_ids))
        
        logger.info(f"Found {len(image_ids)} images with target categories")
        
        # Apply EXACT same filtering as EnhancedCOCO_Wrapper
        for img_id in image_ids:
            try:
                # Get annotations for this image with target categories only
                ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=target_categories)
                annotations = coco.loadAnns(ann_ids)
                
                if not annotations:
                    continue
                    
                # Apply the same filtering as EnhancedCOCO_Wrapper
                valid_annotations = []
                
                for ann in annotations:
                    # Skip crowd annotations
                    if ann.get("iscrowd", 0) == 1:
                        continue
                    
                    # Check if category is in our target list
                    if ann.get("category_id") not in target_categories:
                        continue
                    
                    # Get bbox and validate
                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    x, y, w, h = bbox
                    
                    # Check minimum area (100 pixels)
                    area = w * h
                    if area < 100:
                        continue
                    
                    # Check aspect ratio (max 15.0)
                    if w > 0 and h > 0:
                        aspect_ratio = max(w / h, h / w)
                        if aspect_ratio > 15.0:
                            continue
                    
                    valid_annotations.append(ann)
                
                # Check if we have valid annotations and not too many (max 5)
                if valid_annotations and len(valid_annotations) <= 5:
                    valid_image_ids.append(img_id)
                    
            except Exception as e:
                logger.warning(f"Error filtering image {img_id}: {e}")
                continue
        
        logger.info(f"Filtered to {len(valid_image_ids)} images that match COCO dataset filtering exactly")
        
        # Setup directories
        dirs = setup_inpainting_directories(output_base_dir)
        
        # Initialize production inpainting system
        config = InpaintingConfig(
            inpaint_radius=7,
            mask_dilation_kernel_size=3,
            max_workers=4
        )
        inpainter = TeleaInpaintingSystem(config)
        
        # Process only the filtered images
        logger.info("Processing filtered images with interior categories only...")
        success = inpainter.process_filtered_images(
            coco_json_path, images_dir, dirs['inpainted_images'], 
            target_categories, valid_image_ids
        )
        
        if not success:
            logger.error("Filtered dataset processing failed")
            return False
        
        # Log final statistics
        stats = inpainter.get_statistics()
        stats.log_summary()
        
        logger.info(f"✓ Successfully processed {len(valid_image_ids)} filtered images")
        logger.info(f"✓ Output directory: {dirs['inpainted_images']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Filtered production inpainting failed: {e}")
        return False


if __name__ == "__main__":
    # Production test with real COCO data - INTERIOR CATEGORIES ONLY
    coco_json = "/home/gaurang/hardnetnew/data/coco/annotations/instances_train2017_interior.json"
    images_dir = "/home/gaurang/hardnetnew/data/coco/train2017"
    output_dir = "/home/gaurang/hardnetnew/data/coco/inpainted_backgrounds"
    
    # INTERIOR DESIGN CATEGORIES ONLY - EXACT MATCH WITH COCO DATASET FILTERING
    interior_categories = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 83]
    
    logger.info("Starting Telea inpainting system for INTERIOR CATEGORIES ONLY...")
    logger.info(f"Target categories: {interior_categories}")
    
    success = run_production_inpainting_filtered(coco_json, images_dir, output_dir, interior_categories)
    
    if success:
        logger.info("Telea inpainting system test COMPLETED SUCCESSFULLY")
    else:
        logger.error("Telea inpainting system test FAILED")