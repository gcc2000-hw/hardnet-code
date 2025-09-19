"""
SPRING Telea Background Generator
Creates clean background images using OpenCV Telea inpainting by removing furniture objects.

Following the exact methodology described by SPRING authors:
"For Telea inpainting, you can just use the CV2 implementation. We decided to focus on 
fast, old-school inpainting for this dataset object elimination part."
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
# from tqdm import tqdm  # DISABLED: No progress bars for cleaner output
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# SPRING interior furniture categories (from paper)
SPRING_CATEGORIES = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 83]
CATEGORY_NAMES = [
    "chair", "couch", "potted plant", "bed", "mirror",
    "dining table", "window", "desk", "toilet", "door", 
    "tv", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "blender"
]


class TeleaBackgroundGenerator:
    """
    Generates clean background images using Telea inpainting.
    Removes furniture objects from COCO images following SPRING methodology.
    """
    
    def __init__(self, 
                 coco_root: str,
                 coco_annot: str,
                 output_dir: str,
                 categories: List[int] = None,
                 inpaint_radius: int = 3,
                 mask_dilation: int = 5):
        """
        Initialize Telea background generator.
        
        Args:
            coco_root: Path to COCO images directory
            coco_annot: Path to COCO annotations JSON
            output_dir: Output directory for clean backgrounds
            categories: Object categories to remove (default: SPRING furniture)
            inpaint_radius: Telea inpainting radius
            mask_dilation: Mask dilation size (to ensure clean removal)
        """
        self.coco_root = Path(coco_root)
        self.coco_annot = coco_annot
        self.output_dir = Path(output_dir)
        self.categories = categories or SPRING_CATEGORIES
        self.inpaint_radius = inpaint_radius
        self.mask_dilation = mask_dilation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load COCO annotations
        self.coco_data = self._load_coco_annotations()
        
        # Filter for interior scenes
        self.interior_scenes = self._filter_interior_scenes()
        
        self.logger.info(f"Telea Background Generator initialized")
        self.logger.info(f"Found {len(self.interior_scenes)} interior scenes to process")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for background generation."""
        logger = logging.getLogger('TeleaBackgroundGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_coco_annotations(self) -> Dict:
        """Load COCO annotations JSON."""
        self.logger.info(f"Loading COCO annotations from {self.coco_annot}")
        
        with open(self.coco_annot, 'r') as f:
            coco_data = json.load(f)
        
        self.logger.info(f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        return coco_data
    
    def _filter_interior_scenes(self) -> List[Dict]:
        """Filter COCO scenes that contain furniture objects."""
        # Create image ID to annotations mapping
        image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Filter images that contain furniture
        interior_scenes = []
        for image_info in self.coco_data['images']:
            image_id = image_info['id']
            annotations = image_annotations.get(image_id, [])
            
            # Check if image contains furniture objects
            furniture_objects = []
            for ann in annotations:
                if (ann['category_id'] in self.categories and 
                    ann.get('iscrowd', 0) == 0):  # Exclude crowd annotations
                    furniture_objects.append(ann)
            
            # Include if has 1-5 furniture objects (SPRING filtering)
            if 1 <= len(furniture_objects) <= 5:
                interior_scenes.append({
                    'image_info': image_info,
                    'furniture_objects': furniture_objects
                })
        
        return interior_scenes
    
    def _create_object_mask(self, image_shape: Tuple[int, int], objects: List[Dict]) -> np.ndarray:
        """
        Create binary mask for objects to be inpainted.
        
        Args:
            image_shape: (height, width) of the image
            objects: List of object annotations
            
        Returns:
            Binary mask where 255 = inpaint, 0 = keep
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create PIL Image for drawing
        mask_pil = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_pil)
        
        for obj in objects:
            bbox = obj['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Draw rectangle for bounding box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Draw filled rectangle
            draw.rectangle([x1, y1, x2, y2], fill=255)
        
        # Convert back to numpy
        mask = np.array(mask_pil)
        
        # Dilate mask to ensure clean removal (avoid edge artifacts)
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply Telea inpainting to remove masked objects.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask (255 = inpaint, 0 = keep)
            
        Returns:
            Inpainted image
        """
        # Apply Telea inpainting algorithm
        inpainted = cv2.inpaint(
            image, 
            mask, 
            inpaintRadius=self.inpaint_radius, 
            flags=cv2.INPAINT_TELEA
        )
        
        return inpainted
    
    def process_single_image(self, scene_data: Dict) -> bool:
        """
        Process a single image to create clean background.
        
        Args:
            scene_data: Dictionary containing image_info and furniture_objects
            
        Returns:
            True if successful, False otherwise
        """
        image_info = scene_data['image_info']
        furniture_objects = scene_data['furniture_objects']
        
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Input and output paths
        input_path = self.coco_root / filename
        output_filename = f"{image_id:012d}.jpg"
        output_path = self.output_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            return True
        
        try:
            # Load image
            if not input_path.exists():
                self.logger.warning(f"Image not found: {input_path}")
                return False
            
            # Load with OpenCV (BGR format)
            image = cv2.imread(str(input_path))
            if image is None:
                self.logger.warning(f"Failed to load image: {input_path}")
                return False
            
            height, width = image.shape[:2]
            
            # Create mask for furniture objects
            mask = self._create_object_mask((height, width), furniture_objects)
            
            # Apply Telea inpainting
            inpainted_image = self._inpaint_image(image, mask)
            
            # Save result
            success = cv2.imwrite(str(output_path), inpainted_image)
            if not success:
                self.logger.error(f"Failed to save: {output_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {filename}: {e}")
            return False
    
    def generate_backgrounds(self, 
                           max_workers: int = 4, 
                           max_images: int = None) -> int:
        """
        Generate clean backgrounds for all interior scenes.
        
        Args:
            max_workers: Number of parallel workers
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            Number of successfully processed images
        """
        scenes_to_process = self.interior_scenes
        if max_images:
            scenes_to_process = scenes_to_process[:max_images]
        
        self.logger.info(f"Starting Telea background generation for {len(scenes_to_process)} images")
        self.logger.info(f"Using {max_workers} parallel workers")
        self.logger.info(f"Inpaint radius: {self.inpaint_radius}, Mask dilation: {self.mask_dilation}")
        
        successful_count = 0
        
        if max_workers == 1:
            # Single-threaded processing
            for i, scene_data in enumerate(scenes_to_process):
                if i % 50 == 0:  # Simple progress logging every 50 items
                    print(f"Processing background {i+1}/{len(scenes_to_process)}")
                if self.process_single_image(scene_data):
                    successful_count += 1
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_scene = {
                    executor.submit(self.process_single_image, scene_data): scene_data 
                    for scene_data in scenes_to_process
                }
                
                # Process completed tasks
                processed = 0
                for future in as_completed(future_to_scene):
                    processed += 1
                    if processed % 50 == 0:  # Simple progress logging
                        print(f"Processed {processed}/{len(scenes_to_process)} backgrounds")
                    scene_data = future_to_scene[future]
                    try:
                        if future.result():
                            successful_count += 1
                    except Exception as e:
                        image_id = scene_data['image_info']['id']
                        self.logger.error(f"Task failed for image {image_id}: {e}")
        
        self.logger.info(f"Background generation complete!")
        self.logger.info(f"Successfully processed: {successful_count}/{len(scenes_to_process)} images")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        return successful_count
    
    def create_sample_comparison(self, num_samples: int = 5) -> None:
        """Create side-by-side comparison images for quality checking."""
        comparison_dir = self.output_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating {num_samples} comparison samples")
        
        for i, scene_data in enumerate(self.interior_scenes[:num_samples]):
            image_info = scene_data['image_info']
            furniture_objects = scene_data['furniture_objects']
            
            image_id = image_info['id']
            filename = image_info['file_name']
            
            # Load original image
            original_path = self.coco_root / filename
            original = cv2.imread(str(original_path))
            
            # Load background
            background_path = self.output_dir / f"{image_id:012d}.jpg"
            if not background_path.exists():
                continue
            background = cv2.imread(str(background_path))
            
            # Create mask visualization
            height, width = original.shape[:2]
            mask = self._create_object_mask((height, width), furniture_objects)
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # Combine images side by side
            comparison = np.hstack([original, mask_colored, background])
            
            # Save comparison
            comparison_path = comparison_dir / f"comparison_{i+1:02d}_{image_id:012d}.jpg"
            cv2.imwrite(str(comparison_path), comparison)
        
        self.logger.info(f"Comparison images saved to: {comparison_dir}")


def main():
    """Main function for background generation."""
    parser = argparse.ArgumentParser(description="Generate SPRING Telea backgrounds")
    parser.add_argument("--coco_root", required=True, help="Path to COCO images directory")
    parser.add_argument("--coco_annot", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for backgrounds")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--inpaint_radius", type=int, default=3, help="Telea inpainting radius")
    parser.add_argument("--mask_dilation", type=int, default=5, help="Mask dilation size")
    parser.add_argument("--create_comparisons", action="store_true", help="Create comparison images")
    
    args = parser.parse_args()
    
    # Create generator
    generator = TeleaBackgroundGenerator(
        coco_root=args.coco_root,
        coco_annot=args.coco_annot,
        output_dir=args.output_dir,
        inpaint_radius=args.inpaint_radius,
        mask_dilation=args.mask_dilation
    )
    
    # Generate backgrounds
    successful = generator.generate_backgrounds(
        max_workers=args.max_workers,
        max_images=args.max_images
    )
    
    # Create comparison samples
    if args.create_comparisons:
        generator.create_sample_comparison()
    
    print(f"\n Telea background generation complete!")
    print(f" Generated {successful} background images")
    print(f" Saved to: {args.output_dir}")
    print(f"\n Next step: Update training script to use these backgrounds")


if __name__ == "__main__":
    main()