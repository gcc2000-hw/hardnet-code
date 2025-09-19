#!/usr/bin/env python3
"""
Proper evaluation metrics for SPRING system:
- Position Accuracy: How accurately objects are placed according to constraints
- Object Accuracy: How accurately objects are detected/recognized
- IS (Inception Score): Image quality and diversity
- FID (Fréchet Inception Distance): Distribution similarity to reference images

Reference for these metrics should be real kitchen/living room images from COCO dataset.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import torchvision.models as models
import torch.nn.functional as F
from scipy.linalg import sqrtm

class ComprehensiveEvaluationMetrics:
    """Complete evaluation metrics for spatial layout generation."""
    
    def __init__(self, device='cuda', enable_clip=False):
        self.device = device
        self.enable_clip = enable_clip
        print(f"Initializing comprehensive metrics on device: {device}")

        # Load CLIP for semantic evaluation (only if enabled)
        if enable_clip:
            print("Loading CLIP model for semantic evaluation...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            print("CLIP disabled - skipping CLIP model loading")
            self.clip_model = None
            self.clip_processor = None
        
        # Load Inception for IS and FID
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        
        # Image preprocessing for Inception
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("All models loaded successfully")
    
    def compute_position_accuracy(self, generated_layout: np.ndarray, constraints: List[Any], 
                                object_names: List[str], tolerance: float = 0.02) -> float:
        """
        Compute position accuracy: how well the generated layout satisfies spatial constraints.
        
        Args:
            generated_layout: [N, 4] array of [x, y, w, h] coordinates in [0,1]
            constraints: List of constraint objects
            object_names: List of object names corresponding to layout
            tolerance: Tolerance for constraint satisfaction
            
        Returns:
            Position accuracy as percentage (0-1)
        """
        if not constraints:
            return 1.0  # Perfect if no constraints to satisfy
        
        satisfied_constraints = 0
        total_constraints = len(constraints)
        
        # Create object name to index mapping
        obj_name_to_idx = {name: idx for idx, name in enumerate(object_names)}
        
        for constraint in constraints:
            try:
                satisfied = self._check_constraint_satisfaction(
                    constraint, generated_layout, obj_name_to_idx, tolerance
                )
                if satisfied:
                    satisfied_constraints += 1
            except Exception as e:
                print(f"Warning: Could not check constraint {constraint}: {e}")
                continue
        
        return satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
    
    def _check_constraint_satisfaction(self, constraint, layout: np.ndarray, 
                                     obj_mapping: Dict[str, int], tolerance: float) -> bool:
        """Check if a specific constraint is satisfied."""
        
        # This would need to be implemented based on your constraint types
        # For now, return a placeholder that checks basic spatial relationships
        
        if hasattr(constraint, 'type'):
            if constraint.type == 't1':  # Single object constraint
                obj_idx = constraint.o1
                if obj_idx >= len(layout):
                    return False
                
                obj_pos = layout[obj_idx]
                if constraint.v1 == 0:  # x coordinate
                    val = obj_pos[0]
                elif constraint.v1 == 1:  # y coordinate  
                    val = obj_pos[1]
                else:
                    return False
                
                target_val = constraint.val + constraint.offset
                
                if constraint.c == "gt":
                    return val > target_val - tolerance
                elif constraint.c == "lt":
                    return val < target_val + tolerance
                elif constraint.c == "eq":
                    return abs(val - target_val) < tolerance
                    
            elif constraint.type == 't2':  # Two object constraint
                obj1_idx, obj2_idx = constraint.o1, constraint.o2
                if obj1_idx >= len(layout) or obj2_idx >= len(layout):
                    return False
                
                obj1_pos, obj2_pos = layout[obj1_idx], layout[obj2_idx]
                
                if constraint.v1 == 0:
                    val1 = obj1_pos[0]
                elif constraint.v1 == 1:
                    val1 = obj1_pos[1]
                else:
                    return False
                    
                if constraint.v2 == 0:
                    val2 = obj2_pos[0]
                elif constraint.v2 == 1:
                    val2 = obj2_pos[1]
                else:
                    return False
                
                if constraint.c == "gt":
                    return val1 > val2 + constraint.offset - tolerance
                elif constraint.c == "lt":
                    return val1 < val2 + constraint.offset + tolerance
                elif constraint.c == "eq":
                    return abs(val1 - val2 - constraint.offset) < tolerance
        
        return False  # Unknown constraint type
    
    def compute_object_accuracy(self, generated_objects: List[str], 
                              expected_objects: List[str]) -> float:
        """
        Compute object accuracy: how well the generated objects match expected objects.
        
        Args:
            generated_objects: List of object names that were generated/detected
            expected_objects: List of object names that should be present
            
        Returns:
            Object accuracy as percentage (0-1)
        """
        if not expected_objects:
            return 1.0
        
        # Convert to sets for comparison
        gen_set = set(generated_objects)
        exp_set = set(expected_objects)
        
        # Calculate precision and recall
        true_positives = len(gen_set & exp_set)  # Correctly generated objects
        false_positives = len(gen_set - exp_set)  # Extra objects generated
        false_negatives = len(exp_set - gen_set)  # Missing objects
        
        # Object accuracy = F1 score (harmonic mean of precision and recall)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def compute_inception_score(self, images: List[Image.Image]) -> float:
        """Compute Inception Score for a list of images."""
        if len(images) < 2:
            raise ValueError("IS requires at least 2 images")
        
        # Extract features
        features = []
        for img in images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.inception_model(img_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = F.softmax(logits, dim=1)
                features.append(probs.cpu())
        
        # Stack all probability distributions
        probs = torch.cat(features, dim=0)
        
        # Compute marginal distribution p(y) = E[p(y|x)]
        p_y = torch.mean(probs, dim=0, keepdim=True)
        
        # Compute KL divergence for each sample
        kl_divs = []
        for i in range(probs.shape[0]):
            p_yx = probs[i:i+1]
            kl_div = torch.sum(p_yx * (torch.log(p_yx + 1e-10) - torch.log(p_y + 1e-10)))
            kl_divs.append(kl_div)
        
        # IS = exp(mean KL divergence)
        mean_kl = torch.mean(torch.stack(kl_divs))
        is_score = torch.exp(mean_kl).item()
        
        return float(is_score)
    
    def compute_fid_score(self, generated_images: List[Image.Image], 
                         reference_images: List[Image.Image]) -> float:
        """Compute FID score between generated and reference images."""
        if len(generated_images) < 2 or len(reference_images) < 2:
            raise ValueError("FID requires at least 2 images in each set")
        
        # Extract features for generated images
        gen_features = []
        for img in generated_images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Get features from pool layer (before final classification)
                features = self.inception_model.Mixed_7c(
                    self.inception_model.Mixed_7b(
                        self.inception_model.Mixed_7a(
                            self.inception_model.Mixed_6e(
                                self.inception_model.Mixed_6d(
                                    self.inception_model.Mixed_6c(
                                        self.inception_model.Mixed_6b(
                                            self.inception_model.Mixed_6a(
                                                self.inception_model.Mixed_5d(
                                                    self.inception_model.Mixed_5c(
                                                        self.inception_model.Mixed_5b(
                                                            self.inception_model.maxpool2(
                                                                self.inception_model.Conv2d_4a_3x3(
                                                                    self.inception_model.maxpool1(
                                                                        self.inception_model.Conv2d_3b_1x1(
                                                                            self.inception_model.Conv2d_2b_3x3(
                                                                                self.inception_model.Conv2d_2a_3x3(
                                                                                    self.inception_model.Conv2d_1a_3x3(img_tensor)
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                gen_features.append(features.cpu())
        
        # Extract features for reference images  
        ref_features = []
        for img in reference_images:
            # Same process as above - this is repetitive but clear
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.inception_model.Mixed_7c(
                    self.inception_model.Mixed_7b(
                        self.inception_model.Mixed_7a(
                            self.inception_model.Mixed_6e(
                                self.inception_model.Mixed_6d(
                                    self.inception_model.Mixed_6c(
                                        self.inception_model.Mixed_6b(
                                            self.inception_model.Mixed_6a(
                                                self.inception_model.Mixed_5d(
                                                    self.inception_model.Mixed_5c(
                                                        self.inception_model.Mixed_5b(
                                                            self.inception_model.maxpool2(
                                                                self.inception_model.Conv2d_4a_3x3(
                                                                    self.inception_model.maxpool1(
                                                                        self.inception_model.Conv2d_3b_1x1(
                                                                            self.inception_model.Conv2d_2b_3x3(
                                                                                self.inception_model.Conv2d_2a_3x3(
                                                                                    self.inception_model.Conv2d_1a_3x3(img_tensor)
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                ref_features.append(features.cpu())
        
        # Convert to numpy
        gen_features = torch.cat(gen_features, dim=0).numpy()
        ref_features = torch.cat(ref_features, dim=0).numpy()
        
        # Compute statistics
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        mu_ref = np.mean(ref_features, axis=0)
        sigma_ref = np.cov(ref_features, rowvar=False)
        
        # Compute FID
        diff = mu_gen - mu_ref
        covmean = sqrtm(sigma_gen.dot(sigma_ref))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_score = diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2 * np.trace(covmean)
        
        return float(fid_score)
    
    def get_inception_features(self, image: Image.Image) -> torch.Tensor:
        """Get Inception features for a single image (for caching)."""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.inception_model(img_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            return probs.cpu()
    
    def create_text_description(self, specification: Dict[str, Any]) -> str:
        """Create text description from specification for CLIP scoring."""
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
    
    def compute_clip_score(self, image: Image.Image, text_description: str) -> float:
        """Compute CLIP score between image and text description."""
        if not self.enable_clip:
            print("CLIP disabled - returning dummy score of 0.0")
            return 0.0

        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP model not loaded")

        # Process inputs
        inputs = self.clip_processor(text=[text_description], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image

            # CLIP logits are temperature-scaled cosine similarities
            # Use proper temperature scaling - CLIP uses temperature around 100 by default
            temperature = 100.0  # Match CLIP's internal temperature scaling
            clip_score = torch.sigmoid(torch.tensor(logits_per_image.cpu().item() / temperature)).item()

        return float(clip_score)

def load_coco_reference_images(data_dir: str, room_types: List[str] = ['kitchen'], 
                             max_images: int = 1000) -> List[Image.Image]:
    """
    Load reference images from COCO dataset for FID calculation.
    
    For FID reference data, we need real images from the same domain:
    - Kitchen scenes from COCO dataset 
    - Living room scenes from COCO dataset
    - These represent the "real" distribution we want to match
    """
    reference_images = []
    
    # Look for COCO images in the data directory
    data_path = Path(data_dir)
    
    # Check common COCO image locations
    possible_paths = [
        data_path / "coco" / "images",
        data_path / "val2017",
        data_path / "train2017", 
        data_path / "images",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found COCO images at: {path}")
            
            # Load images (first max_images found)
            image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            
            for img_file in image_files[:max_images]:
                try:
                    img = Image.open(img_file).convert('RGB')
                    reference_images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {img_file}: {e}")
                    continue
            
            if len(reference_images) >= max_images:
                break
    
    if not reference_images:
        print("WARNING: No COCO reference images found!")
        print(f"Searched in: {[str(p) for p in possible_paths]}")
        print("FID calculation will not be possible without reference data.")
    else:
        print(f"Loaded {len(reference_images)} reference images for FID calculation")
    
    return reference_images

if __name__ == "__main__":
    # Test the metrics
    print("Testing comprehensive evaluation metrics...")
    
    # Initialize
    metrics = ComprehensiveEvaluationMetrics()
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    
    # Generate some test images
    test_images = []
    for i in range(10):
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img_array))
    
    # Test IS
    try:
        is_score = metrics.compute_inception_score(test_images)
        print(f"SUCCESS: Inception Score: {is_score:.3f}")
    except Exception as e:
        print(f"ERROR: IS failed: {e}")
    
    # Test FID (using same images as both gen and ref for testing)
    try:
        fid_score = metrics.compute_fid_score(test_images[:5], test_images[5:])
        print(f"SUCCESS: FID Score: {fid_score:.3f}")
    except Exception as e:
        print(f"ERROR: FID failed: {e}")
    
    # Test reference image loading
    print(f"\nLooking for COCO reference images...")
    ref_images = load_coco_reference_images("/home/gaurang/hardnet/data")
    print(f"Found {len(ref_images)} reference images")