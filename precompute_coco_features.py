#!/usr/bin/env python3
"""
Pre-compute COCO val2017 reference features for FID calculation.
Run this ONCE to generate reference_features.pt file.
"""

import torch
import glob
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import os

def precompute_coco_features():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Inception model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    print("Inception model loaded")
    
    # Preprocessing pipeline (same as evaluation)
    fair_comparison_preprocess = transforms.Compose([
        transforms.Resize((1000, 1000)),  # Paper requirement
        transforms.ToTensor()
    ])
    
    inception_preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get COCO images
    coco_dir = "/home/gaurang/hardnet/data/val2017"
    image_files = glob.glob(os.path.join(coco_dir, "*.jpg"))[:500]  # Use 500 for good statistics
    
    if not image_files:
        raise ValueError(f"No COCO images found in {coco_dir}")
    
    print(f"Processing {len(image_files)} COCO images...")
    
    reference_features = []
    
    for i, image_path in enumerate(image_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(image_files)} images")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Step 1: Standardize to 1000x1000
            standardized_tensor = fair_comparison_preprocess(image)
            standardized_pil = transforms.ToPILImage()(standardized_tensor)
            
            # Step 2: Preprocess for Inception
            image_tensor = inception_preprocess(standardized_pil).unsqueeze(0).to(device)
            
            # Step 3: Extract features using avgpool layer (before final fc)
            with torch.no_grad():
                # Use Inception's built-in feature extraction
                # Get features from the avgpool layer (2048-dim)
                features = []
                
                def hook_fn(module, input, output):
                    features.append(output)
                
                # Register hook on avgpool layer
                handle = inception_model.avgpool.register_forward_hook(hook_fn)
                
                # Forward pass - features will be captured by hook
                _ = inception_model(image_tensor)
                
                # Remove hook
                handle.remove()
                
                # Get the captured features
                if features:
                    feature_tensor = features[0].view(features[0].size(0), -1)  # Flatten
                else:
                    raise RuntimeError("No features captured by hook")
                
                reference_features.append(feature_tensor.cpu())
                
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}")
            continue
    
    # Stack all features
    reference_features = torch.cat(reference_features, dim=0)
    print(f"Final reference features shape: {reference_features.shape}")
    
    # Save to disk
    output_file = "/home/gaurang/hardnet/coco_reference_features.pt"
    torch.save(reference_features, output_file)
    print(f"Saved reference features to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    precompute_coco_features()