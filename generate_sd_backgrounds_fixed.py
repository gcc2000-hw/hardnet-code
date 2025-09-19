#!/usr/bin/env python3
"""
Generate 10,000 Stable Diffusion backgrounds for SPRING evaluation
As per the paper instructions:
- Prompts: "A clean, empty, living room." and "A clean, empty, kitchen."
- Parameters: num_inference_steps=80, guidance_scale=15
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime

def setup_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """Setup Stable Diffusion pipeline with specified parameters"""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xformers memory efficient attention")
    except:
        print("xformers not available, using default attention")
    
    return pipe

def generate_backgrounds(
    pipe,
    output_dir,
    num_images=10000,
    num_inference_steps=80,
    guidance_scale=15,
    width=512,
    height=512,
    seed=42,
    resume=True
):
    """Generate backgrounds with alternating prompts"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Two prompts as specified in the paper
    prompts = [
        "A clean, empty, living room.",
        "A clean, empty, kitchen."
    ]
    
    # Set random seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Load existing metadata if resuming
    metadata_path = output_path / "generation_metadata.json"
    if resume and metadata_path.exists():
        print("Loading existing metadata for resume...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        existing_count = len(metadata["images"])
        print(f"Found {existing_count} existing images, resuming from there")
    else:
        metadata = {
            "num_images": num_images,
            "prompts": prompts,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "generation_date": datetime.now().isoformat(),
            "images": []
        }
        existing_count = 0
    
    # Progress bar
    pbar = tqdm(total=num_images, initial=existing_count, desc="Generating backgrounds")
    
    images_generated = existing_count
    prompt_idx = existing_count  # Start from where we left off
    
    while images_generated < num_images:
        # Alternate between prompts
        current_prompt = prompts[prompt_idx % len(prompts)]
        room_type = "livingroom" if prompt_idx % 2 == 0 else "kitchen"
        
        # Generate filename with room type for clarity
        filename = f"background_{images_generated:05d}_{room_type}.jpg"
        filepath = output_path / filename
        
        # Skip if already exists (resume capability)
        if resume and filepath.exists():
            print(f"\nSkipping existing {filename}")
            images_generated += 1
            prompt_idx += 1
            pbar.update(1)
            continue
        
        try:
            # Generate image
            with torch.autocast("cuda"):
                # Use a unique seed for each image for diversity
                image_generator = torch.Generator(device="cuda").manual_seed(seed + images_generated)
                result = pipe(
                    prompt=current_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=image_generator
                )
            
            # Save image as JPEG (standard for evaluation)
            image = result.images[0]
            image.save(filepath, "JPEG", quality=95)
            
            # Verify image was saved correctly
            try:
                test_img = Image.open(filepath)
                test_img.verify()
            except Exception as e:
                print(f"\nWarning: Image verification failed for {filename}: {e}")
            
            # Update metadata
            metadata["images"].append({
                "filename": filename,
                "prompt": current_prompt,
                "room_type": room_type,
                "prompt_idx": prompt_idx % len(prompts),
                "image_idx": images_generated,
                "image_seed": seed + images_generated
            })
            
            images_generated += 1
            prompt_idx += 1
            pbar.update(1)
            
            # Save metadata periodically (every 100 images)
            if images_generated % 100 == 0:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"\nSaved metadata at {images_generated} images")
                
                # Clear CUDA cache periodically
                torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user, saving progress...")
            break
        except Exception as e:
            print(f"\nError generating image {images_generated}: {e}")
            continue
    
    pbar.close()
    
    # Save final metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete! Generated {images_generated} backgrounds")
    print(f"Metadata saved to {metadata_path}")
    
    # Generate summary
    living_room_count = sum(1 for img in metadata["images"] if img["room_type"] == "livingroom")
    kitchen_count = sum(1 for img in metadata["images"] if img["room_type"] == "kitchen")
    
    print("\nSummary:")
    print(f"  Living room backgrounds: {living_room_count}")
    print(f"  Kitchen backgrounds: {kitchen_count}")
    print(f"  Total: {images_generated}")
    
    # Verify all files exist
    print("\nVerifying generated files...")
    missing_files = []
    for img_info in metadata["images"]:
        if not (output_path / img_info["filename"]).exists():
            missing_files.append(img_info["filename"])
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files are missing!")
    else:
        print("All files verified successfully!")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Generate Stable Diffusion backgrounds for SPRING")
    parser.add_argument("--output_dir", type=str, default="data/test_backgrounds",
                        help="Output directory for generated backgrounds")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of backgrounds to generate")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from existing images")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    print("Setting up Stable Diffusion pipeline...")
    pipe = setup_pipeline(args.model_id, args.device)
    
    print(f"\nGenerating {args.num_images} backgrounds to {args.output_dir}")
    print("Parameters from SPRING paper:")
    print(f"  - num_inference_steps: 80")
    print(f"  - guidance_scale: 15")
    print(f"  - Prompts: alternating living room and kitchen")
    print(f"  - Resume from existing: {not args.no_resume}")
    
    # Generate backgrounds with paper-specified parameters
    metadata = generate_backgrounds(
        pipe=pipe,
        output_dir=args.output_dir,
        num_images=args.num_images,
        num_inference_steps=80,  # As specified in paper
        guidance_scale=15,       # As specified in paper
        width=args.width,
        height=args.height,
        seed=args.seed,
        resume=not args.no_resume
    )
    
    print("\nBackground generation complete!")
    print(f"Ready for SPRING evaluation in: {args.output_dir}")

if __name__ == "__main__":
    main()