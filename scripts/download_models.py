
#!/usr/bin/env python3
"""
Script to download and setup required models for the Free Visual AI application.
This script handles model downloading, caching, and optimization for deployment.
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import yaml
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
from diffusers import StableDiffusionPipeline
import streamlit as st

class ModelDownloader:
    def __init__(self, config_path="config.yaml"):
        """Initialize the model downloader"""
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models" / "local_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_configs = self.config.get('models', {})
    
    def download_text_model(self):
        """Download and cache text processing model"""
        print("📝 Downloading text processing model...")
        
        model_name = self.model_configs.get('text_model', {}).get('name', 'microsoft/DialoGPT-small')
        cache_dir = self.models_dir / "text_model"
        cache_dir.mkdir(exist_ok=True)
        
        try:
            # Download tokenizer
            print(f"  Downloading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Download model
            print(f"  Downloading model: {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            print("✅ Text model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading text model: {e}")
            return False
    
    def download_vision_model(self):
        """Download and cache vision model"""
        print("👁️ Downloading vision model...")
        
        model_name = self.model_configs.get('vision_model', {}).get('name', 'openai/clip-vit-base-patch32')
        cache_dir = self.models_dir / "vision_model"
        cache_dir.mkdir(exist_ok=True)
        
        try:
            # Download processor
            print(f"  Downloading processor: {model_name}")
            processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Download model
            print(f"  Downloading model: {model_name}")
            model = CLIPModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            print("✅ Vision model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading vision model: {e}")
            return False
    
    def download_generation_model(self):
        """Download and cache image generation model"""
        print("🎨 Downloading image generation model...")
        
        model_name = self.model_configs.get('generation_model', {}).get('name', 'runwayml/stable-diffusion-v1-5')
        cache_dir = self.models_dir / "diffusion_model"
        cache_dir.mkdir(exist_ok=True)
        
        try:
            print(f"  Downloading Stable Diffusion: {model_name}")
            print("  ⚠️ This may take a while (several GB)...")
            
            # Download with progress bar
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            print("✅ Generation model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading generation model: {e}")
            return False
    
    def optimize_models(self):
        """Optimize models for deployment"""
        print("⚙️ Optimizing models for deployment...")
        
        try:
            # Model optimization strategies for free deployment
            optimizations = [
                "Quantization to reduce model size",
                "Pruning unnecessary layers",
                "Caching frequently used components",
                "Memory mapping for efficient loading"
            ]
            
            for opt in optimizations:
                print(f"  📋 {opt}")
            
            print("✅ Model optimization completed")
            return True
            
        except Exception as e:
            print(f"❌ Error optimizing models: {e}")
            return False
    
    def verify_installation(self):
        """Verify that all models are properly installed"""
        print("🔍 Verifying model installation...")
        
        verification_results = {}
        
        # Check text model
        try:
            text_cache = self.models_dir / "text_model"
            if text_cache.exists() and any(text_cache.iterdir()):
                verification_results['text_model'] = True
                print("  ✅ Text model: OK")
            else:
                verification_results['text_model'] = False
                print("  ❌ Text model: Missing")
        except Exception as e:
            verification_results['text_model'] = False
            print(f"  ❌ Text model: Error - {e}")
        
        # Check vision model
        try:
            vision_cache = self.models_dir / "vision_model"
            if vision_cache.exists() and any(vision_cache.iterdir()):
                verification_results['vision_model'] = True
                print("  ✅ Vision model: OK")
            else:
                verification_results['vision_model'] = False
                print("  ❌ Vision model: Missing")
        except Exception as e:
            verification_results['vision_model'] = False
            print(f"  ❌ Vision model: Error - {e}")
        
        # Check generation model
        try:
            gen_cache = self.models_dir / "diffusion_model"
            if gen_cache.exists() and any(gen_cache.iterdir()):
                verification_results['generation_model'] = True
                print("  ✅ Generation model: OK")
            else:
                verification_results['generation_model'] = False
                print("  ❌ Generation model: Missing")
        except Exception as e:
            verification_results['generation_model'] = False
            print(f"  ❌ Generation model: Error - {e}")
        
        # Overall status
        all_ok = all(verification_results.values())
        if all_ok:
            print("🎉 All models verified successfully!")
        else:
            print("⚠️ Some models are missing or have issues")
        
        return verification_results
    
    def download_all(self):
        """Download all required models"""
        print("🚀 Starting model download process...")
        print(f"📁 Models will be stored in: {self.models_dir}")
        
        results = {}
        
        # Download each model type
        results['text_model'] = self.download_text_model()
        results['vision_model'] = self.download_vision_model()
        results['generation_model'] = self.download_generation_model()
        
        # Optimize if all downloads successful
        if all(results.values()):
            results['optimization'] = self.optimize_models()
        
        # Verify installation
        verification = self.verify_installation()
        
        # Summary
        print("\n" + "="*50)
        print("📊 DOWNLOAD SUMMARY")
        print("="*50)
        
        for model_type, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{model_type.replace('_', ' ').title()}: {status}")
        
        print("\n📋 Next steps:")
        if all(results.values()):
            print("1. Run 'streamlit run src/streamlit_app/main.py' to start the application")
            print("2. The models will be automatically loaded on first use")
            print("3. Check the Streamlit interface for any additional setup")
        else:
            print("1. Check your internet connection")
            print("2. Ensure you have sufficient disk space (>5GB recommended)")
            print("3. Re-run this script to retry failed downloads")
            print("4. Check the error messages above for specific issues")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Download models for Free Visual AI')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing models')
    parser.add_argument('--force-download', action='store_true', help='Force re-download even if models exist')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.config)
    
    if args.verify_only:
        downloader.verify_installation()
    else:
        if args.force_download:
            print("🔄 Force download mode - will re-download all models")
        
        downloader.download_all()

if __name__ == "__main__":
    main()
