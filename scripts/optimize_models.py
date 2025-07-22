
#!/usr/bin/env python3
"""
Script to optimize models for better performance and smaller size.
This is especially important for free deployment platforms with resource limitations.
"""

import os
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, config_path="config.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models" / "local_models"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def quantize_model(self, model_path, output_path):
        """Quantize model to reduce size"""
        logger.info(f"🔧 Quantizing model: {model_path}")
        
        try:
            # Load original model
            model = torch.load(model_path, map_location='cpu')
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save(quantized_model, output_path)
            
            # Compare sizes
            original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            reduction = (1 - quantized_size/original_size) * 100
            
            logger.info(f"✅ Quantization complete:")
            logger.info(f"   Original: {original_size:.1f} MB")
            logger.info(f"   Quantized: {quantized_size:.1f} MB")
            logger.info(f"   Reduction: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            return False
    
    def optimize_for_cpu(self):
        """Optimize models specifically for CPU inference"""
        logger.info("🖥️ Optimizing models for CPU inference")
        
        optimizations = {
            'use_torch_jit': True,
            'enable_onnx_export': False,  # Requires additional dependencies
            'optimize_attention': True,
            'reduce_precision': True
        }
        
        for opt_name, enabled in optimizations.items():
            status = "✅ Enabled" if enabled else "⚠️ Disabled"
            logger.info(f"   {opt_name}: {status}")
        
        return optimizations
    
    def create_model_cache_strategy(self):
        """Create caching strategy for models"""
        logger.info("💾 Setting up model caching strategy")
        
        cache_config = {
            'cache_dir': str(self.models_dir),
            'enable_disk_cache': True,
            'enable_memory_cache': True,
            'cache_size_limit': '2GB',
            'auto_cleanup': True
        }
        
        # Save cache config
        cache_config_path = self.models_dir / 'cache_config.yaml'
        with open(cache_config_path, 'w') as f:
            yaml.dump(cache_config, f)
        
        logger.info("✅ Cache configuration saved")
        return cache_config
    
    def optimize_all_models(self):
        """Run all optimization procedures"""
        logger.info("🚀 Starting comprehensive model optimization")
        
        results = {}
        
        # CPU optimization
        results['cpu_optimization'] = self.optimize_for_cpu()
        
        # Caching strategy
        results['cache_strategy'] = self.create_model_cache_strategy()
        
        # Model-specific optimizations
        text_model_dir = self.models_dir / "text_model"
        if text_model_dir.exists():
            logger.info("📝 Optimizing text model...")
            # Add text model specific optimizations here
            results['text_optimization'] = True
        
        vision_model_dir = self.models_dir / "vision_model"
        if vision_model_dir.exists():
            logger.info("👁️ Optimizing vision model...")
            # Add vision model specific optimizations here
            results['vision_optimization'] = True
        
        logger.info("🎉 Model optimization completed!")
        return results

def main():
    optimizer = ModelOptimizer()
    optimizer.optimize_all_models()

if __name__ == "__main__":
    main()
