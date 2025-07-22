
import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
from diffusers import StableDiffusionPipeline
import os
import yaml
from pathlib import Path

class LocalModelManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
    @st.cache_resource
    def load_text_model(_self):
        """Load lightweight text processing model"""
        try:
            # Using a small, free model for text analysis
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModel.from_pretrained("microsoft/DialoGPT-small")
            
            # Create text analysis pipeline
            text_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer=tokenizer,
                device=0 if _self.device == "cuda" else -1
            )
            
            return {
                'tokenizer': tokenizer,
                'model': model,
                'pipeline': text_pipeline
            }
        except Exception as e:
            st.error(f"Error loading text model: {e}")
            return None
    
    @st.cache_resource
    def load_vision_model(_self):
        """Load CLIP model for image understanding"""
        try:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            return {
                'processor': processor,
                'model': model
            }
        except Exception as e:
            st.error(f"Error loading vision model: {e}")
            return None
    
    @st.cache_resource  
    def load_generation_model(_self):
        """Load Stable Diffusion for image generation"""
        try:
            # Using CPU version for free deployment
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,  # CPU compatible
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to CPU for free deployment
            pipe = pipe.to("cpu")
            pipe.enable_attention_slicing()
            
            return pipe
        except Exception as e:
            st.error(f"Error loading generation model: {e}")
            return None
    
    def get_all_models(self):
        """Load all models at once"""
        if not self.models:
            self.models = {
                'text': self.load_text_model(),
                'vision': self.load_vision_model(),
                'generation': self.load_generation_model()
            }
        return self.models
