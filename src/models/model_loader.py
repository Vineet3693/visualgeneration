
# src/models/model_loader.py

"""
Local Model Manager for AI Visual Generator
Handles loading and management of ML models locally
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModelManager:
    """
    Manages local machine learning models and their loading/caching
    """
    
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize the model manager
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default model configurations"""
        self.model_configs = {
            'text_classifier': {
                'type': 'sklearn',
                'file': 'text_classifier.pkl',
                'description': 'Text classification model for process types'
            },
            'entity_extractor': {
                'type': 'custom',
                'file': 'entity_patterns.json',
                'description': 'Entity extraction patterns'
            },
            'sentiment_analyzer': {
                'type': 'textblob',
                'file': None,
                'description': 'Built-in sentiment analysis'
            },
            'keyword_extractor': {
                'type': 'tfidf',
                'file': 'keywords.json',
                'description': 'TF-IDF based keyword extraction'
            },
            'image_processor': {
                'type': 'opencv',
                'file': None,
                'description': 'OpenCV image processing'
            }
        }
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a specific model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object or None if failed
        """
        try:
            # Return cached model if already loaded
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # Check if model config exists
            if model_name not in self.model_configs:
                logger.warning(f"Model {model_name} not found in configurations")
                return None
            
            config = self.model_configs[model_name]
            model_type = config['type']
            
            # Load based on model type
            if model_type == 'sklearn':
                model = self._load_sklearn_model(model_name, config)
            elif model_type == 'custom':
                model = self._load_custom_model(model_name, config)
            elif model_type == 'textblob':
                model = self._load_textblob_model(model_name, config)
            elif model_type == 'tfidf':
                model = self._load_tfidf_model(model_name, config)
            elif model_type == 'opencv':
                model = self._load_opencv_model(model_name, config)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            # Cache the loaded model
            if model is not None:
                self.loaded_models[model_name] = model
                logger.info(f"Successfully loaded model: {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def _load_sklearn_model(self, model_name: str, config: Dict) -> Optional[Any]:
        """Load scikit-learn model"""
        try:
            from sklearn.base import BaseEstimator
            
            model_path = self.models_dir / config['file']
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            else:
                # Create a simple fallback classifier
                logger.info(f"Creating fallback classifier for {model_name}")
                return self._create_fallback_classifier()
                
        except ImportError:
            logger.warning("scikit-learn not available, using fallback")
            return self._create_simple_classifier()
        except Exception as e:
            logger.error(f"Error loading sklearn model: {str(e)}")
            return None
    
    def _load_custom_model(self, model_name: str, config: Dict) -> Optional[Any]:
        """Load custom model (usually JSON data)"""
        try:
            model_path = self.models_dir / config['file']
            if model_path.exists():
                with open(model_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default patterns
                return self._get_default_patterns(model_name)
                
        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            return self._get_default_patterns(model_name)
    
    def _load_textblob_model(self, model_name: str, config: Dict) -> Optional[Any]:
        """Load TextBlob-based model"""
        try:
            from textblob import TextBlob
            
            # Return a TextBlob wrapper
            class TextBlobWrapper:
                def analyze_sentiment(self, text: str) -> Dict[str, float]:
                    blob = TextBlob(text)
                    return {
                        'polarity': blob.sentiment.polarity,
                        'subjectivity': blob.sentiment.subjectivity
                    }
            
            return TextBlobWrapper()
            
        except ImportError:
            logger.warning("TextBlob not available, using simple sentiment")
            return self._create_simple_sentiment_analyzer()
        except Exception as e:
            logger.error(f"Error creating TextBlob model: {str(e)}")
            return None
    
    def _load_tfidf_model(self, model_name: str, config: Dict) -> Optional[Any]:
        """Load TF-IDF based model"""
        try:
            # Simple keyword extractor
            class SimpleKeywordExtractor:
                def __init__(self):
                    self.common_words = {
                        'process', 'step', 'task', 'action', 'review', 
                        'approve', 'submit', 'complete', 'start', 'end'
                    }
                
                def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
                    words = text.lower().split()
                    # Simple frequency-based extraction
                    word_freq = {}
                    for word in words:
                        if len(word) > 3 and word.isalpha():
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    # Sort by frequency and return top keywords
                    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                    return [word for word, freq in sorted_words[:max_keywords]]
            
            return SimpleKeywordExtractor()
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF model: {str(e)}")
            return None
    
    def _load_opencv_model(self, model_name: str, config: Dict) -> Optional[Any]:
        """Load OpenCV-based model"""
        try:
            import cv2
            
            # Simple OpenCV wrapper
            class OpenCVWrapper:
                def process_image(self, image_path: str) -> Dict[str, Any]:
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            return {'error': 'Could not load image'}
                        
                        height, width = img.shape[:2]
                        return {
                            'width': width,
                            'height': height,
                            'channels': img.shape[2] if len(img.shape) > 2 else 1,
                            'size': img.size
                        }
                    except Exception as e:
                        return {'error': str(e)}
            
            return OpenCVWrapper()
            
        except ImportError:
            logger.warning("OpenCV not available, using PIL fallback")
            return self._create_pil_wrapper()
        except Exception as e:
            logger.error(f"Error creating OpenCV model: {str(e)}")
            return None
    
    def _create_fallback_classifier(self) -> Any:
        """Create a simple fallback classifier"""
        class SimpleClassifier:
            def __init__(self):
                self.process_keywords = {
                    'start': ['begin', 'start', 'initiate', 'launch'],
                    'process': ['process', 'handle', 'execute', 'perform'],
                    'decision': ['decide', 'choose', 'if', 'whether', 'select'],
                    'review': ['review', 'check', 'verify', 'validate'],
                    'end': ['end', 'finish', 'complete', 'done', 'close']
                }
            
            def classify_text(self, text: str) -> str:
                text_lower = text.lower()
                for process_type, keywords in self.process_keywords.items():
                    if any(keyword in text_lower for keyword in keywords):
                        return process_type
                return 'process'  # Default
        
        return SimpleClassifier()
    
    def _create_simple_classifier(self) -> Any:
        """Create simple classifier without sklearn"""
        return self._create_fallback_classifier()
    
    def _create_simple_sentiment_analyzer(self) -> Any:
        """Create simple sentiment analyzer"""
        class SimpleSentiment:
            def __init__(self):
                self.positive_words = {'good', 'great', 'excellent', 'success', 'complete', 'approve'}
                self.negative_words = {'bad', 'error', 'fail', 'problem', 'issue', 'reject'}
            
            def analyze_sentiment(self, text: str) -> Dict[str, float]:
                text_lower = text.lower()
                positive_count = sum(1 for word in self.positive_words if word in text_lower)
                negative_count = sum(1 for word in self.negative_words if word in text_lower)
                
                total = positive_count + negative_count
                if total == 0:
                    return {'polarity': 0.0, 'subjectivity': 0.0}
                
                polarity = (positive_count - negative_count) / total
                subjectivity = min(total / 10.0, 1.0)  # Normalize
                
                return {'polarity': polarity, 'subjectivity': subjectivity}
        
        return SimpleSentiment()
    
    def _create_pil_wrapper(self) -> Any:
        """Create PIL-based image processor as OpenCV fallback"""
        try:
            from PIL import Image
            
            class PILWrapper:
                def process_image(self, image_path: str) -> Dict[str, Any]:
                    try:
                        img = Image.open(image_path)
                        return {
                            'width': img.width,
                            'height': img.height,
                            'mode': img.mode,
                            'format': img.format
                        }
                    except Exception as e:
                        return {'error': str(e)}
            
            return PILWrapper()
            
        except ImportError:
            logger.warning("PIL also not available")
            return None
    
    def _get_default_patterns(self, model_name: str) -> Dict[str, Any]:
        """Get default patterns for custom models"""
        if model_name == 'entity_extractor':
            return {
                'patterns': {
                    'start_patterns': [r'\b(start|begin|initiate)\b'],
                    'process_patterns': [r'\b(process|handle|execute)\b'],
                    'decision_patterns': [r'\b(if|decide|choose)\b'],
                    'end_patterns': [r'\b(end|finish|complete)\b']
                },
                'keywords': {
                    'action_words': ['process', 'review', 'approve', 'submit'],
                    'decision_words': ['if', 'when', 'decide', 'choose'],
                    'flow_words': ['then', 'next', 'after', 'before']
                }
            }
        
        return {'default': True}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.model_configs:
            config = self.model_configs[model_name].copy()
            config['loaded'] = model_name in self.loaded_models
            return config
        return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False
    
    def clear_all_models(self):
        """Clear all loaded models from memory"""
        self.loaded_models.clear()
        logger.info("Cleared all loaded models")

# Convenience function for easy import
def get_model_manager() -> LocalModelManager:
    """Get a singleton instance of LocalModelManager"""
    if not hasattr(get_model_manager, '_instance'):
        get_model_manager._instance = LocalModelManager()
    return get_model_manager._instance
