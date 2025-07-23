
# Add this method to the LocalModelManager class in src/models/model_loader.py

def get_all_models(self) -> Dict[str, Any]:
    """
    Get all available models (load them if not already loaded)
    
    Returns:
        Dictionary of all loaded models
    """
    try:
        # Load all configured models
        for model_name in self.model_configs.keys():
            if model_name not in self.loaded_models:
                self.load_model(model_name)
        
        return self.loaded_models.copy()
    
    except Exception as e:
        logger.error(f"Error getting all models: {str(e)}")
        return {}

def get_model_by_name(self, model_name: str) -> Optional[Any]:
    """
    Get a specific model by name (alias for load_model)
    
    Args:
        model_name: Name of the model to get
        
    Returns:
        The requested model or None
    """
    return self.load_model(model_name)

def get_loaded_models(self) -> Dict[str, Any]:
    """
    Get currently loaded models without loading new ones
    
    Returns:
        Dictionary of currently loaded models
    """
    return self.loaded_models.copy()

def is_model_loaded(self, model_name: str) -> bool:
    """
    Check if a model is currently loaded
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is loaded, False otherwise
    """
    return model_name in self.loaded_models

def get_model_status(self) -> Dict[str, Dict[str, Any]]:
    """
    Get status of all configured models
    
    Returns:
        Dictionary with model names and their status
    """
    status = {}
    
    for model_name, config in self.model_configs.items():
        status[model_name] = {
            'configured': True,
            'loaded': model_name in self.loaded_models,
            'type': config.get('type', 'unknown'),
            'description': config.get('description', 'No description'),
            'file': config.get('file', 'None')
        }
    
    return status

def reload_model(self, model_name: str) -> Optional[Any]:
    """
    Reload a specific model (unload then load)
    
    Args:
        model_name: Name of the model to reload
        
    Returns:
        Reloaded model or None
    """
    try:
        # Unload if loaded
        if model_name in self.loaded_models:
            self.unload_model(model_name)
        
        # Load again
        return self.load_model(model_name)
    
    except Exception as e:
        logger.error(f"Error reloading model {model_name}: {str(e)}")
        return None

def get_model_memory_usage(self) -> Dict[str, int]:
    """
    Get approximate memory usage of loaded models
    
    Returns:
        Dictionary with model names and their approximate memory usage in bytes
    """
    memory_usage = {}
    
    for model_name, model in self.loaded_models.items():
        try:
            import sys
            memory_usage[model_name] = sys.getsizeof(model)
        except:
            memory_usage[model_name] = 0
    
    return memory_usage

def health_check(self) -> Dict[str, Any]:
    """
    Perform health check on model manager
    
    Returns:
        Health status information
    """
    try:
        total_models = len(self.model_configs)
        loaded_models = len(self.loaded_models)
        
        # Test a simple model load
        test_model = self.load_model('sentiment_analyzer')
        test_passed = test_model is not None
        
        return {
            'status': 'healthy' if test_passed else 'degraded',
            'total_models_configured': total_models,
            'models_loaded': loaded_models,
            'load_test_passed': test_passed,
            'available_models': list(self.model_configs.keys()),
            'loaded_models': list(self.loaded_models.keys())
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'total_models_configured': len(self.model_configs),
            'models_loaded': len(self.loaded_models)
        }
