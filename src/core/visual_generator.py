
# src/core/visual_generator.py

"""
Main Visual Generator Class
Orchestrates the entire visualization generation process
"""

import logging
from typing import Dict, List, Any, Optional
from .text_processor import TextProcessor
from .visualization_engine import VisualizationEngine
from .export_manager import ExportManager

logger = logging.getLogger(__name__)

class FreeVisualGenerator:
    """
    Main class that orchestrates the visual generation process
    """
    
    def __init__(self, model_manager):
        """
        Initialize the visual generator
        
        Args:
            model_manager: LocalModelManager instance
        """
        try:
            self.model_manager = model_manager
            
            # Initialize core components with error handling
            self._initialize_components()
            
            # Set default configuration
            self.config = self._get_default_config()
            
            logger.info("FreeVisualGenerator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FreeVisualGenerator: {str(e)}")
            # Initialize with minimal functionality
            self._initialize_fallback_components()
    
    def _initialize_components(self):
        """Initialize core components"""
        try:
            # Initialize text processor
            self.text_processor = TextProcessor(self.model_manager)
            logger.info("TextProcessor initialized")
            
            # Initialize visualization engine
            self.visualization_engine = VisualizationEngine()
            logger.info("VisualizationEngine initialized")
            
            # Initialize export manager
            self.export_manager = ExportManager()
            logger.info("ExportManager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _initialize_fallback_components(self):
        """Initialize minimal components when full initialization fails"""
        logger.warning("Initializing fallback components")
        
        try:
            # Minimal text processor
            self.text_processor = MinimalTextProcessor()
            self.visualization_engine = MinimalVisualizationEngine()
            self.export_manager = MinimalExportManager()
            
        except Exception as e:
            logger.error(f"Failed to initialize even fallback components: {str(e)}")
            # Set None components - will be handled by method calls
            self.text_processor = None
            self.visualization_engine = None
            self.export_manager = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'visualization_types': [
                'flowchart', 'mindmap', 'network', 'timeline', 'infographic'
            ],
            'color_schemes': [
                'professional', 'modern', 'minimal', 'vibrant', 'corporate'
            ],
            'export_formats': ['png', 'html', 'svg', 'pdf'],
            'default_settings': {
                'viz_type': 'flowchart',
                'color_scheme': 'professional',
                'layout': 'auto',
                'quality': 'high'
            }
        }
    
    def generate_visualization(self, text: str, viz_type: str = 'flowchart', 
                             color_scheme: str = 'professional', 
                             **kwargs) -> Dict[str, Any]:
        """
        Generate visualization from text
        
        Args:
            text: Input text to visualize
            viz_type: Type of visualization
            color_scheme: Color scheme to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing visualization results
        """
        try:
            if not self.text_processor:
                return self._generate_error_result("Text processor not available")
            
            # Extract entities from text
            entities = self.text_processor.extract_entities(text)
            
            if not entities:
                return self._generate_error_result("No entities found in text")
            
            # Generate visualization
            if self.visualization_engine:
                visualization = self.visualization_engine.create_visualization(
                    entities, viz_type, color_scheme, **kwargs
                )
            else:
                return self._generate_error_result("Visualization engine not available")
            
            # Prepare result
            result = {
                'success': True,
                'visualization': visualization,
                'entities': entities,
                'metadata': {
                    'entity_count': len(entities),
                    'viz_type': viz_type,
                    'color_scheme': color_scheme,
                    'processing_time': 0.0  # Could add timing
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure"""
        return {
            'success': False,
            'error': error_message,
            'visualization': None,
            'entities': [],
            'metadata': {
                'entity_count': 0,
                'viz_type': 'none',
                'color_scheme': 'none',
                'processing_time': 0.0
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the generator"""
        try:
            status = {
                'overall_status': 'healthy',
                'components': {
                    'model_manager': 'connected' if self.model_manager else 'disconnected',
                    'text_processor': 'available' if self.text_processor else 'unavailable',
                    'visualization_engine': 'available' if self.visualization_engine else 'unavailable',
                    'export_manager': 'available' if self.export_manager else 'unavailable'
                },
                'capabilities': {
                    'text_processing': self.text_processor is not None,
                    'visualization_generation': self.visualization_engine is not None,
                    'export_functionality': self.export_manager is not None
                }
            }
            
            # Overall health check
            if not any(status['capabilities'].values()):
                status['overall_status'] = 'critical'
            elif not all(status['capabilities'].values()):
                status['overall_status'] = 'degraded'
            
            return status
            
        except Exception as e:
            return {
                'overall_status': 'error',
                'error': str(e),
                'components': {},
                'capabilities': {}
            }


# Minimal fallback classes
class MinimalTextProcessor:
    """Minimal text processor for fallback"""
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Simple entity extraction"""
        try:
            # Split by arrows or line breaks
            import re
            steps = re.split(r'[→\->\n]', text)
            entities = []
            
            for i, step in enumerate(steps):
                step = step.strip()
                if step:
                    entities.append({
                        'id': f'step_{i}',
                        'text': step,
                        'order': i,
                        'type': 'process',
                        'importance': 1.0
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in minimal entity extraction: {str(e)}")
            return []

class MinimalVisualizationEngine:
    """Minimal visualization engine for fallback"""
    
    def create_visualization(self, entities: List[Dict[str, Any]], 
                           viz_type: str, color_scheme: str, **kwargs):
        """Create minimal visualization"""
        # This would return a simple text-based representation
        # or a basic chart - implement based on your needs
        return {
            'type': 'minimal',
            'message': 'Minimal visualization mode - full features unavailable',
            'entities_count': len(entities)
        }

class MinimalExportManager:
    """Minimal export manager for fallback"""
    
    def export_visualization(self, visualization, format: str):
        """Minimal export functionality"""
        return {
            'success': False,
            'message': 'Export functionality limited in minimal mode'
        }
