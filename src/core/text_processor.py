
# src/core/text_processor.py

"""
Text Processing Module for Visual Generator
Handles text analysis and entity extraction
"""

import logging
from typing import Dict, List, Any, Optional
import re
from collections import Counter

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Advanced text processor for visual generation
    """
    
    def __init__(self, model_manager):
        """
        Initialize text processor with model manager
        
        Args:
            model_manager: LocalModelManager instance
        """
        try:
            self.model_manager = model_manager
            
            # Load essential models with fallback
            self.models = {}
            self._load_essential_models()
            
            # Initialize processing components
            self._initialize_processors()
            
            logger.info("TextProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing TextProcessor: {str(e)}")
            # Initialize with minimal functionality
            self.models = {}
            self._initialize_fallback_processors()
    
    def _load_essential_models(self):
        """Load essential models for text processing"""
        essential_models = [
            'entity_extractor',
            'sentiment_analyzer', 
            'keyword_extractor',
            'text_classifier'
        ]
        
        for model_name in essential_models:
            try:
                model = self.model_manager.load_model(model_name)
                if model:
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Failed to load model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
    
    def _initialize_processors(self):
        """Initialize processing components"""
        # Entity extraction patterns
        self.entity_patterns = self._get_entity_patterns()
        
        # Common process indicators
        self.process_indicators = {
            'start': ['start', 'begin', 'initiate', 'launch', 'kick off'],
            'process': ['process', 'handle', 'execute', 'perform', 'conduct'],
            'decision': ['if', 'decide', 'choose', 'when', 'whether'],
            'review': ['review', 'check', 'verify', 'validate', 'examine'],
            'end': ['end', 'finish', 'complete', 'done', 'close']
        }
        
        # Flow connectors
        self.flow_connectors = ['→', '->', 'then', 'next', 'after', 'followed by']
    
    def _initialize_fallback_processors(self):
        """Initialize minimal processors when models fail to load"""
        logger.info("Initializing fallback processors")
        
        self.entity_patterns = self._get_default_patterns()
        self.process_indicators = {
            'start': ['start', 'begin'],
            'process': ['process', 'handle'],
            'decision': ['if', 'decide'],
            'review': ['review', 'check'],
            'end': ['end', 'finish']
        }
        self.flow_connectors = ['→', '->', 'then', 'next']
    
    def _get_entity_patterns(self) -> Dict[str, Any]:
        """Get entity patterns from loaded model or defaults"""
        try:
            if 'entity_extractor' in self.models:
                extractor = self.models['entity_extractor']
                if hasattr(extractor, 'patterns'):
                    return extractor.patterns
                elif isinstance(extractor, dict) and 'patterns' in extractor:
                    return extractor['patterns']
        except Exception as e:
            logger.error(f"Error getting entity patterns: {str(e)}")
        
        return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Get default entity patterns"""
        return {
            'start_patterns': [r'\b(start|begin|initiate)\b'],
            'process_patterns': [r'\b(process|handle|execute)\b'],
            'decision_patterns': [r'\b(if|decide|choose)\b'],
            'review_patterns': [r'\b(review|check|verify)\b'],
            'end_patterns': [r'\b(end|finish|complete)\b']
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted entities
        """
        try:
            entities = []
            
            # Split by common flow indicators
            steps = self._split_into_steps(text)
            
            for i, step in enumerate(steps):
                if step.strip():
                    entity = self._analyze_step(step, i)
                    if entity:
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return self._extract_simple_entities(text)
    
    def _split_into_steps(self, text: str) -> List[str]:
        """Split text into individual steps"""
        # Split by arrows and common connectors
        for connector in self.flow_connectors:
            text = text.replace(connector, '|STEP_SEPARATOR|')
        
        steps = text.split('|STEP_SEPARATOR|')
        return [step.strip() for step in steps if step.strip()]
    
    def _analyze_step(self, step: str, order: int) -> Optional[Dict[str, Any]]:
        """Analyze a single step to extract entity information"""
        try:
            entity = {
                'id': f'step_{order}',
                'text': step.strip(),
                'order': order,
                'type': self._classify_step_type(step),
                'importance': self._calculate_importance(step),
                'sentiment': self._analyze_sentiment(step),
                'keywords': self._extract_keywords(step)
            }
            
            return entity
            
        except Exception as e:
            logger.error(f"Error analyzing step: {str(e)}")
            return {
                'id': f'step_{order}',
                'text': step.strip(),
                'order': order,
                'type': 'process',
                'importance': 1.0,
                'sentiment': 0.0,
                'keywords': []
            }
    
    def _classify_step_type(self, step: str) -> str:
        """Classify the type of a step"""
        step_lower = step.lower()
        
        # Check against indicators
        for step_type, indicators in self.process_indicators.items():
            if any(indicator in step_lower for indicator in indicators):
                return step_type
        
        # Default classification
        return 'process'
    
    def _calculate_importance(self, step: str) -> float:
        """Calculate importance score for a step (1.0 - 5.0)"""
        try:
            score = 1.0
            step_lower = step.lower()
            
            # High importance indicators
            high_importance = ['critical', 'important', 'key', 'essential', 'must']
            if any(word in step_lower for word in high_importance):
                score += 2.0
            
            # Medium importance indicators
            medium_importance = ['should', 'need', 'require', 'verify']
            if any(word in step_lower for word in medium_importance):
                score += 1.0
            
            # Length-based adjustment
            if len(step.split()) > 10:
                score += 0.5
            
            return min(score, 5.0)
            
        except Exception as e:
            logger.error(f"Error calculating importance: {str(e)}")
            return 1.0
    
    def _analyze_sentiment(self, step: str) -> float:
        """Analyze sentiment of a step (-1.0 to 1.0)"""
        try:
            if 'sentiment_analyzer' in self.models:
                analyzer = self.models['sentiment_analyzer']
                if hasattr(analyzer, 'analyze_sentiment'):
                    result = analyzer.analyze_sentiment(step)
                    return result.get('polarity', 0.0)
            
            # Simple fallback sentiment analysis
            positive_words = ['success', 'complete', 'good', 'approve', 'accept']
            negative_words = ['error', 'fail', 'problem', 'reject', 'issue']
            
            step_lower = step.lower()
            pos_count = sum(1 for word in positive_words if word in step_lower)
            neg_count = sum(1 for word in negative_words if word in step_lower)
            
            if pos_count > neg_count:
                return 0.5
            elif neg_count > pos_count:
                return -0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    def _extract_keywords(self, step: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from a step"""
        try:
            if 'keyword_extractor' in self.models:
                extractor = self.models['keyword_extractor']
                if hasattr(extractor, 'extract_keywords'):
                    return extractor.extract_keywords(step, max_keywords)
            
            # Simple keyword extraction
            words = step.lower().split()
            # Filter out common words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _extract_simple_entities(self, text: str) -> List[Dict[str, Any]]:
        """Fallback simple entity extraction"""
        try:
            # Split by arrows or line breaks
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
                        'importance': 1.0,
                        'sentiment': 0.0,
                        'keywords': []
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in simple entity extraction: {str(e)}")
            return []
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity metrics
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        try:
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            sentences = max(sentences, 1)  # Avoid division by zero
            
            metrics = {
                'word_count': len(words),
                'sentence_count': sentences,
                'avg_sentence_length': len(words) / sentences,
                'complexity_score': min(len(words) / 20, 10.0),  # Scale 0-10
                'readability_score': max(10 - (len(words) / sentences), 1.0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing text complexity: {str(e)}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'complexity_score': 1.0,
                'readability_score': 5.0
            }
    
    def get_process_suggestions(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Get suggestions for improving process description
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            if not entities:
                suggestions.append("💡 Add more detailed steps to your process description")
                return suggestions
            
            # Check for start/end points
            has_start = any(e['type'] == 'start' for e in entities)
            has_end = any(e['type'] == 'end' for e in entities)
            
            if not has_start:
                suggestions.append("🚀 Consider adding a clear starting point to your process")
            
            if not has_end:
                suggestions.append("🏁 Consider adding a clear ending point to your process")
            
            # Check for decision points
            has_decisions = any(e['type'] == 'decision' for e in entities)
            if not has_decisions and len(entities) > 3:
                suggestions.append("🤔 Add decision points to make your process more dynamic")
            
            # Check for reviews
            has_reviews = any(e['type'] == 'review' for e in entities)
            if not has_reviews and len(entities) > 5:
                suggestions.append("✅ Consider adding review or validation steps")
            
            # Length-based suggestions
            if len(entities) > 15:
                suggestions.append("📊 Your process is complex - consider breaking it into phases")
            elif len(entities) < 3:
                suggestions.append("📝 Add more detail to make your process visualization more informative")
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            suggestions.append("💡 Try describing your process in more detail")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def validate_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extracted entities and provide quality metrics
        
        Args:
            entities: List of entities to validate
            
        Returns:
            Validation results and quality metrics
        """
        try:
            validation = {
                'valid': True,
                'entity_count': len(entities),
                'quality_score': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            if not entities:
                validation['valid'] = False
                validation['issues'].append("No entities extracted from text")
                return validation
            
            # Check entity diversity
            types = [e['type'] for e in entities]
            unique_types = set(types)
            
            if len(unique_types) == 1:
                validation['recommendations'].append("Add more variety to your process steps")
            
            # Check for logical flow
            has_start = 'start' in types
            has_end = 'end' in types
            
            quality_factors = []
            
            # Quality factors
            quality_factors.append(min(len(entities) / 5, 2.0))  # Appropriate length
            quality_factors.append(len(unique_types) / 2)  # Type diversity
            quality_factors.append(1.0 if has_start else 0.5)  # Has start
            quality_factors.append(1.0 if has_end else 0.5)  # Has end
            
            validation['quality_score'] = min(sum(quality_factors) / len(quality_factors) * 10, 10.0)
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating entities: {str(e)}")
            return {
                'valid': False,
                'entity_count': len(entities) if entities else 0,
                'quality_score': 1.0,
                'issues': [f"Validation error: {str(e)}"],
                'recommendations': ["Try simplifying your input text"]
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        try:
            status = {
                'models_loaded': len(self.models),
                'available_models': list(self.models.keys()),
                'model_manager_status': 'connected' if self.model_manager else 'disconnected'
            }
            
            # Test each model
            for model_name, model in self.models.items():
                status[f'{model_name}_status'] = 'working' if model else 'failed'
            
            return status
            
        except Exception as e:
            return {
                'error': str(e),
                'models_loaded': 0,
                'available_models': [],
                'model_manager_status': 'error'
            }
