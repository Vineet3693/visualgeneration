
# Add after the existing AdvancedTextProcessor class

class AIEnhancedProcessor:
    """AI-enhanced text processing with advanced features"""
    
    def __init__(self):
        self.entity_cache = {}
        self.relationship_patterns = {
            'sequence': ['->', '→', 'then', 'next', 'after', 'followed by'],
            'parallel': ['meanwhile', 'simultaneously', 'at the same time', 'while'],
            'conditional': ['if', 'when', 'unless', 'provided that', 'in case of'],
            'feedback': ['review', 'feedback', 'back to', 'return to', 'loop back']
        }
    
    def enhanced_entity_extraction(self, text: str) -> Dict[str, Any]:
        """Extract entities with AI enhancement"""
        # Use TF-IDF for better keyword extraction
        try:
            vectorizer = TfidfVectorizer(
                max_features=50, 
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            # Prepare text for vectorization
            sentences = text.split('.')
            if len(sentences) > 1:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top keywords
                scores = tfidf_matrix.sum(axis=0).A1
                keywords = [(feature_names[i], scores[i]) for i in scores.argsort()[::-1][:20]]
            else:
                keywords = [(word, 1.0) for word in text.split()[:10]]
            
            return {
                'keywords': keywords,
                'complexity_score': self._calculate_complexity(text),
                'readability_score': self._calculate_readability(text),
                'entity_density': len(text.split()) / max(len(text.split('.')), 1)
            }
        except:
            # Fallback to simple processing
            words = text.split()
            return {
                'keywords': [(word, 1.0) for word in words[:10]],
                'complexity_score': min(len(words) / 10, 10),
                'readability_score': max(10 - len(words) / 20, 1),
                'entity_density': len(words)
            }
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        factors = {
            'length': min(len(text) / 100, 5),
            'sentences': min(len(text.split('.')) / 5, 3),
            'unique_words': len(set(text.lower().split())) / max(len(text.split()), 1) * 2,
            'punctuation': len([c for c in text if c in '.,;:!?']) / max(len(text), 1) * 100
        }
        return sum(factors.values())
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score"""
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_words_per_sentence = len(words) / max(sentences, 1)
        
        # Simple readability formula
        score = max(1, 15 - (avg_words_per_sentence / 5))
        return min(score, 10)
    
    def smart_layout_suggestion(self, entities: List[Dict]) -> Dict[str, Any]:
        """Suggest optimal layout based on content analysis"""
        entity_count = len(entities)
        
        # Analyze entity types
        type_distribution = {}
        for entity in entities:
            entity_type = entity.get('type', 'process')
            type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1
        
        # Suggest layout
        suggestions = {
            'recommended_type': self._recommend_viz_type(entity_count, type_distribution),
            'layout_style': self._recommend_layout_style(entities),
            'color_scheme': self._recommend_colors(type_distribution),
            'complexity_level': 'high' if entity_count > 10 else 'medium' if entity_count > 5 else 'simple'
        }
        
        return suggestions
    
    def _recommend_viz_type(self, count: int, types: Dict) -> str:
        """Recommend visualization type based on analysis"""
        if count <= 3:
            return "Simple Flow"
        elif count <= 8 and types.get('decision', 0) > 0:
            return "Decision Flowchart"
        elif count > 15:
            return "Network Diagram"
        elif types.get('start', 0) > 0 and types.get('end', 0) > 0:
            return "Process Flowchart"
        else:
            return "Mind Map"
    
    def _recommend_layout_style(self, entities: List[Dict]) -> str:
        """Recommend layout style"""
        if len(entities) <= 5:
            return "linear"
        elif len(entities) <= 10:
            return "hierarchical"
        else:
            return "network"
    
    def _recommend_colors(self, types: Dict) -> str:
        """Recommend color scheme"""
        dominant_type = max(types, key=types.get) if types else 'process'
        
        color_map = {
            'start': 'Greens',
            'end': 'Reds', 
            'decision': 'Oranges',
            'process': 'Blues',
            'review': 'Purples'
        }
        
        return color_map.get(dominant_type, 'Blues')
