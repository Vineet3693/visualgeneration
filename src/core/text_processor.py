
import re
import networkx as nx
import spacy
from typing import List, Dict, Tuple
import streamlit as st

class TextProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.models = model_manager.get_all_models()
        
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        try:
            # Simple entity extraction using regex and keywords
            entities = []
            
            # Common entity patterns
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                'ORG': r'\b[A-Z][a-z]+\s+(Inc|Corp|LLC|Ltd|Company)\b',
                'PROCESS': r'\b(process|step|phase|stage|method)\b',
                'CONCEPT': r'\b[A-Z][a-z]{3,}\b'
            }
            
            for entity_type, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': entity_type,
                        'importance': len(match) / 10  # Simple importance scoring
                    })
            
            return entities[:20]  # Limit to top 20 entities
        except Exception as e:
            st.error(f"Entity extraction error: {e}")
            return []
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship extraction based on proximity
        entity_texts = [e['text'].lower() for e in entities]
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_entities = [e for e in entity_texts if e in sentence_lower]
            
            # Create relationships between entities in same sentence
            for i, entity1 in enumerate(found_entities):
                for entity2 in found_entities[i+1:]:
                    relationships.append({
                        'from': entity1,
                        'to': entity2,
                        'type': 'relates_to',
                        'strength': min(len(sentence) / 100, 1.0)
                    })
        
        return relationships[:50]  # Limit relationships
    
    def determine_visualization_type(self, text: str) -> str:
        """Determine best visualization type from text"""
        text_lower = text.lower()
        
        # Keyword-based classification
        if any(word in text_lower for word in ['flow', 'process', 'step', 'then', 'next']):
            return 'flowchart'
        elif any(word in text_lower for word in ['connect', 'relationship', 'network', 'link']):
            return 'network'
        elif any(word in text_lower for word in ['hierarchy', 'structure', 'organization', 'tree']):
            return 'tree'
        elif any(word in text_lower for word in ['data', 'statistics', 'numbers', 'chart']):
            return 'infographic'
        else:
            return 'mindmap'
    
    def create_graph_structure(self, entities: List[Dict], relationships: List[Dict]) -> nx.Graph:
        """Create NetworkX graph from entities and relationships"""
        G = nx.Graph()
        
        # Add nodes
        for entity in entities:
            G.add_node(
                entity['text'],
                type=entity['type'],
                importance=entity['importance']
            )
        
        # Add edges
        for rel in relationships:
            if rel['from'] in G.nodes and rel['to'] in G.nodes:
                G.add_edge(
                    rel['from'],
                    rel['to'],
                    weight=rel['strength'],
                    type=rel['type']
                )
        
        return G
