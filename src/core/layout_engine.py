
import networkx as nx
import numpy as np
import math
from typing import Dict, List, Tuple
import streamlit as st

class LayoutEngine:
    def __init__(self):
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout
        }
    
    def calculate_layout(self, graph: nx.Graph, layout_type: str = 'spring') -> Dict:
        """Calculate optimal layout for graph"""
        try:
            if layout_type in self.layout_algorithms:
                pos = self.layout_algorithms[layout_type](graph)
            else:
                pos = nx.spring_layout(graph)
            
            # Normalize positions to fit in standard viewport
            pos = self.normalize_positions(pos)
            
            return {
                'positions': pos,
                'nodes': list(graph.nodes(data=True)),
                'edges': list(graph.edges(data=True))
            }
        except Exception as e:
            st.error(f"Layout calculation error: {e}")
            return {'positions': {}, 'nodes': [], 'edges': []}
    
    def normalize_positions(self, positions: Dict) -> Dict:
        """Normalize positions to 0-1000 range"""
        if not positions:
            return positions
        
        # Get min/max coordinates
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Normalize to 0-1000 range with padding
        normalized = {}
        for node, (x, y) in positions.items():
            norm_x = ((x - x_min) / (x_max - x_min) * 800) + 100 if x_max != x_min else 500
            norm_y = ((y - y_min) / (y_max - y_min) * 600) + 100 if y_max != y_min else 400
            normalized[node] = (norm_x, norm_y)
        
        return normalized
    
    def create_flowchart_layout(self, nodes: List, relationships: List) -> Dict:
        """Create hierarchical flowchart layout"""
        layout = {'positions': {}, 'connections': []}
        
        # Simple top-to-bottom flowchart
        y_spacing = 150
        x_spacing = 200
        
        for i, node in enumerate(nodes[:10]):  # Limit to 10 nodes
            row = i // 3
            col = i % 3
            
            layout['positions'][node['text']] = {
                'x': col * x_spacing + 200,
                'y': row * y_spacing + 100,
                'width': 120,
                'height': 60
            }
        
        return layout
    
    def create_mindmap_layout(self, central_node: str, related_nodes: List) -> Dict:
        """Create radial mindmap layout"""
        layout = {'positions': {}}
        
        # Central node
        layout['positions'][central_node] = {'x': 500, 'y': 400, 'width': 150, 'height': 80}
        
        # Surrounding nodes in circle
        angle_step = 2 * math.pi / len(related_nodes) if related_nodes else 0
        radius = 250
        
        for i, node in enumerate(related_nodes[:8]):  # Limit to 8 nodes
            angle = i * angle_step
            x = 500 + radius * math.cos(angle)
            y = 400 + radius * math.sin(angle)
            
            layout['positions'][node] = {
                'x': x,
                'y': y,
                'width': 100,
                'height': 50
            }
        
        return layout
