
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Optional
from src.core.text_processor import TextProcessor
from src.core.layout_engine import LayoutEngine
from src.utils.svg_generator import SVGGenerator
import plotly.graph_objects as go
import plotly.express as px

class FreeVisualGenerator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.text_processor = TextProcessor(model_manager)
        self.layout_engine = LayoutEngine()
        self.svg_generator = SVGGenerator()
        
    def create_visualization(self, text: str, vis_type: str, **kwargs) -> Dict:
        """Main visualization creation method"""
        try:
            # Process text
            entities = self.text_processor.extract_entities(text)
            relationships = self.text_processor.extract_relationships(text, entities)
            
            if not entities:
                return self._create_fallback_visualization(text, vis_type)
            
            # Create graph structure
            graph = self.text_processor.create_graph_structure(entities, relationships)
            
            # Generate layout
            layout_data = self.layout_engine.calculate_layout(graph, vis_type)
            
            # Create visualization based on type
            if vis_type == 'flowchart':
                return self._create_flowchart(layout_data, text)
            elif vis_type == 'mindmap':
                return self._create_mindmap(layout_data, text)
            elif vis_type == 'network':
                return self._create_network_diagram(layout_data, text)
            elif vis_type == 'infographic':
                return self._create_infographic(entities, text)
            else:
                return self._create_interactive_chart(entities, relationships)
                
        except Exception as e:
            st.error(f"Visualization generation error: {e}")
            return self._create_fallback_visualization(text, vis_type)
    
    def _create_flowchart(self, layout_data: Dict, original_text: str) -> Dict:
        """Create flowchart visualization"""
        svg_content = self.svg_generator.create_flowchart_svg(layout_data)
        
        return {
            'type': 'svg',
            'data': svg_content,
            'title': 'Generated Flowchart',
            'description': f'Flowchart based on: {original_text[:100]}...'
        }
    
    def _create_mindmap(self, layout_data: Dict, original_text: str) -> Dict:
        """Create mindmap visualization"""
        svg_content = self.svg_generator.create_mindmap_svg(layout_data)
        
        return {
            'type': 'svg',
            'data': svg_content,
            'title': 'Generated Mind Map',
            'description': f'Mind map based on: {original_text[:100]}...'
        }
    
    def _create_network_diagram(self, layout_data: Dict, original_text: str) -> Dict:
        """Create network diagram using Plotly"""
        nodes = layout_data.get('nodes', [])
        edges = layout_data.get('edges', [])
        positions = layout_data.get('positions', {})
        
        if not nodes:
            return self._create_fallback_visualization(original_text, 'network')
        
        # Extract node and edge data for Plotly
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for i, (node_name, node_data) in enumerate(nodes):
            pos = positions.get(node_name, (500, 400))
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_text.append(node_name)
            node_colors.append(f'hsl({(i * 50) % 360}, 70%, 50%)')
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in edges:
            source_pos = positions.get(edge[0], (500, 400))
            target_pos = positions.get(edge[1], (500, 400))
            
            edge_x.extend([source_pos[0], target_pos[0], None])
            edge_y.extend([source_pos[1], target_pos[1], None])
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            hovertext=node_text
        ))
        
        fig.update_layout(
            title='Network Diagram',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text=f"Generated from: {original_text[:50]}...",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return {
            'type': 'plotly',
            'data': fig,
            'title': 'Interactive Network Diagram',
            'description': f'Network visualization of relationships in: {original_text[:100]}...'
        }
    
    def _create_infographic(self, entities: List[Dict], original_text: str) -> Dict:
        """Create simple infographic using PIL"""
        try:
            # Create image
            img = Image.new('RGB', (1000, 800), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use default font, fallback if not available
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                text_font = ImageFont.truetype("arial.ttf", 16)
            except:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Draw title
            draw.text((50, 50), "Generated Infographic", fill='black', font=title_font)
            
            # Draw entity boxes
            y_offset = 120
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            
            for i, entity in enumerate(entities[:6]):  # Limit to 6 entities
                color = colors[i % len(colors)]
                
                # Draw rectangle
                draw.rectangle(
                    [50, y_offset, 450, y_offset + 80],
                    fill=color,
                    outline='black',
                    width=2
                )
                
                # Draw text
                draw.text(
                    (60, y_offset + 20),
                    f"{entity['text']} ({entity['type']})",
                    fill='white',
                    font=text_font
                )
                
                # Draw importance bar
                importance_width = int(entity['importance'] * 100)
                draw.rectangle(
                    [60, y_offset + 50, 60 + importance_width, y_offset + 60],
                    fill='white',
                    outline='black'
                )
                
                y_offset += 100
            
            # Save to BytesIO
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return {
                'type': 'image',
                'data': img,
                'buffer': img_buffer,
                'title': 'Generated Infographic',
                'description': f'Infographic based on: {original_text[:100]}...'
            }
        
        except Exception as e:
            st.error(f"Infographic generation error: {e}")
            return self._create_fallback_visualization(original_text, 'infographic')
    
    def _create_interactive_chart(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """Create interactive chart using Plotly"""
        # Create data for bar chart
        entity_names = [e['text'][:20] for e in entities[:10]]
        importance_scores = [e['importance'] for e in entities[:10]]
        
        fig = px.bar(
            x=entity_names,
            y=importance_scores,
            title='Entity Importance Analysis',
            labels={'x': 'Entities', 'y': 'Importance Score'},
            color=importance_scores,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
        return {
            'type': 'plotly',
            'data': fig,
            'title': 'Interactive Entity Analysis',
            'description': 'Interactive analysis of extracted entities and their importance'
        }
    
    def _create_fallback_visualization(self, text: str, vis_type: str) -> Dict:
        """Create simple fallback visualization"""
        # Create a simple text-based visualization
        img = Image.new('RGB', (800, 600), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw fallback message
        lines = [
            f"Visualization Type: {vis_type.title()}",
            "",
            "Input Text:",
            text[:200] + "..." if len(text) > 200 else text,
            "",
            "Note: This is a simplified visualization.",
            "For better results, try providing more structured text."
        ]
        
        y_offset = 50
        for line in lines:
            draw.text((50, y_offset), line, fill='black', font=font)
            y_offset += 40
        
        return {
            'type': 'image',
            'data': img,
            'title': f'Fallback {vis_type.title()}',
            'description': 'Simplified visualization due to processing limitations'
        }
