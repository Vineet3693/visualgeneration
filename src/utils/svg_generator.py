
import xml.etree.ElementTree as ET
from typing import Dict, List
import colorsys

class SVGGenerator:
    def __init__(self):
        self.colors = self.generate_color_palette()
        
    def generate_color_palette(self) -> List[str]:
        """Generate harmonious color palette"""
        colors = []
        for i in range(8):
            hue = i / 8.0
            saturation = 0.7
            lightness = 0.6
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors
    
    def create_flowchart_svg(self, layout_data: Dict) -> str:
        """Generate SVG for flowchart"""
        svg = ET.Element('svg')
        svg.set('width', '1000')
        svg.set('height', '800')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Add styles
        style = ET.SubElement(svg, 'style')
        style.text = """
            .node-rect { fill: #4A90E2; stroke: #2E5C8A; stroke-width: 2; }
            .node-text { fill: white; font-family: Arial; font-size: 12px; text-anchor: middle; }
            .edge-line { stroke: #666; stroke-width: 2; marker-end: url(#arrowhead); }
        """
        
        # Add arrow marker
        defs = ET.SubElement(svg, 'defs')
        marker = ET.SubElement(defs, 'marker')
        marker.set('id', 'arrowhead')
        marker.set('markerWidth', '10')
        marker.set('markerHeight', '7')
        marker.set('refX', '9')
        marker.set('refY', '3.5')
        marker.set('orient', 'auto')
        
        polygon = ET.SubElement(marker, 'polygon')
        polygon.set('points', '0 0, 10 3.5, 0 7')
        polygon.set('fill', '#666')
        
        # Draw nodes
        for i, (node_name, position) in enumerate(layout_data.get('positions', {}).items()):
            color = self.colors[i % len(self.colors)]
            
            # Node rectangle
            rect = ET.SubElement(svg, 'rect')
            rect.set('x', str(position.get('x', 100) - 60))
            rect.set('y', str(position.get('y', 100) - 30))
            rect.set('width', '120')
            rect.set('height', '60')
            rect.set('fill', color)
            rect.set('stroke', '#333')
            rect.set('stroke-width', '2')
            rect.set('rx', '5')
            
            # Node text
            text = ET.SubElement(svg, 'text')
            text.set('x', str(position.get('x', 100)))
            text.set('y', str(position.get('y', 100) + 5))
            text.set('class', 'node-text')
            text.text = node_name[:15] + "..." if len(node_name) > 15 else node_name
        
        # Draw edges
        for edge in layout_data.get('edges', []):
            source_pos = layout_data['positions'].get(edge[0], {'x': 100, 'y': 100})
            target_pos = layout_data['positions'].get(edge[1], {'x': 200, 'y': 200})
            
            line = ET.SubElement(svg, 'line')
            line.set('x1', str(source_pos['x']))
            line.set('y1', str(source_pos['y']))
            line.set('x2', str(target_pos['x']))
            line.set('y2', str(target_pos['y']))
            line.set('class', 'edge-line')
        
        return ET.tostring(svg, encoding='unicode', method='xml')
    
    def create_mindmap_svg(self, layout_data: Dict) -> str:
        """Generate SVG for mindmap"""
        svg = ET.Element('svg')
        svg.set('width', '1000')
        svg.set('height', '800')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Add central node
        positions = layout_data.get('positions', {})
        if positions:
            central_node = list(positions.keys())[0]
            central_pos = positions[central_node]
            
            # Central circle
            circle = ET.SubElement(svg, 'circle')
            circle.set('cx', str(central_pos['x']))
            circle.set('cy', str(central_pos['y']))
            circle.set('r', '60')
            circle.set('fill', self.colors[0])
            circle.set('stroke', '#333')
            circle.set('stroke-width', '3')
            
            # Central text
            text = ET.SubElement(svg, 'text')
            text.set('x', str(central_pos['x']))
            text.set('y', str(central_pos['y'] + 5))
            text.set('text-anchor', 'middle')
            text.set('font-family', 'Arial')
            text.set('font-size', '14')
            text.set('fill', 'white')
            text.text = central_node[:10] + "..." if len(central_node) > 10 else central_node
            
            # Draw branches and child nodes
            for i, (node_name, position) in enumerate(list(positions.items())[1:]):
                color = self.colors[(i + 1) % len(self.colors)]
                
                # Branch line
                line = ET.SubElement(svg, 'line')
                line.set('x1', str(central_pos['x']))
                line.set('y1', str(central_pos['y']))
                line.set('x2', str(position['x']))
                line.set('y2', str(position['y']))
                line.set('stroke', color)
                line.set('stroke-width', '3')
                
                # Child node
                circle = ET.SubElement(svg, 'circle')
                circle.set('cx', str(position['x']))
                circle.set('cy', str(position['y']))
                circle.set('r', '40')
                circle.set('fill', color)
                circle.set('stroke', '#333')
                circle.set('stroke-width', '2')
                
                # Child text
                text = ET.SubElement(svg, 'text')
                text.set('x', str(position['x']))
                text.set('y', str(position['y'] + 5))
                text.set('text-anchor', 'middle')
                text.set('font-family', 'Arial')
                text.set('font-size', '10')
                text.set('fill', 'white')
                text.text = node_name[:8] + "..." if len(node_name) > 8 else node_name
        
        return ET.tostring(svg, encoding='unicode', method='xml')
