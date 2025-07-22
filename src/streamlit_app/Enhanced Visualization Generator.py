
class EnhancedVisualizationGenerator:
    """Enhanced visualization generator with multiple advanced types"""
    
    def __init__(self):
        self.color_schemes = {
            'professional': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD'],
            'modern': ['#1ABC9C', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C'],
            'minimal': ['#34495E', '#7F8C8D', '#BDC3C7', '#ECF0F1', '#95A5A6'],
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
            'corporate': ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#27AE60']
        }
        
        self.layout_templates = {
            'linear': {'spacing': 2, 'direction': 'horizontal'},
            'hierarchical': {'levels': 3, 'branching': 2},
            'circular': {'radius_increment': 1.5, 'angle_offset': 0},
            'grid': {'cols': 3, 'cell_size': 1.5},
            'force_directed': {'iterations': 50, 'k_factor': 2}
        }
    
    def create_advanced_flowchart(self, text: str, entities: List[Dict] = None) -> go.Figure:
        """Create advanced flowchart with decision nodes and parallel processes"""
        if not entities:
            processor = AdvancedTextProcessor()
            entities = processor.extract_entities(text)
        
        if not entities:
            return self._create_empty_chart("No entities found for flowchart")
        
        fig = go.Figure()
        
        # Analyze flow structure
        flow_analysis = self._analyze_flow_structure(entities)
        
        # Calculate advanced positioning
        positions = self._calculate_advanced_positions(entities, flow_analysis)
        
        # Add nodes with advanced styling
        for i, entity in enumerate(entities):
            x, y = positions[i]
            
            # Determine node style based on type
            node_style = self._get_node_style(entity)
            
            # Add main node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=node_style['size'],
                    color=node_style['color'],
                    line=dict(width=3, color=node_style['border_color']),
                    symbol=node_style['symbol']
                ),
                text=self._format_node_text(entity['text'], node_style['max_chars']),
                textposition="middle center",
                textfont=dict(
                    size=node_style['text_size'],
                    color=node_style['text_color'],
                    family="Arial Black"
                ),
                name=entity['text'],
                hovertemplate=self._create_hover_template(entity),
                customdata=[entity]
            ))
        
        # Add advanced connections
        self._add_advanced_connections(fig, entities, positions, flow_analysis)
        
        # Add decision paths if detected
        self._add_decision_paths(fig, entities, positions)
        
        # Apply advanced styling
        fig.update_layout(
            title=dict(
                text="📊 Advanced Process Flowchart",
                x=0.5,
                font=dict(size=24, family="Arial Black")
            ),
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(248,249,250,0.8)',
            margin=dict(l=50, r=50, t=100, b=50),
            annotations=self._create_flow_annotations(entities)
        )
        
        return fig
    
    def create_dynamic_mindmap(self, text: str, entities: List[Dict] = None) -> go.Figure:
        """Create dynamic mind map with multiple levels and clustering"""
        if not entities:
            processor = AdvancedTextProcessor()
            entities = processor.extract_entities(text)
        
        if not entities:
            return self._create_empty_chart("No entities found for mind map")
        
        fig = go.Figure()
        
        # Analyze entity relationships for clustering
        clusters = self._cluster_entities(entities)
        
        # Create central node
        central_text = self._determine_central_topic(entities, text)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=150, color='#E74C3C', line=dict(width=4, color='#C0392B')),
            text=self._wrap_text(central_text, 15),
            textposition="middle center",
            textfont=dict(size=14, color='white', family='Arial Black'),
            name="Central Topic",
            hovertemplate=f"<b>{central_text}</b><br>Central Topic<extra></extra>"
        ))
        
        # Add clustered branches
        self._add_clustered_branches(fig, clusters, entities)
        
        # Add connecting lines with styling
        self._add_mindmap_connections(fig, clusters)
        
        # Apply dynamic styling
        fig.update_layout(
            title=dict(
                text="🧠 Dynamic Mind Map",
                x=0.5,
                font=dict(size=24, family="Arial Black")
            ),
            showlegend=False,
            height=700,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4]),
            plot_bgcolor='rgba(250,251,252,0.9)',
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig
    
    def create_interactive_network(self, text: str, entities: List[Dict] = None) -> go.Figure:
        """Create interactive network diagram with force-directed layout"""
        if not entities:
            processor = AdvancedTextProcessor()
            entities = processor.extract_entities(text)
        
        if not entities:
            return self._create_empty_chart("No entities found for network")
        
        # Build network graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for entity in entities:
            G.add_node(
                entity['id'],
                label=entity['text'],
                type=entity['type'],
                importance=entity.get('importance', 1.0),
                sentiment=entity.get('sentiment', 0.0)
            )
        
        # Add edges based on relationships
        relationships = self._extract_advanced_relationships(entities, text)
        for rel in relationships:
            G.add_edge(rel['from'], rel['to'], weight=rel['strength'], type=rel['type'])
        
        # Calculate layout using force-directed algorithm
        if len(G.nodes()) > 1:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        else:
            pos = {list(G.nodes())[0]: (0, 0)}
        
        fig = go.Figure()
        
        # Add edges with varying thickness
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1.0)
            edge_type = edge[2].get('type', 'default')
            
            # Style edge based on type
            edge_style = self._get_edge_style(edge_type, weight)
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge_style['width'], color=edge_style['color']),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes with advanced styling
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_data = node[1]
            
            # Calculate node size based on importance
            base_size = 60
            importance_factor = node_data.get('importance', 1.0)
            node_size = base_size * (0.5 + importance_factor)
            
            # Color based on type and sentiment
            node_color = self._get_network_node_color(node_data)
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=self._wrap_text(node_data['label'], 12),
                textposition="middle center",
                textfont=dict(size=9, color='white', family='Arial'),
                name=node_data['label'],
                hovertemplate=self._create_network_hover_template(node_data)
            ))
        
        # Add network statistics
        network_stats = self._calculate_network_stats(G)
        
        fig.update_layout(
            title=dict(
                text=f"🌐 Interactive Network Diagram<br><sub>Nodes: {network_stats['nodes']} | Edges: {network_stats['edges']} | Density: {network_stats['density']:.2f}</sub>",
                x=0.5,
                font=dict(size=20, family="Arial Black")
            ),
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(245,246,247,0.8)'
        )
        
        return fig
    
    def create_smart_timeline(self, text: str, entities: List[Dict] = None) -> go.Figure:
        """Create smart timeline with milestones and phases"""
        if not entities:
            processor = AdvancedTextProcessor()
            entities = processor.extract_entities(text)
        
        if not entities:
            return self._create_empty_chart("No timeline data found")
        
        fig = go.Figure()
        
        # Generate smart dates
        timeline_data = self._generate_timeline_data(entities)
        
        # Detect phases
        phases = self._detect_timeline_phases(timeline_data)
        
        # Add phase backgrounds
        for i, phase in enumerate(phases):
            fig.add_shape(
                type="rect",
                x0=phase['start_date'], x1=phase['end_date'],
                y0=-0.5, y1=1.5,
                fillcolor=phase['color'],
                opacity=0.2,
                line_width=0
            )
            
            # Add phase label
            fig.add_annotation(
                x=(phase['start_date'] + phase['end_date']) / 2,
                y=1.3,
                text=phase['name'],
                showarrow=False,
                font=dict(size=12, color=phase['color'])
            )
        
        # Add timeline events
        for i, event in enumerate(timeline_data):
            # Determine event type and styling
            event_style = self._get_timeline_event_style(event)
            
            fig.add_trace(go.Scatter(
                x=[event['date']],
                y=[1],
                mode='markers+text',
                marker=dict(
                    size=event_style['size'],
                    color=event_style['color'],
                    line=dict(width=2, color='white'),
                    symbol=event_style['symbol']
                ),
                text=self._wrap_text(event['text'], 15),
                textposition="top center",
                textfont=dict(size=10, color=event_style['color']),
                name=event['text'],
                hovertemplate=self._create_timeline_hover_template(event)
            ))
        
        # Add connecting timeline
        dates = [event['date'] for event in timeline_data]
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1] * len(dates),
            mode='lines',
            line=dict(color='#34495E', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add milestones
        milestones = self._identify_milestones(timeline_data)
        for milestone in milestones:
            fig.add_trace(go.Scatter(
                x=[milestone['date']],
                y=[0.5],
                mode='markers+text',
                marker=dict(size=30, color='#E74C3C', symbol='star'),
                text="★",
                textposition="middle center",
                name=f"Milestone: {milestone['text']}",
                hovertemplate=f"<b>🎯 Milestone</b><br>{milestone['text']}<extra></extra>"
            ))
        
        fig.update_layout(
            title=dict(
                text="⏰ Smart Timeline Visualization",
                x=0.5,
                font=dict(size=24, family="Arial Black")
            ),
            xaxis=dict(
                title="Timeline",
                tickangle=45,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                range=[0, 2],
                showticklabels=False,
                showgrid=False
            ),
            height=500,
            plot_bgcolor='rgba(250,251,252,0.9)',
            showlegend=False
        )
        
        return fig
    
    def create_data_infographic(self, text: str, entities: List[Dict] = None) -> go.Figure:
        """Create data-rich infographic with statistics and insights"""
        if not entities:
            processor = AdvancedTextProcessor()
            entities = processor.extract_entities(text)
        
        if not entities:
            return self._create_empty_chart("No data for infographic")
        
        # Analyze content for data insights
        data_insights = self._extract_data_insights(text, entities)
        
        fig = go.Figure()
        
        # Create subplot structure
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Process Distribution', 'Complexity Analysis', 'Timeline Flow', 'Key Metrics'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # 1. Process Distribution (Pie Chart)
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        if type_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    name="Process Types",
                    marker_colors=self.color_schemes['professional'][:len(type_counts)]
                ),
                row=1, col=1
            )
        
        # 2. Complexity Analysis (Bar Chart)
        complexity_data = self._calculate_entity_complexity(entities)
        if complexity_data:
            fig.add_trace(
                go.Bar(
                    x=[f"Item {i+1}" for i in range(len(complexity_data))],
                    y=complexity_data,
                    name="Complexity Score",
                    marker_color='#3498DB'
                ),
                row=1, col=2
            )
        
        # 3. Timeline Flow (Line Chart)
        if len(entities) > 1:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(entities))),
                    y=[entity.get('importance', 1) for entity in entities],
                    mode='lines+markers',
                    name="Process Flow",
                    line=dict(color='#E74C3C', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
        
        # 4. Key Metrics (Gauge/Indicator)
        metrics = self._calculate_process_metrics(entities, text)
        if metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=metrics['efficiency_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Process Efficiency"},
                    gauge={'axis': {'range': [None, 10]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 5], 'color': "lightgray"},
                               {'range': [5, 10], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 8}}
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="📈 Process Analytics Infographic",
            title_x=0.5,
            title_font_size=24,
            height=700,
            showlegend=False
        )
        
        return fig
    
    # Helper methods for enhanced visualizations
    
    def _analyze_flow_structure(self, entities: List[Dict]) -> Dict:
        """Analyze flow structure for advanced positioning"""
        structure = {
            'has_parallel_paths': False,
            'decision_points': [],
            'start_nodes': [],
            'end_nodes': [],
            'process_depth': 1
        }
        
        for entity in entities:
            if entity['type'] == 'start':
                structure['start_nodes'].append(entity)
            elif entity['type'] == 'end':
                structure['end_nodes'].append(entity)
            elif entity['type'] == 'decision':
                structure['decision_points'].append(entity)
        
        # Estimate process depth based on entity count
        structure['process_depth'] = max(1, len(entities) // 4)
        
        return structure
    
    def _calculate_advanced_positions(self, entities: List[Dict], flow_analysis: Dict) -> List[Tuple[float, float]]:
        """Calculate advanced positions for flowchart elements"""
        positions = []
        
        if len(entities) <= 1:
            return [(0, 0)]
        
        # Use hierarchical layout for better flow visualization
        levels = max(3, len(entities) // 3)
        items_per_level = len(entities) // levels
        
        for i, entity in enumerate(entities):
            level = i // max(1, items_per_level)
            position_in_level = i % max(1, items_per_level)
            
            # Calculate x position (horizontal spread)
            if items_per_level > 1:
                x = position_in_level - (items_per_level - 1) / 2
            else:
                x = 0
            
            # Calculate y position (vertical levels)
            y = -(level - (levels - 1) / 2) * 1.5
            
            # Add some randomization for decision nodes
            if entity['type'] == 'decision':
                x += 0.3 * (i % 2 * 2 - 1)  # Slight offset
            
            positions.append((x * 2.5, y))
        
        return positions
    
    def _get_node_style(self, entity: Dict) -> Dict:
        """Get node styling based on entity type"""
        styles = {
            'start': {
                'size': 120,
                'color': '#27AE60',
                'border_color': '#1E8449',
                'symbol': 'circle',
                'text_size': 11,
                'text_color': 'white',
                'max_chars': 20
            },
            'end': {
                'size': 120,
                'color': '#E74C3C',
                'border_color': '#C0392B',
                'symbol': 'circle',
                'text_size': 11,
                'text_color': 'white',
                'max_chars': 20
            },
            'decision': {
                'size': 100,
                'color': '#F39C12',
                'border_color': '#D68910',
                'symbol': 'diamond',
                'text_size': 10,
                'text_color': 'white',
                'max_chars': 18
            },
            'process': {
                'size': 110,
                'color': '#3498DB',
                'border_color': '#2874A6',
                'symbol': 'square',
                'text_size': 10,
                'text_color': 'white',
                'max_chars': 22
            }
        }
      return styles.get(entity.get('type', 'process'), styles['process'])
    
    def _format_node_text(self, text: str, max_chars: int) -> str:
        """Format node text with proper wrapping"""
        if len(text) <= max_chars:
            return text
        
        # Smart truncation at word boundaries
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Word is too long, truncate it
                    lines.append(word[:max_chars-3] + '...')
                    current_line = []
                    current_length = 0
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines[:3])  # Max 3 lines
    
    def _create_hover_template(self, entity: Dict) -> str:
        """Create detailed hover template for entities"""
        template = f"<b>{entity['text']}</b><br>"
        template += f"Type: {entity['type'].title()}<br>"
        
        if entity.get('importance'):
            template += f"Importance: {entity['importance']:.1f}/5<br>"
        
        if entity.get('sentiment') is not None:
            sentiment = "Positive" if entity['sentiment'] > 0.1 else "Negative" if entity['sentiment'] < -0.1 else "Neutral"
            template += f"Sentiment: {sentiment}<br>"
        
        if entity.get('keywords'):
            keywords = ', '.join(entity['keywords'][:3])
            template += f"Keywords: {keywords}<br>"
        
        template += "<extra></extra>"
        return template
    
    def _add_advanced_connections(self, fig: go.Figure, entities: List[Dict], positions: List[Tuple], flow_analysis: Dict):
        """Add advanced connections between entities"""
        for i in range(len(entities) - 1):
            x0, y0 = positions[i]
            x1, y1 = positions[i + 1]
            
            # Determine connection style based on entity types
            from_type = entities[i]['type']
            to_type = entities[i + 1]['type']
            
            # Calculate connection path (curved for decisions)
            if from_type == 'decision':
                # Add curved path for decision branches
                self._add_curved_connection(fig, x0, y0, x1, y1, '#F39C12')
            else:
                # Straight connection
                self._add_straight_connection(fig, x0, y0, x1, y1, '#2C3E50')
    
    def _add_curved_connection(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, color: str):
        """Add curved connection for decision paths"""
        # Calculate control points for bezier curve
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2 + 0.5  # Slight curve upward
        
        # Create curved path using multiple points
        t_values = np.linspace(0, 1, 20)
        curve_x = []
        curve_y = []
        
        for t in t_values:
            # Quadratic bezier curve
            x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
            y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
            curve_x.append(x)
            curve_y.append(y)
        
        fig.add_trace(go.Scatter(
            x=curve_x + [None],
            y=curve_y + [None],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add arrowhead
        fig.add_annotation(
            x=x1, y=y1,
            ax=curve_x[-2], ay=curve_y[-2],
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=color
        )
    
    def _add_straight_connection(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, color: str):
        """Add straight connection with arrow"""
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=color
        )
    
    def _cluster_entities(self, entities: List[Dict]) -> List[List[Dict]]:
        """Cluster entities for mind map organization"""
        if len(entities) <= 4:
            return [entities]  # No clustering needed for small sets
        
        # Simple clustering based on entity types
        clusters = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in clusters:
                clusters[entity_type] = []
            clusters[entity_type].append(entity)
        
        # Convert to list of clusters
        cluster_list = list(clusters.values())
        
        # Ensure no cluster is too large
        final_clusters = []
        for cluster in cluster_list:
            if len(cluster) > 6:
                # Split large clusters
                mid = len(cluster) // 2
                final_clusters.append(cluster[:mid])
                final_clusters.append(cluster[mid:])
            else:
                final_clusters.append(cluster)
        
        return final_clusters
    
    def _determine_central_topic(self, entities: List[Dict], text: str) -> str:
        """Determine the central topic for mind map"""
        # Try to extract main topic from text
        if len(entities) > 0:
            first_entity = entities[0]['text']
            if any(word in first_entity.lower() for word in ['process', 'system', 'workflow']):
                return first_entity
        
        # Fallback to generic topic
        words = text.split()[:3]
        return ' '.join(words) if words else "Main Topic"
    
    def _add_clustered_branches(self, fig: go.Figure, clusters: List[List[Dict]], all_entities: List[Dict]):
        """Add clustered branches to mind map"""
        cluster_colors = self.color_schemes['vibrant']
        
        for cluster_idx, cluster in enumerate(clusters):
            # Calculate cluster position
            n_clusters = len(clusters)
            angle = 2 * np.pi * cluster_idx / n_clusters
            cluster_radius = 2.0 + (cluster_idx % 2) * 0.5
            
            cluster_center_x = cluster_radius * np.cos(angle)
            cluster_center_y = cluster_radius * np.sin(angle)
            
            # Add cluster nodes
            for node_idx, entity in enumerate(cluster):
                # Position within cluster
                if len(cluster) > 1:
                    sub_angle = angle + (node_idx - len(cluster)/2) * 0.3
                    sub_radius = cluster_radius + 0.3 * (node_idx % 2)
                else:
                    sub_angle = angle
                    sub_radius = cluster_radius
                
                x = sub_radius * np.cos(sub_angle)
                y = sub_radius * np.sin(sub_angle)
                
                color = cluster_colors[cluster_idx % len(cluster_colors)]
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=80,
                        color=color,
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=self._wrap_text(entity['text'], 12),
                    textposition="middle center",
                    textfont=dict(size=9, color='white', family='Arial'),
                    name=entity['text'],
                    hovertemplate=self._create_hover_template(entity)
                ))
    
    def _extract_advanced_relationships(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract advanced relationships between entities"""
        relationships = []
        
        # Sequential relationships
        for i in range(len(entities) - 1):
            relationships.append({
                'from': entities[i]['id'],
                'to': entities[i + 1]['id'],
                'type': 'sequence',
                'strength': 1.0
            })
        
        # Look for explicit relationships in text
        text_lower = text.lower()
        
        # Parallel relationships
        if any(keyword in text_lower for keyword in ['meanwhile', 'simultaneously', 'parallel']):
            # Add parallel relationships for entities that might run simultaneously
            for i in range(len(entities) - 2):
                relationships.append({
                    'from': entities[i]['id'],
                    'to': entities[i + 2]['id'],
                    'type': 'parallel',
                    'strength': 0.5
                })
        
        # Feedback relationships
        if any(keyword in text_lower for keyword in ['review', 'feedback', 'loop', 'return']):
            # Add feedback loop from end to beginning or middle
            if len(entities) > 2:
                relationships.append({
                    'from': entities[-1]['id'],
                    'to': entities[len(entities)//2]['id'],
                    'type': 'feedback',
                    'strength': 0.3
                })
        
        return relationships
    
    def _get_edge_style(self, edge_type: str, weight: float) -> Dict:
        """Get edge styling based on type and weight"""
        styles = {
            'sequence': {'color': '#2C3E50', 'width': 2 + weight * 2},
            'parallel': {'color': '#3498DB', 'width': 1 + weight * 2},
            'feedback': {'color': '#E74C3C', 'width': 1 + weight * 1.5},
            'default': {'color': '#7F8C8D', 'width': 1 + weight}
        }
        
        return styles.get(edge_type, styles['default'])
    
    def _get_network_node_color(self, node_data: Dict) -> str:
        """Get node color for network based on properties"""
        node_type = node_data.get('type', 'process')
        sentiment = node_data.get('sentiment', 0)
        
        base_colors = {
            'start': '#27AE60',
            'end': '#E74C3C',
            'decision': '#F39C12',
            'process': '#3498DB',
            'review': '#9B59B6'
        }
        
        base_color = base_colors.get(node_type, base_colors['process'])
        
        # Modify based on sentiment (simplified)
        if sentiment > 0.2:
            return base_color  # Keep bright for positive
        elif sentiment < -0.2:
            return base_color.replace('3', '2')  # Slightly darker for negative
        else:
            return base_color  # Neutral
    
    def _create_network_hover_template(self, node_data: Dict) -> str:
        """Create hover template for network nodes"""
        template = f"<b>{node_data['label']}</b><br>"
        template += f"Type: {node_data.get('type', 'Unknown').title()}<br>"
        
        if node_data.get('importance'):
            template += f"Importance: {node_data['importance']:.1f}<br>"
        
        if node_data.get('sentiment') is not None:
            sentiment_text = "Positive" if node_data['sentiment'] > 0.1 else "Negative" if node_data['sentiment'] < -0.1 else "Neutral"
            template += f"Sentiment: {sentiment_text}<br>"
        
        template += "<extra></extra>"
        return template
    
    def _calculate_network_stats(self, G) -> Dict:
        """Calculate network statistics"""
        stats = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G) if len(G.nodes()) > 1 else 0,
            'connected_components': nx.number_connected_components(G),
        }
        
        if len(G.nodes()) > 0:
            try:
                stats['average_clustering'] = nx.average_clustering(G)
            except:
                stats['average_clustering'] = 0
        
        return stats
    
    def _generate_timeline_data(self, entities: List[Dict]) -> List[Dict]:
        """Generate timeline data with smart date assignment"""
        timeline_data = []
        
        # Generate dates spanning a reasonable period
        start_date = datetime.now()
        total_duration = timedelta(days=len(entities) * 7)  # 1 week per entity
        
        for i, entity in enumerate(entities):
            # Calculate date for this entity
            progress = i / max(len(entities) - 1, 1)
            entity_date = start_date + total_duration * progress
            
            timeline_data.append({
                'text': entity['text'],
                'date': entity_date,
                'type': entity['type'],
                'importance': entity.get('importance', 1.0),
                'order': i
            })
        
        return timeline_data
    
    def _detect_timeline_phases(self, timeline_data: List[Dict]) -> List[Dict]:
        """Detect phases in timeline for background coloring"""
        phases = []
        
        if len(timeline_data) <= 3:
            # Single phase
            phases.append({
                'name': 'Process Execution',
                'start_date': timeline_data[0]['date'],
                'end_date': timeline_data[-1]['date'],
                'color': '#3498DB'
            })
        else:
            # Multiple phases
            phase_size = len(timeline_data) // 3
            phase_colors = ['#3498DB', '#E74C3C', '#27AE60']
            phase_names = ['Planning', 'Execution', 'Completion']
            
            for i in range(3):
                start_idx = i * phase_size
                end_idx = (i + 1) * phase_size if i < 2 else len(timeline_data)
                
                if start_idx < len(timeline_data):
                    phases.append({
                        'name': phase_names[i],
                        'start_date': timeline_data[start_idx]['date'],
                        'end_date': timeline_data[min(end_idx - 1, len(timeline_data) - 1)]['date'],
                        'color': phase_colors[i]})
        
        return phases
    
    def _get_timeline_event_style(self, event: Dict) -> Dict:
        """Get styling for timeline events"""
        styles = {
            'start': {'size': 25, 'color': '#27AE60', 'symbol': 'circle'},
            'end': {'size': 25, 'color': '#E74C3C', 'symbol': 'circle'},
            'decision': {'size': 20, 'color': '#F39C12', 'symbol': 'diamond'},
            'process': {'size': 18, 'color': '#3498DB', 'symbol': 'square'},
            'review': {'size': 20, 'color': '#9B59B6', 'symbol': 'hexagon'}
        }
        
        base_style = styles.get(event.get('type', 'process'), styles['process'])
        
        # Adjust size based on importance
        importance_factor = event.get('importance', 1.0)
        base_style['size'] = int(base_style['size'] * (0.7 + importance_factor * 0.6))
        
        return base_style
    
    def _create_timeline_hover_template(self, event: Dict) -> str:
        """Create hover template for timeline events"""
        template = f"<b>{event['text']}</b><br>"
        template += f"Date: {event['date'].strftime('%Y-%m-%d')}<br>"
        template += f"Type: {event.get('type', 'Unknown').title()}<br>"
        
        if event.get('importance'):
            template += f"Importance: {event['importance']:.1f}/5<br>"
        
        template += f"Order: {event.get('order', 0) + 1}<br>"
        template += "<extra></extra>"
        return template
    
    def _identify_milestones(self, timeline_data: List[Dict]) -> List[Dict]:
        """Identify key milestones in timeline"""
        milestones = []
        
        # Mark high importance events as milestones
        for event in timeline_data:
            if event.get('importance', 1.0) > 3.0 or event.get('type') in ['start', 'end']:
                milestones.append(event)
        
        # If no clear milestones, mark key phases
        if not milestones and len(timeline_data) > 3:
            # Mark beginning, middle, and end
            indices = [0, len(timeline_data) // 2, len(timeline_data) - 1]
            for idx in indices:
                milestones.append(timeline_data[idx])
        
        return milestones
    
    def _extract_data_insights(self, text: str, entities: List[Dict]) -> Dict:
        """Extract data insights for infographic"""
        insights = {
            'total_steps': len(entities),
            'complexity_score': self._calculate_text_complexity(text),
            'decision_points': len([e for e in entities if e['type'] == 'decision']),
            'automation_potential': self._estimate_automation_potential(entities),
            'estimated_duration': self._estimate_process_duration(entities),
            'risk_level': self._assess_risk_level(text, entities)
        }
        
        return insights
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        
        complexity_factors = {
            'word_count': min(len(words) / 50, 5),
            'sentence_complexity': min(len(words) / max(sentences, 1) / 10, 3),
            'technical_terms': len([w for w in words if len(w) > 8]) / len(words) * 5,
            'decision_keywords': len([w for w in words if w.lower() in ['if', 'when', 'unless', 'decide']]) * 0.5
        }
        
        return min(sum(complexity_factors.values()), 10)
    
    def _estimate_automation_potential(self, entities: List[Dict]) -> float:
        """Estimate automation potential percentage"""
        automatable_types = ['process', 'data']
        manual_types = ['decision', 'review']
        
        automatable_count = len([e for e in entities if e['type'] in automatable_types])
        manual_count = len([e for e in entities if e['type'] in manual_types])
        
        if len(entities) == 0:
            return 0
        
        automation_score = (automatable_count / len(entities)) * 100
        automation_score -= (manual_count / len(entities)) * 30  # Reduce for manual steps
        
        return max(0, min(automation_score, 95))
    
    def _estimate_process_duration(self, entities: List[Dict]) -> str:
        """Estimate process duration based on complexity"""
        base_minutes_per_step = {
            'start': 5,
            'process': 15,
            'decision': 20,
            'review': 25,
            'end': 5
        }
        
        total_minutes = sum(base_minutes_per_step.get(e['type'], 15) for e in entities)
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        elif total_minutes < 480:  # 8 hours
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"
        else:
            days = total_minutes // (60 * 8)  # 8-hour work days
            return f"{days} days"
    
    def _assess_risk_level(self, text: str, entities: List[Dict]) -> str:
        """Assess process risk level"""
        risk_keywords = ['error', 'fail', 'problem', 'issue', 'risk', 'critical', 'urgent']
        decision_count = len([e for e in entities if e['type'] == 'decision'])
        
        text_lower = text.lower()
        risk_mentions = sum(1 for keyword in risk_keywords if keyword in text_lower)
        
        risk_score = risk_mentions * 2 + decision_count * 0.5
        
        if risk_score > 5:
            return "High"
        elif risk_score > 2:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_entity_complexity(self, entities: List[Dict]) -> List[float]:
        """Calculate complexity score for each entity"""
        complexity_scores = []
        
        for entity in entities:
            score = 1.0  # Base score
            
            # Add complexity based on text length
            score += len(entity['text'].split()) * 0.2
            
            # Add complexity based on type
            type_complexity = {
                'start': 1.0,
                'process': 2.0,
                'decision': 3.5,
                'review': 3.0,
                'end': 1.0
            }
            score += type_complexity.get(entity['type'], 2.0)
            
            # Add complexity based on keywords
            complex_keywords = ['analyze', 'evaluate', 'assess', 'determine', 'optimize']
            if any(keyword in entity['text'].lower() for keyword in complex_keywords):
                score += 1.5
            
            complexity_scores.append(min(score, 10.0))
        
        return complexity_scores
    
    def _calculate_process_metrics(self, entities: List[Dict], text: str) -> Dict:
        """Calculate overall process metrics"""
        metrics = {}
        
        # Efficiency Score (0-10)
        decision_ratio = len([e for e in entities if e['type'] == 'decision']) / max(len(entities), 1)
        review_ratio = len([e for e in entities if e['type'] == 'review']) / max(len(entities), 1)
        
        efficiency = 8.0 - (decision_ratio * 3) - (review_ratio * 2)  # Fewer decisions/reviews = more efficient
        efficiency += 1 if any(word in text.lower() for word in ['automat', 'streamlin', 'optim']) else 0
        
        metrics['efficiency_score'] = max(1, min(efficiency, 10))
        
        # Clarity Score
        avg_text_length = np.mean([len(e['text']) for e in entities]) if entities else 0
        clarity = 8.0 - (avg_text_length / 50)  # Shorter, clearer descriptions
        metrics['clarity_score'] = max(1, min(clarity, 10))
        
        # Completeness Score
        has_start = any(e['type'] == 'start' for e in entities)
        has_end = any(e['type'] == 'end' for e in entities)
        completeness = 5 + (3 if has_start else 0) + (2 if has_end else 0)
        metrics['completeness_score'] = min(completeness, 10)
        
        return metrics
    
    def _wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text for display"""
        if len(text) <= max_length:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines[:2])  # Max 2 lines
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
