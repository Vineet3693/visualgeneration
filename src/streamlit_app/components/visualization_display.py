
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET

class VisualizationDisplay:
    """Handle display of various visualization types"""
    
    def __init__(self):
        self.display_handlers = {
            'svg': self.display_svg,
            'plotly': self.display_plotly,
            'image': self.display_image,
            'html': self.display_html,
            'error': self.display_error
        }
    
    def display_visualization(self, result: Dict[str, Any], container=None) -> None:
        """Main display method that routes to appropriate handler"""
        if container is None:
            container = st
        
        viz_type = result.get('type', 'error')
        
        if viz_type in self.display_handlers:
            self.display_handlers[viz_type](result, container)
        else:
            container.error(f"Unsupported visualization type: {viz_type}")
    
    def display_svg(self, result: Dict[str, Any], container) -> None:
        """Display SVG visualization"""
        try:
            svg_data = result['data']
            
            # Display the SVG
            container.markdown(svg_data, unsafe_allow_html=True)
            
            # Add download button
            self.add_download_button(
                container,
                svg_data,
                filename="visualization.svg",
                mime_type="image/svg+xml",
                label="📥 Download SVG"
            )
            
            # Show SVG info
            with container.expander("ℹ️ SVG Information"):
                try:
                    root = ET.fromstring(svg_data)
                    width = root.get('width', 'Unknown')
                    height = root.get('height', 'Unknown')
                    
                    st.write(f"**Dimensions:** {width} x {height}")
                    st.write(f"**Elements:** {len(list(root.iter()))} SVG elements")
                    st.write(f"**Size:** {len(svg_data)} characters")
                except:
                    st.write("SVG information not available")
            
        except Exception as e:
            container.error(f"Error displaying SVG: {e}")
    
    def display_plotly(self, result: Dict[str, Any], container) -> None:
        """Display Plotly visualization"""
        try:
            fig = result['data']
            
            # Display the chart
            container.plotly_chart(fig, use_container_width=True)
            
            # Add download buttons
            col1, col2, col3 = container.columns(3)
            
            with col1:
                # HTML download
                html_str = fig.to_html(include_plotlyjs='cdn')
                self.add_download_button(
                    st,
                    html_str,
                    filename="visualization.html",
                    mime_type="text/html",
                    label="📥 HTML"
                )
            
            with col2:
                # JSON download
                json_str = fig.to_json()
                self.add_download_button(
                    st,
                    json_str,
                    filename="visualization.json",
                    mime_type="application/json",
                    label="📥 JSON"
                )
            
            with col3:
                # Static image (requires kaleido)
                try:
                    img_bytes = fig.to_image(format="png")
                    self.add_download_button(
                        st,
                        img_bytes,
                        filename="visualization.png",
                        mime_type="image/png",
                        label="📥 PNG"
                    )
                except:
                    st.caption("PNG export not available")
            
            # Show chart info
            with container.expander("ℹ️ Chart Information"):
                st.write(f"**Type:** {type(fig).__name__}")
                st.write(f"**Data traces:** {len(fig.data)}")
                
                if hasattr(fig, 'layout') and fig.layout.title:
                    st.write(f"**Title:** {fig.layout.title.text}")
            
        except Exception as e:
            container.error(f"Error displaying Plotly chart: {e}")
    
    def display_image(self, result: Dict[str, Any], container) -> None:
        """Display PIL Image"""
        try:
            image = result['data']
            title = result.get('title', 'Generated Visualization')
            
            # Display the image
            container.image(image, caption=title, use_column_width=True)
            
            # Add download button
            if 'buffer' in result:
                self.add_download_button(
                    container,
                    result['buffer'].getvalue(),
                    filename="visualization.png",
                    mime_type="image/png",
                    label="📥 Download PNG"
                )
            else:
                # Create buffer if not provided
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                self.add_download_button(
                    container,
                    img_buffer.getvalue(),
                    filename="visualization.png",
                    mime_type="image/png",
                    label="📥 Download PNG"
                )
            
            # Show image info
            with container.expander("ℹ️ Image Information"):
                st.write(f"**Format:** {image.format}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
                
                # Color analysis
                if image.mode == 'RGB':
                    colors = image.getcolors(maxcolors=256*256*256)
                    if colors:
                        st.write(f"**Unique colors:** {len(colors)}")
            
        except Exception as e:
            container.error(f"Error displaying image: {e}")
    
    def display_html(self, result: Dict[str, Any], container) -> None:
        """Display HTML content"""
        try:
            html_data = result['data']
            
            # Display HTML
            st.components.v1.html(html_data, height=600)
            
            # Add download button
            self.add_download_button(
                container,
                html_data,
                filename="visualization.html",
                mime_type="text/html",
                label="📥 Download HTML"
            )
            
        except Exception as e:
            container.error(f"Error displaying HTML: {e}")
    
    def display_error(self, result: Dict[str, Any], container) -> None:
        """Display error message"""
        error_msg = result.get('error', 'Unknown error occurred')
        container.error(f"❌ Visualization Error: {error_msg}")
        
        # Show additional error details if available
        if 'details' in result:
            with container.expander("🔍 Error Details"):
                st.code(result['details'])
        
        # Suggest solutions
        container.info("💡 Try:\n- Simplifying your input\n- Choosing a different visualization type\n- Checking your text for clarity\n- Using fewer entities or relationships")
    
    def add_download_button(self, container, data, filename: str, mime_type: str, label: str) -> None:
        """Add a download button for the visualization"""
        try:
            container.download_button(
                label=label,
                data=data,
                file_name=filename,
                mime=mime_type,
                key=f"download_{filename}_{hash(str(data))}"
            )
        except Exception as e:
            container.error(f"Download button error: {e}")
    
    def display_with_metadata(self, result: Dict[str, Any], show_metadata: bool = True) -> None:
        """Display visualization with optional metadata"""
        # Display the main visualization
        self.display_visualization(result)
        
        # Display metadata if requested
        if show_metadata and 'metadata' in result:
            with st.expander("📊 Generation Metadata"):
                metadata = result['metadata']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                    st.write(f"**Type:** {metadata.get('visualization_type', 'N/A')}")
                    st.write(f"**Processing Time:** {metadata.get('processing_time', 0):.2f}s")
                
                with col2:
                    st.write(f"**Source:** {metadata.get('source', 'N/A')}")
                    st.write(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
                    st.write(f"**Success:** {'✅' if result['type'] != 'error' else '❌'}")
                
                # Show original text
                if 'original_text' in metadata:
                    st.text_area(
                        "Original Input:",
                        metadata['original_text'],
                        height=100,
                        disabled=True
                    )
    
    def create_comparison_view(self, results: list) -> None:
        """Display multiple visualizations for comparison"""
        if not results:
            st.info("No visualizations to compare")
            return
        
        st.subheader("🔄 Visualization Comparison")
        
        # Create columns for comparison
        num_results = len(results)
        if num_results == 1:
            self.display_visualization(results[0])
        elif num_results == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Version 1**")
                self.display_visualization(results[0])
            with col2:
                st.write("**Version 2**")
                self.display_visualization(results[1])
        else:
            # For more than 2, use tabs
            tabs = st.tabs([f"Version {i+1}" for i in range(num_results)])
            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(results):
                        self.display_visualization(results[i])
    
    def display_gallery(self, results: list, items_per_row: int = 3) -> None:
        """Display multiple visualizations in a gallery format"""
        if not results:
            st.info("No visualizations in gallery")
            return
        
        st.subheader("🎨 Visualization Gallery")
        
        # Group results into rows
        for i in range(0, len(results), items_per_row):
            cols = st.columns(items_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(results):
                    result = results[idx]
                    
                    with col:
                        st.write(f"**#{idx + 1}**")
                        
                        # Display thumbnail or preview
                        if result['type'] == 'svg':
                            # For SVG, show a smaller version
                            svg_data = result['data']
                            # Modify SVG to be smaller for gallery
                            modified_svg = svg_data.replace('width="1000"', 'width="300"')
                            modified_svg = modified_svg.replace('height="800"', 'height="240"')
                            st.markdown(modified_svg, unsafe_allow_html=True)
                            
                        elif result['type'] == 'image':
                            # For images, use column width
                            st.image(result['data'], use_column_width=True)
                            
                        elif result['type'] == 'plotly':
                            # For Plotly, show smaller version
                            fig = result['data']
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add metadata
                        if 'metadata' in result:
                            metadata = result['metadata']
                            st.caption(f"Type: {metadata.get('visualization_type', 'N/A')}")
                            st.caption(f"Time: {metadata.get('processing_time', 0):.1f}s")
