
# Add this complete image analysis system

class ImageAnalyzer:
    """Advanced image analysis and OCR processing"""
    
    def __init__(self):
        self.supported_formats = ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'TIFF']
    
    def analyze_uploaded_image(self, image_file) -> Dict[str, Any]:
        """Analyze uploaded image and extract information"""
        try:
            # Load image
            image = Image.open(image_file)
            
            # Basic image properties
            analysis = {
                'format': image.format,
                'size': image.size,
                'mode': image.mode,
                'has_transparency': image.mode in ('RGBA', 'LA'),
            }
            
            # Extract text if possible
            extracted_text = self.extract_text_from_image(image)
            analysis['extracted_text'] = extracted_text
            
            # Analyze image content
            analysis['content_analysis'] = self.analyze_image_content(image)
            
            # Detect existing diagrams
            analysis['diagram_detection'] = self.detect_diagram_elements(image)
            
            return analysis
            
        except Exception as e:
            return {'error': f"Image analysis failed: {str(e)}"}
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text using OCR (fallback method without pytesseract)"""
        try:
            # Simple text extraction fallback
            # In production, you could use pytesseract here
            return "OCR not available - please install pytesseract for text extraction"
        except:
            return ""
    
    def analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image content and properties"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        if colors:
            # Most dominant colors
            dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
            color_info = {
                'dominant_colors': [f"rgb{color[1]}" for color in dominant_colors[:3]],
                'color_diversity': len(colors),
                'is_colorful': len(colors) > 100
            }
        else:
            color_info = {'dominant_colors': [], 'color_diversity': 0, 'is_colorful': False}
        
        # Image characteristics
        width, height = image.size
        aspect_ratio = width / height
        
        analysis = {
            'dimensions': {'width': width, 'height': height},
            'aspect_ratio': round(aspect_ratio, 2),
            'orientation': 'landscape' if aspect_ratio > 1.2 else 'portrait' if aspect_ratio < 0.8 else 'square',
            'size_category': 'large' if width > 1000 or height > 1000 else 'medium' if width > 500 else 'small',
            **color_info
        }
        
        return analysis
    
    def detect_diagram_elements(self, image: Image.Image) -> Dict[str, Any]:
        """Detect if image contains diagram elements"""
        try:
            # Convert to numpy array for analysis
            import numpy as np
            img_array = np.array(image.convert('RGB'))
            
            # Simple shape detection (basic approach)
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Detect edges (simple gradient)
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            
            edge_density = (np.sum(grad_x > 30) + np.sum(grad_y > 30)) / gray.size
            
            # Analyze patterns
            has_rectangular_patterns = edge_density > 0.1
            has_circular_patterns = False  # Simplified
            has_text_regions = edge_density > 0.05 and edge_density < 0.3
            
            return {
                'has_shapes': has_rectangular_patterns,
                'has_text': has_text_regions,
                'edge_density': float(edge_density),
                'likely_diagram': has_rectangular_patterns or has_text_regions,
                'complexity': 'high' if edge_density > 0.2 else 'medium' if edge_density > 0.1 else 'low'
            }
        except:
            return {
                'has_shapes': False,
                'has_text': False,
                'edge_density': 0,
                'likely_diagram': False,
                'complexity': 'unknown'
            }
    
    def suggest_visualization_from_image(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Suggest visualization type based on image analysis"""
        suggestions = {}
        
        if analysis.get('diagram_detection', {}).get('likely_diagram', False):
            suggestions['type'] = "Diagram Recreation"
            suggestions['reason'] = "Image appears to contain existing diagram elements"
        elif analysis.get('extracted_text'):
            suggestions['type'] = "Text-based Visualization"  
            suggestions['reason'] = "Text extracted from image can be visualized"
        else:
            suggestions['type'] = "Content Visualization"
            suggestions['reason'] = "Image content can inspire visualization themes"
            
        return suggestions

def create_image_analysis_interface():
    """Create image analysis interface"""
    st.subheader("📸 Image Analysis & OCR")
    
    uploaded_file = st.file_uploader(
        "Upload an image for analysis:",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload images containing text, diagrams, or content you want to analyze"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Analyze image
            with st.spinner("Analyzing image..."):
                analyzer = ImageAnalyzer()
                analysis = analyzer.analyze_uploaded_image(uploaded_file)
                
                if 'error' not in analysis:
                    st.success("✅ Image analysis completed!")
                    
                    # Show analysis results
                    with st.expander("🔍 Analysis Results"):
                        st.json(analysis)
                    
                    # Show extracted text if available
                    if analysis.get('extracted_text'):
                        st.subheader("📝 Extracted Text")
                        extracted_text = st.text_area(
                            "Text found in image:",
                            value=analysis['extracted_text'],
                            height=100
                        )
                        
                        if st.button("🎨 Visualize Extracted Text"):
                            return extracted_text
                    
                    # Show suggestions
                    suggestions = analyzer.suggest_visualization_from_image(analysis)
                    if suggestions:
                        st.info(f"💡 **Suggestion**: {suggestions['type']}")
                        st.write(f"**Reason**: {suggestions['reason']}")
                
                else:
                    st.error(f"❌ {analysis['error']}")
    
    return None
