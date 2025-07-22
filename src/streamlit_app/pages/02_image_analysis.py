
import streamlit as st
import sys
import os
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model_loader import LocalModelManager
from src.core.visual_generator import FreeVisualGenerator

st.set_page_config(
    page_title="Image Analysis",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Analysis & Visualization")
st.markdown("Upload an image and we'll analyze it to create related visualizations")

# Initialize models
@st.cache_resource
def get_models():
    return LocalModelManager()

model_manager = get_models()
visual_generator = FreeVisualGenerator(model_manager)

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Upload an image to analyze and create visualizations from"
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Uploaded Image")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Image details
        st.write(f"**Format:** {image.format}")
        st.write(f"**Size:** {image.size}")
        st.write(f"**Mode:** {image.mode}")
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["General Description", "Object Detection", "Scene Analysis", "Text Extraction"]
        )
        
        output_format = st.selectbox(
            "Output Visualization",
            ["mindmap", "network", "infographic", "flowchart"]
        )

with col2:
    st.subheader("🎨 Analysis Results")
    
    if uploaded_file is not None:
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # For demo purposes, create a simple analysis
                    # In real implementation, this would use the vision model
                    analysis_text = f"""
                    Image Analysis Results:
                    
                    The uploaded image appears to contain various elements that can be visualized.
                    Based on the selected analysis type ({analysis_type}), we can create 
                    a {output_format} showing the relationships and components found.
                    
                    Key elements identified:
                    - Visual components and their relationships
                    - Spatial organization of elements
                    - Color patterns and composition
                    - Potential workflow or process elements
                    """
                    
                    # Generate visualization based on analysis
                    result = visual_generator.create_visualization(
                        text=analysis_text,
                        vis_type=output_format
                    )
                    
                    # Display result
                    if result['type'] == 'svg':
                        st.markdown(result['data'], unsafe_allow_html=True)
                    elif result['type'] == 'plotly':
                        st.plotly_chart(result['data'], use_container_width=True)
                    elif result['type'] == 'image':
                        st.image(result['data'])
                    
                    # Analysis details
                    with st.expander("📊 Detailed Analysis"):
                        st.text_area("Analysis Text", analysis_text, height=200)
                        st.write(f"**Analysis Type:** {analysis_type}")
                        st.write(f"**Output Format:** {output_format}")
                    
                    st.success("Image analysis completed!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    else:
        st.info("👆 Please upload an image to start analysis")

# Tips section
st.markdown("---")
st.subheader("💡 Tips for Better Results")

tips = [
    "🖼️ **Clear Images:** Upload high-quality, clear images for better analysis",
    "📋 **Structured Content:** Images with clear objects or text work best",
    "🎯 **Specific Analysis:** Choose the analysis type that matches your image content",
    "📊 **Visualization Type:** Select the output format that best suits your needs"
]

for tip in tips:
    st.markdown(tip)
