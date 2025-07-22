
import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_loader import LocalModelManager
from src.core.visual_generator import FreeVisualGenerator
import plotly.graph_objects as go
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Free Visual AI Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models with caching"""
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        try:
            model_manager = LocalModelManager()
            return model_manager
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Free Visual AI Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform text into beautiful visualizations - completely free!</p>', unsafe_allow_html=True)
    
    # Initialize models
    model_manager = load_models()
    
    if model_manager is None:
        st.error("Failed to load models. Please refresh the page.")
        st.stop()
    
    # Initialize visual generator
    visual_generator = FreeVisualGenerator(model_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Input type selection
        input_type = st.selectbox(
            "Input Type",
            ["Text Description", "Upload Image", "Mixed Input"],
            help="Choose how you want to provide input"
        )
        
        # Visualization type
        vis_type = st.selectbox(
            "Visualization Type",
            ["flowchart", "mindmap", "network", "infographic", "interactive"],
            help="Choose the type of visualization to generate"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            max_entities = st.slider("Max Entities", 5, 20, 10)
            color_scheme = st.selectbox("Color Scheme", ["default", "corporate", "vibrant", "minimal"])
            export_format = st.selectbox("Export Format", ["PNG", "SVG", "PDF", "HTML"])
    
    # Main content area
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📝 Input")
        
        # Input based on type
        user_input = None
        uploaded_file = None
        
        if input_type == "Text Description":
            user_input = st.text_area(
                "Describe what you want to visualize:",
                height=200,
                placeholder="Enter your text here... For example: 'Create a flowchart showing the process of making coffee: grind beans, boil water, brew coffee, add milk, serve hot'"
            )
            
        elif input_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image to analyze and visualize",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image and we'll analyze it to create visualizations"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
        elif input_type == "Mixed Input":
            user_input = st.text_area(
                "Describe your visualization:",
                height=150,
                placeholder="Describe what you want to create..."
            )
            uploaded_file = st.file_uploader(
                "Optional: Upload reference image",
                type=['png', 'jpg', 'jpeg']
            )
        
        # Generation button
        generate_button = st.button(
            "🚀 Generate Visualization",
            type="primary",
            use_container_width=True
        )
        
        # Example prompts
        with st.expander("💡 Example Prompts"):
            st.write("**Flowchart Examples:**")
            st.code("Create a flowchart for user registration process")
            st.code("Show the steps to deploy a web application")
            
            st.write("**Mind Map Examples:**")
            st.code("Create a mind map about artificial intelligence concepts")
            st.code("Visualize the components of a healthy lifestyle")
            
            st.write("**Network Examples:**")
            st.code("Show relationships between team members and projects")
            st.code("Visualize connections in a social media network")
    
    with col2:
        st.subheader("🎨 Generated Visualization")
        
        if generate_button:
            if not user_input and not uploaded_file:
                st.warning("Please provide some input to generate a visualization.")
            else:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Update progress
                    progress_bar.progress(25)
                    status_text.text("Processing input...")
                    
                    # Generate visualization
                    input_text = user_input or "Analyzing uploaded image..."
                    
                    progress_bar.progress(50)
                    status_text.text("Creating visualization...")
                    
                    result = visual_generator.create_visualization(
                        text=input_text,
                        vis_type=vis_type
                    )
                    
                    progress_bar.progress(75)
                    status_text.text("Finalizing output...")
                    
                    # Display result
                    if result['type'] == 'svg':
                        st.markdown(result['data'], unsafe_allow_html=True)
                        
                        # Download button for SVG
                        st.download_button(
                            "📥 Download SVG",
                            result['data'],
                            file_name=f"{vis_type}_visualization.svg",
                            mime="image/svg+xml"
                        )
                        
                    elif result['type'] == 'plotly':
                        st.plotly_chart(result['data'], use_container_width=True)
                        
                        # Export options for Plotly
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📥 Download HTML"):
                                html_str = result['data'].to_html()
                                st.download_button(
                                    "Download Interactive HTML",
                                    html_str,
                                    file_name=f"{vis_type}_interactive.html",
                                    mime="text/html"
                                )
                        
                    elif result['type'] == 'image':
                        st.image(result['data'], caption=result.get('title', 'Generated Visualization'))
                        
                        # Download button for image
                        if 'buffer' in result:
                            st.download_button(
                                "📥 Download PNG",
                                result['buffer'].getvalue(),
                                file_name=f"{vis_type}_visualization.png",
                                mime="image/png"
                            )
                    
                    progress_bar.progress(100)
                    status_text.markdown('<p class="success-message">✅ Visualization generated successfully!</p>', unsafe_allow_html=True)
                    
                    # Display metadata
                    with st.expander("ℹ️ Generation Details"):
                        st.write(f"**Title:** {result.get('title', 'N/A')}")
                        st.write(f"**Description:** {result.get('description', 'N/A')}")
                        st.write(f"**Type:** {result['type']}")
                        st.write(f"**Visualization Style:** {vis_type}")
                    
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("")
                    st.error(f"Generation failed: {str(e)}")
                    st.info("Try simplifying your input or choosing a different visualization type.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🎨 Free Visual AI Generator | Built with Streamlit | No API keys required</p>
        <p>💡 Tip: For best results, provide clear, structured descriptions of what you want to visualize.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
