
import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model_loader import LocalModelManager
from src.core.visual_generator import FreeVisualGenerator

st.set_page_config(
    page_title="Text to Visual",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Text to Visual Converter")
st.markdown("Convert any text description into beautiful visualizations")

# Initialize models (cached from main app)
@st.cache_resource
def get_models():
    return LocalModelManager()

model_manager = get_models()
visual_generator = FreeVisualGenerator(model_manager)

# Input section
st.subheader("Input Text")
input_text = st.text_area(
    "Enter your text description:",
    height=200,
    placeholder="Describe a process, concept, or relationship you want to visualize..."
)

# Options
col1, col2, col3 = st.columns(3)

with col1:
    vis_type = st.selectbox(
        "Visualization Type",
        ["flowchart", "mindmap", "network", "infographic"]
    )

with col2:
    style = st.selectbox(
        "Style",
        ["professional", "creative", "minimal", "corporate"]
    )

with col3:
    size = st.selectbox(
        "Size",
        ["standard (1000x800)", "large (1200x900)", "wide (1400x800)"]
    )

# Generate button
if st.button("Generate Visualization", type="primary"):
    if input_text:
        with st.spinner("Generating visualization..."):
            try:
                result = visual_generator.create_visualization(
                    text=input_text,
                    vis_type=vis_type
                )
                
                # Display result
                if result['type'] == 'svg':
                    st.markdown(result['data'], unsafe_allow_html=True)
                elif result['type'] == 'plotly':
                    st.plotly_chart(result['data'])
                elif result['type'] == 'image':
                    st.image(result['data'])
                
                st.success("Visualization generated successfully!")
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text to visualize.")

# Examples section
st.subheader("💡 Example Prompts")

examples = {
    "Process Flow": "Create a flowchart showing the steps to launch a new product: market research, product development, testing, marketing campaign, launch event, customer feedback",
    "Mind Map": "Create a mind map about machine learning: supervised learning, unsupervised learning, neural networks, data preprocessing, model training, evaluation metrics",
    "Network Diagram": "Show relationships in a software team: project manager connects to developers, QA testers, and designers. Developers work with database admin and DevOps engineer",
    "Infographic": "Create an infographic about healthy eating: fruits and vegetables provide vitamins, whole grains give energy, proteins build muscle, water keeps you hydrated"
}

for title, example in examples.items():
    with st.expander(f"📋 {title} Example"):
        st.code(example)
        if st.button(f"Use {title} Example", key=f"example_{title}"):
            st.session_state.example_text = example
            st.experimental_rerun()

# Pre-fill with example if selected
if 'example_text' in st.session_state:
    input_text = st.session_state.example_text
    del st.session_state.example_text
