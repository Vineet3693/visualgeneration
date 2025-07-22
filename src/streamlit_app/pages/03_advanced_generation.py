
import streamlit as st
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model_loader import LocalModelManager
from src.core.visual_generator import FreeVisualGenerator

st.set_page_config(
    page_title="Advanced Generation",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Advanced Visualization Generation")
st.markdown("Fine-tune your visualizations with advanced settings and templates")

# Initialize models
@st.cache_resource
def get_models():
    return LocalModelManager()

model_manager = get_models()
visual_generator = FreeVisualGenerator(model_manager)

# Advanced settings in sidebar
with st.sidebar:
    st.header("🔧 Advanced Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    temperature = st.slider("Creativity Level", 0.1, 2.0, 0.7, 0.1)
    max_entities = st.slider("Max Entities", 5, 50, 15)
    relationship_threshold = st.slider("Relationship Threshold", 0.1, 1.0, 0.5, 0.1)
    
    # Visual settings
    st.subheader("Visual Configuration")
    color_palette = st.selectbox(
        "Color Palette",
        ["default", "corporate", "vibrant", "pastel", "monochrome", "nature"]
    )
    
    layout_algorithm = st.selectbox(
        "Layout Algorithm",
        ["spring", "circular", "hierarchical", "random", "force_directed"]
    )
    
    # Export settings
    st.subheader("Export Options")
    export_width = st.number_input("Width (px)", 500, 2000, 1000, 100)
    export_height = st.number_input("Height (px)", 400, 1500, 800, 100)
    include_metadata = st.checkbox("Include Metadata", True)

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Custom Generation", "📋 Templates", "🔄 Batch Processing"])

with tab1:
    st.subheader("Custom Visualization Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Text Description", "Structured Data", "JSON Import"]
        )
        
        if input_method == "Text Description":
            user_input = st.text_area(
                "Detailed Description",
                height=200,
                placeholder="Provide a detailed description with specific relationships and hierarchies..."
            )
            
        elif input_method == "Structured Data":
            st.write("Define your data structure:")
            
            # Entity definition
            st.write("**Entities:**")
            num_entities = st.number_input("Number of Entities", 2, 20, 5)
            
            entities = []
            for i in range(num_entities):
                entity_name = st.text_input(f"Entity {i+1} Name", key=f"entity_{i}")
                entity_type = st.selectbox(f"Entity {i+1} Type", 
                                         ["concept", "person", "process", "data", "system"], 
                                         key=f"type_{i}")
                if entity_name:
                    entities.append({"name": entity_name, "type": entity_type})
            
            # Relationship definition
            if len(entities) > 1:
                st.write("**Relationships:**")
                for i, entity1 in enumerate(entities):
                    for j, entity2 in enumerate(entities[i+1:], i+1):
                        if st.checkbox(f"{entity1['name']} → {entity2['name']}", key=f"rel_{i}_{j}"):
                            rel_type = st.selectbox(f"Relationship Type", 
                                                  ["connects to", "depends on", "contains", "flows to"],
                                                  key=f"rel_type_{i}_{j}")
        
        elif input_method == "JSON Import":
            json_input = st.text_area(
                "JSON Data",
                height=300,
                placeholder='''{
  "entities": [
    {"name": "Start", "type": "process"},
    {"name": "Decision", "type": "decision"},
    {"name": "End", "type": "process"}
  ],
  "relationships": [
    {"from": "Start", "to": "Decision", "type": "flows_to"},
    {"from": "Decision", "to": "End", "type": "flows_to"}
  ]
}'''
            )
    
    with col2:
        st.subheader("Visualization Preview")
        
        # Visualization type with advanced options
        vis_type = st.selectbox(
            "Visualization Type",
            ["flowchart", "mindmap", "network", "infographic", "tree", "timeline"]
        )
        
        # Type-specific options
        if vis_type == "flowchart":
            direction = st.selectbox("Flow Direction", ["top-bottom", "left-right", "bottom-top", "right-left"])
            shape_style = st.selectbox("Node Shape", ["rectangle", "rounded", "circle", "diamond"])
        
        elif vis_type == "network":
            force_strength = st.slider("Force Strength", 0.1, 2.0, 1.0, 0.1)
            show_labels = st.checkbox("Show Node Labels", True)
        
        # Generate button
        if st.button("🚀 Generate Advanced Visualization", type="primary"):
            with st.spinner("Generating with advanced settings..."):
                try:
                    # Prepare input based on method
                    if input_method == "Text Description":
                        input_text = user_input
                    elif input_method == "Structured Data":
                        # Convert structured data to text description
                        input_text = f"Create a {vis_type} with these entities: {', '.join([e['name'] for e in entities])}"
                    elif input_method == "JSON Import":
                        try:
                            json_data = json.loads(json_input)
                            entities_text = ', '.join([e['name'] for e in json_data.get('entities', [])])
                            input_text = f"Create a {vis_type} with entities: {entities_text}"
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format")
                            input_text = ""
                    
                    if input_text:
                        result = visual_generator.create_visualization(
                            text=input_text,
                            vis_type=vis_type
                        )
                        
                        # Display result
                        if result['type'] == 'svg':
                            st.markdown(result['data'], unsafe_allow_html=True)
                        elif result['type'] == 'plotly':
                            st.plotly_chart(result['data'], use_container_width=True)
                        elif result['type'] == 'image':
                            st.image(result['data'])
                        
                        # Export options
                        st.download_button(
                            "📥 Download Result",
                            result.get('data', ''),
                            file_name=f"advanced_{vis_type}.svg",
                            mime="image/svg+xml"
                        )
                
                except Exception as e:
                    st.error(f"Generation failed: {e}")

with tab2:
    st.subheader("📋 Visualization Templates")
    
    # Template categories
    template_category = st.selectbox(
        "Template Category",
        ["Business Process", "Software Architecture", "Data Flow", "Organizational", "Educational"]
    )
    
    # Template examples based on category
    templates = {
        "Business Process": {
            "Customer Journey": "Customer sees ad → Visits website → Signs up → Makes purchase → Receives product → Provides feedback",
            "Project Workflow": "Planning phase → Design phase → Development phase → Testing phase → Deployment phase → Maintenance",
            "Sales Funnel": "Awareness → Interest → Consideration → Purchase → Retention → Advocacy"
        },
        "Software Architecture": {
            "MVC Pattern": "User Interface → Controller → Model → Database, Controller handles requests, Model processes data",
            "Microservices": "API Gateway → User Service → Product Service → Order Service → Payment Service → Database cluster",
            "CI/CD Pipeline": "Code commit → Build → Test → Deploy to staging → Manual approval → Deploy to production"
        },
        "Data Flow": {
            "ETL Process": "Extract from sources → Transform data → Load to warehouse → Generate reports → Business decisions",
            "ML Pipeline": "Data collection → Data cleaning → Feature engineering → Model training → Model validation → Deployment",
            "Analytics Flow": "Raw data → Processing → Analysis → Visualization → Insights → Actions"
        }
    }
    
    if template_category in templates:
        template_name = st.selectbox("Template", list(templates[template_category].keys()))
        template_text = templates[template_category][template_name]
        
        st.text_area("Template Description", template_text, height=100)
        
        col_a, col_b = st.columns(2)
        with col_a:
            template_vis_type = st.selectbox("Visualization Type for Template", 
                                           ["flowchart", "mindmap", "network"])
        
        with col_b:
            if st.button("🎯 Use Template", type="primary"):
                with st.spinner("Generating from template..."):
                    try:
                        result = visual_generator.create_visualization(
                            text=template_text,
                            vis_type=template_vis_type
                        )
                        
                        # Display template result
                        if result['type'] == 'svg':
                            st.markdown(result['data'], unsafe_allow_html=True)
                        elif result['type'] == 'plotly':
                            st.plotly_chart(result['data'], use_container_width=True)
                        elif result['type'] == 'image':
                            st.image(result['data'])
                        
                        st.success(f"Generated {template_name} template!")
                        
                    except Exception as e:
                        st.error(f"Template generation failed: {e}")

with tab3:
    st.subheader("🔄 Batch Processing")
    
    st.info("Process multiple inputs at once to generate multiple visualizations")
    
    # Batch input methods
    batch_method = st.radio(
        "Batch Input Method",
        ["Text List", "File Upload", "CSV Data"]
    )
    
    if batch_method == "Text List":
        batch_text = st.text_area(
            "Enter multiple descriptions (one per line)",
            height=200,
            placeholder="Process 1: Customer registration flow\nProcess 2: Payment processing system\nProcess 3: Order fulfillment workflow"
        )
        
        if batch_text:
            lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
            st.write(f"Found {len(lines)} items to process")
            
            batch_vis_type = st.selectbox("Batch Visualization Type", 
                                        ["flowchart", "mindmap", "network", "auto-detect"])
            
            if st.button("🚀 Process Batch", type="primary"):
                if lines:
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    for i, line in enumerate(lines):
                        with results_container:
                            st.subheader(f"Result {i+1}: {line[:50]}...")
                            
                            try:
                                # Determine visualization type
                                vis_type = batch_vis_type
                                if batch_vis_type == "auto-detect":
                                    # Simple auto-detection logic
                                    if "flow" in line.lower() or "process" in line.lower():
                                        vis_type = "flowchart"
                                    elif "relationship" in line.lower() or "network" in line.lower():
                                        vis_type = "network"
                                    else:
                                        vis_type = "mindmap"
                                
                                result = visual_generator.create_visualization(
                                    text=line,
                                    vis_type=vis_type
                                )
                                
                                # Display result
                                if result['type'] == 'svg':
                                    st.markdown(result['data'], unsafe_allow_html=True)
                                elif result['type'] == 'plotly':
                                    st.plotly_chart(result['data'], use_container_width=True)
                                elif result['type'] == 'image':
                                    st.image(result['data'])
                                
                                # Download option
                                st.download_button(
                                    f"📥 Download Result {i+1}",
                                    result.get('data', ''),
                                    file_name=f"batch_result_{i+1}.svg",
                                    mime="image/svg+xml",
                                    key=f"download_{i}"
                                )
                                
                            except Exception as e:
                                st.error(f"Failed to process item {i+1}: {e}")
                            
                            progress_bar.progress((i + 1) / len(lines))
                            st.markdown("---")
                    
                    st.success(f"Batch processing completed! Generated {len(lines)} visualizations.")
    
    elif batch_method == "File Upload":
        uploaded_batch_file = st.file_uploader(
            "Upload text file with descriptions",
            type=['txt'],
            help="Upload a .txt file with one description per line"
        )
        
        if uploaded_batch_file:
            content = uploaded_batch_file.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.write(f"File contains {len(lines)} descriptions")
            st.text_area("Preview", '\n'.join(lines[:5]) + ('\n...' if len(lines) > 5 else ''), height=150)
            
            if st.button("🚀 Process File", type="primary"):
                st.info("File processing would happen here...")
    
    elif batch_method == "CSV Data":
        st.write("Upload a CSV file with columns: 'description', 'type' (optional)")
        
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have 'description' column and optional 'type' column"
        )
        
        if uploaded_csv:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_csv)
                
                st.dataframe(df.head())
                st.write(f"CSV contains {len(df)} rows")
                
                if 'description' in df.columns:
                    if st.button("🚀 Process CSV", type="primary"):
                        st.info("CSV processing would happen here...")
                else:
                    st.error("CSV must contain a 'description' column")
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

# Performance monitoring section
st.markdown("---")
st.subheader("📊 Performance Monitoring")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Visualizations Generated", "0", "0")

with col2:
    st.metric("Average Processing Time", "0.0s", "0.0s")

with col3:
    st.metric("Success Rate", "100%", "0%")
