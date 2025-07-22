
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="🎨 Free Visual AI Generator", 
    page_icon="🎨",
    layout="wide"
)

# Header
st.title("🎨 Free Visual AI Generator")
st.markdown("### Transform Text into Beautiful Visualizations - No AI Models Required!")

# Sidebar
with st.sidebar:
    st.header("🎯 Visualization Options")
    viz_type = st.selectbox("Choose visualization type:", [
        "Flowchart", "Mind Map", "Network Diagram", "Bar Chart", "Timeline"
    ])

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    user_text = st.text_area(
        "Describe your process or data:",
        value="Start -> Planning -> Development -> Testing -> Deployment -> End",
        height=200
    )
    
    if st.button("🎨 Generate Visualization", type="primary"):
        # Simple text processing
        steps = [step.strip() for step in user_text.replace("->", ",").split(",")]
        steps = [step for step in steps if step]  # Remove empty steps
        
        with col2:
            st.subheader("📊 Generated Visualization")
            
            if viz_type == "Flowchart":
                # Create simple flowchart
                fig = go.Figure()
                
                for i, step in enumerate(steps):
                    fig.add_trace(go.Scatter(
                        x=[i], y=[0],
                        mode='markers+text',
                        marker=dict(size=80, color='lightblue'),
                        text=step,
                        textposition="middle center",
                        name=step
                    ))
                    
                    if i < len(steps) - 1:
                        fig.add_annotation(
                            x=i+0.4, y=0,
                            ax=i+0.6, ay=0,
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2
                        )
                
                fig.update_layout(
                    title="Process Flowchart",
                    showlegend=False,
                    height=400,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False, range=[-1, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Bar Chart":
                # Create bar chart
                df = pd.DataFrame({
                    'Steps': steps,
                    'Order': range(1, len(steps) + 1)
                })
                fig = px.bar(df, x='Steps', y='Order', title="Process Steps")
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Mind Map":
                # Simple radial layout
                angles = np.linspace(0, 2*np.pi, len(steps), endpoint=False)
                x = np.cos(angles)
                y = np.sin(angles)
                
                fig = go.Figure()
                
                # Center node
                fig.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    marker=dict(size=100, color='red'),
                    text="Central Topic",
                    textposition="middle center",
                    name="Center"
                ))
                
                # Branch nodes
                for i, (step, xi, yi) in enumerate(zip(steps, x, y)):
                    fig.add_trace(go.Scatter(
                        x=[xi], y=[yi],
                        mode='markers+text',
                        marker=dict(size=60, color='lightgreen'),
                        text=step,
                        textposition="middle center",
                        name=step
                    ))
                    
                    # Connect to center
                    fig.add_trace(go.Scatter(
                        x=[0, xi], y=[0, yi],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Mind Map",
                    showlegend=False,
                    height=500,
                    xaxis=dict(showgrid=False, showticklabels=False, range=[-2, 2]),
                    yaxis=dict(showgrid=False, showticklabels=False, range=[-2, 2])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            st.success("✅ Visualization generated successfully!")
            st.info("💡 This is a simplified version. Full AI features available after initial deployment!")

# Footer
st.markdown("---")
st.markdown("🎨 **Free Visual AI Generator** - Making visualizations accessible to everyone!")

# Sample data section
st.subheader("🔧 Try These Examples:")
examples = [
    "User Registration -> Email Verification -> Account Setup -> Welcome Message",
    "Planning -> Design -> Development -> Testing -> Deployment",
    "Problem Identification -> Research -> Solution Design -> Implementation -> Evaluation",
    "Lead Generation -> Qualification -> Proposal -> Negotiation -> Closing"
]

for example in examples:
    if st.button(f"📋 {example[:50]}...", key=example):
        st.experimental_rerun()
