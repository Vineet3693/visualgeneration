
import streamlit as st
import sys
import os
import pandas as pd
from typing import List, Dict
import time
import zipfile
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model_loader import LocalModelManager
from src.core.visual_generator import FreeVisualGenerator

st.set_page_config(
    page_title="Batch Processing",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Batch Visualization Processing")
st.markdown("Process multiple inputs efficiently and download results in bulk")

# Initialize models
@st.cache_resource
def get_models():
    return LocalModelManager()

model_manager = get_models()
visual_generator = FreeVisualGenerator(model_manager)

# Batch processing functions
def process_batch_items(items: List[Dict], vis_type: str = "auto") -> List[Dict]:
    """Process multiple items and return results"""
    results = []
    
    for i, item in enumerate(items):
        try:
            # Auto-detect visualization type if needed
            if vis_type == "auto":
                detected_type = visual_generator.text_processor.determine_visualization_type(item['text'])
            else:
                detected_type = vis_type
            
            result = visual_generator.create_visualization(
                text=item['text'],
                vis_type=detected_type
            )
            
            result['original_text'] = item['text']
            result['item_index'] = i
            result['processing_time'] = time.time() - item.get('start_time', time.time())
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'type': 'error',
                'error': str(e),
                'original_text': item['text'],
                'item_index': i
            })
    
    return results

def create_download_zip(results: List[Dict]) -> bytes:
    """Create a zip file with all generated visualizations"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, result in enumerate(results):
            if result['type'] == 'svg':
                filename = f"visualization_{i+1}.svg"
                zip_file.writestr(filename, result['data'])
            elif result['type'] == 'image':
                filename = f"visualization_{i+1}.png"
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                result['data'].save(img_buffer, format='PNG')
                zip_file.writestr(filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Main interface
tab1, tab2, tab3 = st.tabs(["📋 Input Setup", "⚡ Processing", "📥 Results"])

with tab1:
    st.subheader("Setup Batch Processing")
    
    # Input method selection
    input_method = st.selectbox(
        "Choose Input Method",
        [
            "Text Area (Multiple Lines)",
            "File Upload (.txt)",
            "CSV File",
            "JSON File"
        ]
    )
    
    batch_items = []
    
    if input_method == "Text Area (Multiple Lines)":
        batch_input = st.text_area(
            "Enter multiple descriptions (one per line)",
            height=300,
            placeholder="""Customer registration process: user enters email, verifies account, sets password, completes profile
Payment workflow: select items, enter payment details, process transaction, send confirmation
Order fulfillment: receive order, check inventory, pack items, ship to customer, track delivery
User onboarding: welcome screen, tutorial walkthrough, setup preferences, first interaction
Bug reporting: user reports issue, developer triages, assigns priority, fixes bug, deploys solution"""
        )
        
        if batch_input:
            lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
            batch_items = [{'text': line, 'source': 'text_area'} for line in lines]
            
            st.success(f"✅ Found {len(batch_items)} items to process")
    
    elif input_method == "File Upload (.txt)":
        uploaded_txt = st.file_uploader(
            "Upload text file",
            type=['txt'],
            help="One description per line"
        )
        
        if uploaded_txt:
            content = uploaded_txt.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            batch_items = [{'text': line, 'source': 'file_upload'} for line in lines]
            
            st.success(f"✅ Loaded {len(batch_items)} items from file")
            
            # Preview
            with st.expander("📋 Preview Items"):
                for i, item in enumerate(batch_items[:10]):
                    st.write(f"{i+1}. {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
                if len(batch_items) > 10:
                    st.write(f"... and {len(batch_items) - 10} more items")
    
    elif input_method == "CSV File":
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have columns: 'description' (required), 'type' (optional), 'title' (optional)"
        )
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                
                if 'description' not in df.columns:
                    st.error("❌ CSV must contain a 'description' column")
                else:
                    st.dataframe(df.head(10))
                    
                    # Convert to batch items
                    for _, row in df.iterrows():
                        item = {
                            'text': row['description'],
                            'source': 'csv',
                            'title': row.get('title', f"Item {len(batch_items) + 1}"),
                            'preferred_type': row.get('type', 'auto')
                        }
                        batch_items.append(item)
                    
                    st.success(f"✅ Loaded {len(batch_items)} items from CSV")
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
    
    elif input_method == "JSON File":
        uploaded_json = st.file_uploader(
            "Upload JSON file",
            type=['json'],
            help="JSON array with objects containing 'text' field"
        )
        
        if uploaded_json:
            try:
                import json
                data = json.load(uploaded_json)
                
                if isinstance(data, list):
                    for item in data:
                        if 'text' in item:
                            batch_items.append({
                                'text': item['text'],
                                'source': 'json',
                                'title': item.get('title', f"Item {len(batch_items) + 1}"),
                                'preferred_type': item.get('type', 'auto')
                            })
                    
                    st.success(f"✅ Loaded {len(batch_items)} items from JSON")
                else:
                    st.error("❌ JSON should contain an array of objects")
                    
            except Exception as e:
                st.error(f"❌ Error reading JSON: {e}")
    
    # Processing settings
    if batch_items:
        st.markdown("---")
        st.subheader("⚙️ Processing Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_vis_type = st.selectbox(
                "Default Visualization Type",
                ["auto", "flowchart", "mindmap", "network", "infographic"],
                help="Type to use when not specified per item"
            )
        
        with col2:
            batch_size = st.slider(
                "Batch Size",
                1, min(20, len(batch_items)), 
                min(5, len(batch_items)),
                help="Number of items to process at once"
            )
        
        with col3:
            include_metadata = st.checkbox(
                "Include Metadata",
                True,
                help="Include processing details in results"
            )
        
        # Store settings in session state
        st.session_state.batch_settings = {
            'items': batch_items,
            'vis_type': default_vis_type,
            'batch_size': batch_size,
            'include_metadata': include_metadata
        }

with tab2:
    st.subheader("⚡ Batch Processing")
    
    if 'batch_settings' in st.session_state:
        settings = st.session_state.batch_settings
        items = settings['items']
        
        st.info(f"Ready to process {len(items)} items with {settings['vis_type']} visualization type")
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_processing = st.button("🚀 Start Batch Processing", type="primary")
        
        with col2:
            if 'processing_results' in st.session_state:
                st.button("⏹️ Stop Processing", disabled=True)  # Placeholder for stop functionality
        
        with col3:
            if 'processing_results' in st.session_state:
                st.button("🔄 Reset Results", key="reset_results")
        
        # Processing execution
        if start_processing:
            st.session_state.processing_started = True
            st.session_state.processing_results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_preview = st.empty()
            
            # Process items in batches
            total_items = len(items)
            batch_size = settings['batch_size']
            
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                current_batch = items[batch_start:batch_end]
                
                status_text.text(f"Processing items {batch_start + 1}-{batch_end} of {total_items}")
                
                # Process current batch
                for i, item in enumerate(current_batch):
                    global_index = batch_start + i
                    
                    try:
                        # Add start time for performance tracking
                        item['start_time'] = time.time()
                        
                        # Determine visualization type
                        vis_type = item.get('preferred_type', settings['vis_type'])
                        if vis_type == 'auto':
                            vis_type = visual_generator.text_processor.determine_visualization_type(item['text'])
                        
                        # Generate visualization
                        result = visual_generator.create_visualization(
                            text=item['text'],
                            vis_type=vis_type
                        )
                        
                        # Add metadata
                        if settings['include_metadata']:
                            result['metadata'] = {
                                'original_text': item['text'],
                                'processing_time': time.time() - item['start_time'],
                                'visualization_type': vis_type,
                                'source': item.get('source', 'unknown'),
                                'title': item.get('title', f'Item {global_index + 1}'),
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        
                        st.session_state.processing_results.append(result)
                        
                    except Exception as e:
                        error_result = {
                            'type': 'error',
                            'error': str(e),
                            'metadata': {
                                'original_text': item['text'],
                                'processing_time': time.time() - item.get('start_time', time.time()),
                                'visualization_type': 'error',
                                'source': item.get('source', 'unknown'),
                                'title': item.get('title', f'Item {global_index + 1}'),
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        }
                        st.session_state.processing_results.append(error_result)
                    
                    # Update progress
                    progress = (global_index + 1) / total_items
                    progress_bar.progress(progress)
                    
                    # Show preview of latest result
                    if st.session_state.processing_results:
                        latest_result = st.session_state.processing_results[-1]
                        with results_preview.container():
                            st.write(f"**Latest:** {latest_result.get('metadata', {}).get('title', 'Unknown')}")
                            if latest_result['type'] == 'error':
                                st.error(f"Error: {latest_result['error']}")
            
            status_text.success("✅ Batch processing completed!")
            
            # Processing summary
            results = st.session_state.processing_results
            successful = len([r for r in results if r['type'] != 'error'])
            failed = len([r for r in results if r['type'] == 'error'])
            
            st.success(f"Processing Summary: {successful} successful, {failed} failed out of {total_items} total")
    
    else:
        st.info("👆 Please set up your batch items in the 'Input Setup' tab first")

with tab3:
    st.subheader("📥 Processing Results")
    
    if 'processing_results' in st.session_state:
        results = st.session_state.processing_results
        
        if results:
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            successful = len([r for r in results if r['type'] != 'error'])
            failed = len([r for r in results if r['type'] == 'error'])
            avg_time = sum([r.get('metadata', {}).get('processing_time', 0) for r in results]) / len(results)
            
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                st.metric("Successful", successful, f"{successful/len(results)*100:.1f}%")
            with col3:
                st.metric("Failed", failed, f"-{failed/len(results)*100:.1f}%" if failed > 0 else "0%")
            with col4:
                st.metric("Avg Time", f"{avg_time:.2f}s")
            
            # Download options
            st.markdown("---")
            st.subheader("📥 Download Options")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("📦 Download All as ZIP"):
                    successful_results = [r for r in results if r['type'] != 'error']
                    if successful_results:
                        zip_data = create_download_zip(successful_results)
                        st.download_button(
                            "📥 Download ZIP File",
                            zip_data,
                            file_name=f"batch_visualizations_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    else:
                        st.error("No successful results to download")
            
            with col_b:
                if st.button("📊 Export Metadata CSV"):
                    metadata_list = []
                    for i, result in enumerate(results):
                        metadata = result.get('metadata', {})
                        metadata['result_index'] = i + 1
                        metadata['status'] = 'success' if result['type'] != 'error' else 'error'
                        if result['type'] == 'error':
                            metadata['error_message'] = result.get('error', 'Unknown error')
                        metadata_list.append(metadata)
                    
                    df = pd.DataFrame(metadata_list)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        file_name=f"batch_metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col_c:
                if st.button("📋 Generate Report"):
                    # Generate a processing report
                    report = f"""
# Batch Processing Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Items Processed: {len(results)}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {successful/len(results)*100:.1f}%
- Average Processing Time: {avg_time:.2f}s

## Detailed Results
"""
                    
                    for i, result in enumerate(results):
                        metadata = result.get('metadata', {})
                        status = '✅ Success' if result['type'] != 'error' else '❌ Error'
                        report += f"\n{i+1}. {metadata.get('title', f'Item {i+1}')} - {status}"
                        if result['type'] == 'error':
                            report += f"\n   Error: {result.get('error', 'Unknown error')}"
                        report += f"\n   Processing Time: {metadata.get('processing_time', 0):.2f}s"
                        report += f"\n   Type: {metadata.get('visualization_type', 'unknown')}\n"
                    
                    st.download_button(
                        "📥 Download Report",
                        report,
                        file_name=f"batch_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            
            # Individual results display
            st.markdown("---")
            st.subheader("🔍 Individual Results")
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                show_successful = st.checkbox("Show Successful", True)
                show_errors = st.checkbox("Show Errors", True)
            
            with filter_col2:
                result_type_filter = st.multiselect(
                    "Filter by Type",
                    ["svg", "plotly", "image", "error"],
                    default=["svg", "plotly", "image", "error"]
                )
            
            # Display filtered results
            filtered_results = []
            for result in results:
                if result['type'] == 'error' and not show_errors:
                    continue
                if result['type'] != 'error' and not show_successful:
                    continue
                if result['type'] not in result_type_filter:
                    continue
                filtered_results.append(result)
            
            st.write(f"Showing {len(filtered_results)} of {len(results)} results")
            
            # Paginated display
            items_per_page = 5
            total_pages = (len(filtered_results) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
            else:
                page = 0
            
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_results))
            
            for i in range(start_idx, end_idx):
                result = filtered_results[i]
                metadata = result.get('metadata', {})
                
                with st.expander(f"Result {i+1}: {metadata.get('title', f'Item {i+1}')}"):
                    if result['type'] == 'error':
                        st.error(f"Processing failed: {result['error']}")
                        st.write(f"**Original Text:** {metadata.get('original_text', 'N/A')}")
                    else:
                        # Display visualization
                        if result['type'] == 'svg':
                            st.markdown(result['data'], unsafe_allow_html=True)
                        elif result['type'] == 'plotly':
                            st.plotly_chart(result['data'], use_container_width=True)
                        elif result['type'] == 'image':
                            st.image(result['data'])
                        
                        # Individual download
                        if result['type'] in ['svg', 'image']:
                            file_ext = 'svg' if result['type'] == 'svg' else 'png'
                            filename = f"{metadata.get('title', f'item_{i+1}')}.{file_ext}"
                            
                            if result['type'] == 'svg':
                                st.download_button(
                                    f"📥 Download {filename}",
                                    result['data'],
                                    file_name=filename,
                                    mime=f"image/{file_ext}",
                                    key=f"download_individual_{i}"
                                )
                            elif result['type'] == 'image':
                                img_buffer = io.BytesIO()
                                result['data'].save(img_buffer, format='PNG')
                                st.download_button(
                                    f"📥 Download {filename}",
                                    img_buffer.getvalue(),
                                    file_name=filename,
                                    mime="image/png",
                                    key=f"download_individual_{i}"
                                )
                    
                    # Show metadata
                    if metadata:
                        st.json(metadata)
        
        else:
            st.info("No results yet. Start processing in the 'Processing' tab.")
    
    else:
        st.info("No processing has been started yet.")
