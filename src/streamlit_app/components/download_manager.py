
import streamlit as st
import zipfile
import io
import json
import csv
from typing import List, Dict, Any
from datetime import datetime
import base64

class DownloadManager:
    """Manage downloads for visualizations and data"""
    
    def __init__(self):
        self.supported_formats = {
            'svg': {'mime': 'image/svg+xml', 'extension': '.svg'},
            'png': {'mime': 'image/png', 'extension': '.png'},
            'html': {'mime': 'text/html', 'extension': '.html'},
            'json': {'mime': 'application/json', 'extension': '.json'},
            'csv': {'mime': 'text/csv', 'extension': '.csv'},
            'txt': {'mime': 'text/plain', 'extension': '.txt'},
            'zip': {'mime': 'application/zip', 'extension': '.zip'}
        }
    
    def create_single_download(self, 
                             data: Any, 
                             filename: str, 
                             format_type: str,
                             label: str = None) -> None:
        """Create download button for single item"""
        
        if format_type not in self.supported_formats:
            st.error(f"Unsupported format: {format_type}")
            return
        
        format_info = self.supported_formats[format_type]
        
        # Ensure filename has correct extension
        if not filename.endswith(format_info['extension']):
            filename += format_info['extension']
        
        # Prepare data based on type
        if isinstance(data, str):
            download_data = data.encode('utf-8') if format_type in ['svg', 'html', 'json', 'csv', 'txt'] else data
        else:
            download_data = data
        
        # Create download button
        button_label = label or f"📥 Download {format_type.upper()}"
        
        st.download_button(
            label=button_label,
            data=download_data,
            file_name=filename,
            mime=format_info['mime'],
            key=f"download_{filename}_{hash(str(data))}"
        )
    
    def create_bulk_download(self, 
                           results: List[Dict[str, Any]], 
                           filename: str = None) -> None:
        """Create ZIP download for multiple visualizations"""
        
        if not results:
            st.warning("No results to download")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualizations_bulk_{timestamp}.zip"
        
        try:
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # Add each visualization to ZIP
                for i, result in enumerate(results):
                    if result.get('type') == 'error':
                        continue
                    
                    # Determine filename for this item
                    metadata = result.get('metadata', {})
                    item_title = metadata.get('title', f'visualization_{i+1}')
                    
                    # Clean filename
                    clean_title = "".join(c for c in item_title if c.isalnum() or c in (' ', '-', '_')).strip()
                    clean_title = clean_title.replace(' ', '_')
                    
                    # Add visualization file
                    if result['type'] == 'svg':
                        zip_file.writestr(f"{clean_title}.svg", result['data'])
                    
                    elif result['type'] == 'image':
                        # Convert PIL image to bytes
                        img_buffer = io.BytesIO()
                        result['data'].save(img_buffer, format='PNG')
                        zip_file.writestr(f"{clean_title}.png", img_buffer.getvalue())
                    
                    elif result['type'] == 'plotly':
                        # Save as HTML
                        html_str = result['data'].to_html(include_plotlyjs='cdn')
                        zip_file.writestr(f"{clean_title}.html", html_str)
                    
                    # Add metadata file
                    if metadata:
                        metadata_json = json.dumps(metadata, indent=2)
                        zip_file.writestr(f"{clean_title}_metadata.json", metadata_json)
                
                # Add summary file
                summary = self.create_bulk_summary(results)
                zip_file.writestr("summary.txt", summary)
            
            zip_buffer.seek(0)
            
            # Create download button
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name=filename,
                mime=self.supported_formats['zip']['mime']
            )
            
        except Exception as e:
            st.error(f"Error creating bulk download: {e}")
    
    def create_bulk_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create summary text for bulk download"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        successful = len([r for r in results if r.get('type') != 'error'])
        failed = len([r for r in results if r.get('type') == 'error'])
        
        summary = f"""Bulk Visualization Download Summary
Generated: {timestamp}

Total Items: {len(results)}
Successful: {successful}
Failed: {failed}
Success Rate: {(successful/len(results)*100):.1f}%

File Contents:
"""
        
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            title = metadata.get('title', f'Item {i+1}')
            status = '✅ Success' if result.get('type') != 'error' else '❌ Error'
            vis_type = metadata.get('visualization_type', result.get('type', 'unknown'))
            
            summary += f"\n{i+1}. {title} - {status} ({vis_type})"
            
            if result.get('type') == 'error':
                summary += f"\n   Error: {result.get('error', 'Unknown error')}"
        
        summary += f"\n\nGenerated by Free Visual AI Generator"
        
        return summary
    
    def create_data_export(self, 
                          data: List[Dict], 
                          format_type: str,
                          filename: str = None) -> None:
        """Export data in various formats"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}"
        
        try:
            if format_type == 'csv':
                # Convert to CSV
                if data:
                    # Flatten nested dictionaries for CSV
                    flattened_data = []
                    for item in data:
                        flat_item = {}
                        for key, value in item.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    flat_item[f"{key}_{sub_key}"] = sub_value
                            else:
                                flat_item[key] = value
                        flattened_data.append(flat_item)
                    
                    # Create CSV string
                    output = io.StringIO()
                    if flattened_data:
                        fieldnames = flattened_data[0].keys()
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened_data)
                    
                    self.create_single_download(
                        output.getvalue(),
                        filename,
                        'csv',
                        "📊 Download CSV"
                    )
                
            elif format_type == 'json':
                # Convert to JSON
                json_str = json.dumps(data, indent=2, default=str)
                self.create_single_download(
                    json_str,
                    filename,
                    'json',
                    "📋 Download JSON"
                )
            
            elif format_type == 'txt':
                # Convert to text format
                text_content = ""
                for i, item in enumerate(data):
                    text_content += f"Item {i+1}:\n"
                    for key, value in item.items():
                        text_content += f"  {key}: {value}\n"
                    text_content += "\n"
                
                self.create_single_download(
                    text_content,
                    filename,
                    'txt',
                    "📄 Download TXT"
                )
            
        except Exception as e:
            st.error(f"Error creating data export: {e}")
    
    def create_report_download(self, 
                             results: List[Dict[str, Any]], 
                             include_details: bool = True) -> None:
        """Create a detailed report for download"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate report content
        report = f"""# Visual AI Generation Report
Generated: {timestamp}

## Summary Statistics
"""
        
        # Calculate statistics
        total = len(results)
        successful = len([r for r in results if r.get('type') != 'error'])
        failed = total - successful
        
        if total > 0:
            success_rate = (successful / total) * 100
            avg_time = sum([r.get('metadata', {}).get('processing_time', 0) for r in results]) / total
        else:
            success_rate = 0
            avg_time = 0
        
        report += f"""
- Total Visualizations: {total}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {success_rate:.1f}%
- Average Processing Time: {avg_time:.2f}s
"""
        
        # Type breakdown
        type_counts = {}
        for result in results:
            if result.get('type') != 'error':
                vis_type = result.get('metadata', {}).get('visualization_type', 'unknown')
                type_counts[vis_type] = type_counts.get(vis_type, 0) + 1
        
        if type_counts:
            report += "\n## Visualization Types\n"
            for vis_type, count in sorted(type_counts.items()):
                report += f"- {vis_type}: {count}\n"
        
        # Detailed results
        if include_details:
            report += "\n## Detailed Results\n"
            
            for i, result in enumerate(results):
                metadata = result.get('metadata', {})
                title = metadata.get('title', f'Item {i+1}')
                
                report += f"\n### {i+1}. {title}\n"
                
                if result.get('type') == 'error':
                    report += f"- Status: ❌ Failed\n"
                    report += f"- Error: {result.get('error', 'Unknown error')}\n"
                else:
                    report += f"- Status: ✅ Success\n"
                    report += f"- Type: {metadata.get('visualization_type', 'unknown')}\n"
                    report += f"- Processing Time: {metadata.get('processing_time', 0):.2f}s\n"
                
                if 'original_text' in metadata:
                    original = metadata['original_text'][:200] + "..." if len(metadata['original_text']) > 200 else metadata['original_text']
                    report += f"- Input: {original}\n"
        
        report += "\n---\nGenerated by Free Visual AI Generator"
        
        # Create download
        self.create_single_download(
            report,
            f"visual_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'txt',
            "📄 Download Report"
        )
    
    def create_custom_export_options(self, results: List[Dict[str, Any]]) -> None:
        """Create custom export options interface"""
        st.subheader("📥 Export Options")
        
        if not results:
            st.info("No results to export")
            return
        
        # Export format selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Individual Downloads:**")
            export_format = st.selectbox(
                "Format",
                ["svg", "png", "html", "json"]
            )
            
            if st.button("📋 Create Individual Downloads"):
                for i, result in enumerate(results):
                    if result.get('type') != 'error':
                        metadata = result.get('metadata', {})
                        title = metadata.get('title', f'visualization_{i+1}')
                        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                        
                        try:
                            if export_format == 'svg' and result['type'] == 'svg':
                                self.create_single_download(result['data'], clean_title, 'svg')
                            elif export_format == 'png' and result['type'] == 'image':
                                img_buffer = io.BytesIO()
                                result['data'].save(img_buffer, format='PNG')
                                self.create_single_download(img_buffer.getvalue(), clean_title, 'png')
                            elif export_format == 'html' and result['type'] == 'plotly':
                                html_str = result['data'].to_html(include_plotlyjs='cdn')
                                self.create_single_download(html_str, clean_title, 'html')
                            elif export_format == 'json':
                                json_str = json.dumps(metadata, indent=2, default=str)
                                self.create_single_download(json_str, f"{clean_title}_metadata", 'json')
                        except Exception as e:
                            st.error(f"Error creating download for {title}: {e}")
        
        with col2:
            st.write("**Bulk Downloads:**")
            
            include_metadata = st.checkbox("Include Metadata", True)
            create_summary = st.checkbox("Include Summary", True)
            
            if st.button("📦 Create ZIP Package"):
                self.create_bulk_download(results)
        
        with col3:
            st.write("**Data Exports:**")
            
            data_format = st.selectbox(
                "Data Format",
                ["csv", "json", "txt"]
            )
            
            if st.button("📊 Export Data"):
                # Extract metadata for export
                export_data = []
                for i, result in enumerate(results):
                    metadata = result.get('metadata', {})
                    export_item = {
                        'index': i + 1,
                        'title': metadata.get('title', f'Item {i+1}'),
                        'status': 'success' if result.get('type') != 'error' else 'error',
                        'type': metadata.get('visualization_type', result.get('type', 'unknown')),
                        'processing_time': metadata.get('processing_time', 0),
                        'timestamp': metadata.get('timestamp', ''),
                        'source': metadata.get('source', ''),
                    }
                    
                    if result.get('type') == 'error':
                        export_item['error_message'] = result.get('error', 'Unknown error')
                    
                    export_data.append(export_item)
                
                self.create_data_export(export_data, data_format)
        
        # Advanced export options
        with st.expander("⚙️ Advanced Export Options"):
            
            # Filter options
            st.write("**Filter Results:**")
            
            filter_successful = st.checkbox("Only Successful", True)
            filter_by_type = st.multiselect(
                "Filter by Type",
                ["flowchart", "mindmap", "network", "infographic", "tree"],
                default=[]
            )
            
            # Custom filename
            custom_filename = st.text_input(
                "Custom Filename (optional)",
                placeholder="my_visualizations"
            )
            
            if st.button("📋 Create Filtered Export"):
                # Apply filters
                filtered_results = results.copy()
                
                if filter_successful:
                    filtered_results = [r for r in filtered_results if r.get('type') != 'error']
                
                if filter_by_type:
                    filtered_results = [
                        r for r in filtered_results 
                        if r.get('metadata', {}).get('visualization_type') in filter_by_type
                    ]
                
                if filtered_results:
                    filename = custom_filename if custom_filename else None
                    self.create_bulk_download(filtered_results, filename)
                else:
                    st.warning("No results match the selected filters")
