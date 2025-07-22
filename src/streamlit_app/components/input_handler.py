
import streamlit as st
from typing import Dict, List, Any, Optional
from PIL import Image
import io
import base64

class InputHandler:
    """Handle various input types for the visual AI application"""
    
    def __init__(self):
        self.supported_image_formats = ['png', 'jpg', 'jpeg', 'bmp', 'gif']
        self.supported_text_formats = ['txt', 'md']
        self.supported_data_formats = ['csv', 'json']
    
    def handle_text_input(self, 
                         label: str = "Enter your text", 
                         height: int = 200,
                         placeholder: str = "Describe what you want to visualize...") -> Optional[str]:
        """Handle text area input"""
        text_input = st.text_area(
            label,
            height=height,
            placeholder=placeholder
        )
        
        if text_input:
            # Basic text validation
            word_count = len(text_input.split())
            char_count = len(text_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Words: {word_count}")
            with col2:
                st.caption(f"Characters: {char_count}")
            
            # Show warnings for very short or very long text
            if word_count < 3:
                st.warning("⚠️ Text is very short. Consider adding more details for better results.")
            elif word_count > 500:
                st.warning("⚠️ Text is very long. Consider breaking it into smaller parts.")
        
        return text_input if text_input else None
    
    def handle_image_upload(self, 
                           label: str = "Upload an image",
                           multiple: bool = False) -> Optional[Any]:
        """Handle image file upload"""
        uploaded_file = st.file_uploader(
            label,
            type=self.supported_image_formats,
            accept_multiple_files=multiple,
            help=f"Supported formats: {', '.join(self.supported_image_formats)}"
        )
        
        if uploaded_file:
            if multiple:
                # Handle multiple files
                images = []
                for file in uploaded_file:
                    try:
                        image = Image.open(file)
                        images.append({
                            'image': image,
                            'filename': file.name,
                            'size': image.size,
                            'format': image.format,
                            'mode': image.mode
                        })
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
                
                # Display preview of uploaded images
                if images:
                    st.write(f"Uploaded {len(images)} images:")
                    cols = st.columns(min(len(images), 4))
                    for i, img_data in enumerate(images[:4]):
                        with cols[i]:
                            st.image(img_data['image'], caption=img_data['filename'], use_column_width=True)
                    
                    if len(images) > 4:
                        st.write(f"... and {len(images) - 4} more images")
                
                return images
            
            else:
                # Handle single file
                try:
                    image = Image.open(uploaded_file)
                    
                    # Display image info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
                    
                    with col2:
                        st.write("**Image Info:**")
                        st.write(f"Format: {image.format}")
                        st.write(f"Size: {image.size}")
                        st.write(f"Mode: {image.mode}")
                        
                        # File size
                        file_size = len(uploaded_file.getvalue())
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024 * 1024:
                            size_str = f"{file_size / 1024:.1f} KB"
                        else:
                            size_str = f"{file_size / (1024 * 1024):.1f} MB"
                        st.write(f"File Size: {size_str}")
                    
                    return {
                        'image': image,
                        'filename': uploaded_file.name,
                        'size': image.size,
                        'format': image.format,
                        'mode': image.mode,
                        'file_size': file_size
                    }
                
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    return None
        
        return None
    
    def handle_file_upload(self, 
                          file_types: List[str],
                          label: str = "Upload file",
                          multiple: bool = False) -> Optional[Any]:
        """Handle general file upload"""
        uploaded_file = st.file_uploader(
            label,
            type=file_types,
            accept_multiple_files=multiple,
            help=f"Supported formats: {', '.join(file_types)}"
        )
        
        if uploaded_file:
            if multiple:
                files_data = []
                for file in uploaded_file:
                    try:
                        content = file.read()
                        files_data.append({
                            'filename': file.name,
                            'content': content,
                            'size': len(content),
                            'type': file.type
                        })
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                
                st.write(f"Uploaded {len(files_data)} files")
                return files_data
            
            else:
                try:
                    content = uploaded_file.read()
                    
                    # Show file info
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Size:** {len(content)} bytes")
                    st.write(f"**Type:** {uploaded_file.type}")
                    
                    return {
                        'filename': uploaded_file.name,
                        'content': content,
                        'size': len(content),
                        'type': uploaded_file.type
                    }
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    return None
        
        return None
    
    def handle_url_input(self, 
                        label: str = "Enter URL",
                        placeholder: str = "https://example.com") -> Optional[str]:
        """Handle URL input with validation"""
        url_input = st.text_input(
            label,
            placeholder=placeholder
        )
        
        if url_input:
            # Basic URL validation
            import re
            url_pattern = r'^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&=]*)$'
            
            if re.match(url_pattern, url_input):
                st.success("✅ Valid URL format")
                return url_input
            else:
                st.error("❌ Invalid URL format")
                return None
        
        return None
    
    def handle_json_input(self, 
                         label: str = "Enter JSON data",
                         height: int = 200) -> Optional[Dict]:
        """Handle JSON input with validation"""
        json_input = st.text_area(
            label,
            height=height,
            placeholder='{"key": "value", "array": [1, 2, 3]}'
        )
        
        if json_input:
            try:
                import json
                parsed_json = json.loads(json_input)
                st.success("✅ Valid JSON format")
                
                # Show JSON preview
                with st.expander("📋 JSON Preview"):
                    st.json(parsed_json)
                
                return parsed_json
            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
                return None
        
        return None
    
    def handle_multimodal_input(self) -> Dict[str, Any]:
        """Handle multiple input types simultaneously"""
        st.subheader("🔄 Multi-Modal Input")
        
        input_data = {}
        
        # Text input
        with st.expander("📝 Text Input", expanded=True):
            text = self.handle_text_input("Describe your visualization")
            if text:
                input_data['text'] = text
        
        # Image input
        with st.expander("🖼️ Image Input"):
            image = self.handle_image_upload("Upload reference image")
            if image:
                input_data['image'] = image
        
        # URL input
        with st.expander("🌐 URL Input"):
            url = self.handle_url_input("Reference URL")
            if url:
                input_data['url'] = url
        
        # JSON input
        with st.expander("📊 Structured Data Input"):
            json_data = self.handle_json_input("Additional structured data")
            if json_data:
                input_data['json'] = json_data
        
        # Show input summary
        if input_data:
            st.success(f"✅ Collected {len(input_data)} input types: {', '.join(input_data.keys())}")
        
        return input_data
    
    def validate_input_combination(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input combination makes sense"""
        if not input_data:
            st.warning("⚠️ No input provided")
            return False
        
        # At least text or image should be provided
        if 'text' not in input_data and 'image' not in input_data:
            st.warning("⚠️ Please provide either text description or image")
            return False
        
        # Check text length if provided
        if 'text' in input_data:
            text_length = len(input_data['text'].split())
            if text_length < 3:
                st.warning("⚠️ Text description is too short")
                return False
        
        return True
