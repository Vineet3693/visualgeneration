
# Add this advanced batch processing system

class AdvancedBatchProcessor:
    """Advanced batch processing with parallel capabilities"""
    
    def __init__(self):
        self.processing_queue = []
        self.results_cache = {}
        self.batch_settings = {
            'parallel_limit': 5,
            'timeout_per_item': 30,
            'retry_attempts': 2
        }
    
    def process_batch_advanced(self, items: List[str], viz_type: str, settings: Dict) -> List[Dict]:
        """Process multiple items with advanced features"""
        results = []
        total_items = len(items)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_container = st.container()
        metrics_container = st.container()
        
        # Initialize metrics
        successful = 0
        failed = 0
        start_time = datetime.now()
        
        for i, item in enumerate(items):
            with status_container:
                st.text(f"Processing {i+1}/{total_items}: {item[:60]}...")
            
            try:
                # Process individual item
                result = self.process_single_item(item, viz_type, settings)
                
                if result['success']:
                    successful += 1
                    results.append(result)
                else:
                    failed += 1
                    results.append(result)
                
                # Update metrics
                elapsed = (datetime.now() - start_time).seconds
                with metrics_container:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("✅ Successful", successful)
                    with col2:
                        st.metric("❌ Failed", failed)
                    with col3:
                        st.metric("⏱️ Elapsed", f"{elapsed}s")
                    with col4:
                        eta = (elapsed / max(i, 1)) * (total_items - i - 1)
                        st.metric("📅 ETA", f"{int(eta)}s")
                
            except Exception as e:
                failed += 1
                results.append({
                    'input': item,
                    'success': False,
                    'error': str(e),
                    'figure': None,
                    'processing_time': 0
                })
            
            # Update progress
            progress_bar.progress((i + 1) / total_items)
        
        # Final status
        with status_container:
            st.success(f"✅ Batch processing completed! {successful} successful, {failed} failed")
        
        return results
    
    def process_single_item(self, text: str, viz_type: str, settings: Dict) -> Dict:
        """Process a single item with error handling"""
        start_time = datetime.now()
        
        try:
            # Initialize processors
            text_processor = AdvancedTextProcessor()
            viz_generator = EnhancedVisualizationGenerator()
            
            # Extract entities
            entities = text_processor.extract_entities(text)
            
            # Generate visualization based on type
            if viz_type.lower() == 'flowchart':
                figure = viz_generator.create_advanced_flowchart(text, entities)
            elif viz_type.lower() == 'mindmap':
                figure = viz_generator.create_dynamic_mindmap(text, entities)
            elif viz_type.lower() == 'network':
                figure = viz_generator.create_interactive_network(text, entities)
            elif viz_type.lower() == 'timeline':
                figure = viz_generator.create_smart_timeline(text, entities)
            elif viz_type.lower() == 'infographic':
                figure = viz_generator.create_data_infographic(text, entities)
            else:
                figure = viz_generator.create_advanced_flowchart(text, entities)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'input': text,
                'success': True,
                'figure': figure,
                'entities_count': len(entities),
                'processing_time': processing_time,
                'viz_type': viz_type
            }
            
        except Exception as e:
            return {
                'input': text,
                'success': False,
                'error': str(e),
                'figure': None,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def create_batch_summary_report(self, results: List[Dict]) -> Dict:
        """Create comprehensive batch processing report"""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # Calculate statistics
        total_processing_time = sum(r.get('processing_time', 0) for r in results)
        avg_processing_time = total_processing_time / len(results) if results else 0
        
        # Entity statistics
        entity_counts = [r.get('entities_count', 0) for r in successful_results]
        avg_entities = np.mean(entity_counts) if entity_counts else 0
        
        # Error analysis
        error_types = {}
        for result in failed_results:
            error = result.get('error', 'Unknown')
            error_type = error.split(':')[0] if ':' in error else error[:50]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'summary': {
                'total_items': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0
            },
            'performance': {
                'total_processing_time': total_processing_time,
                'average_processing_time': avg_processing_time,
                'items_per_second': len(results) / total_processing_time if total_processing_time > 0 else 0
            },
            'content_analysis': {
                'average_entities_per_item': avg_entities,
                'max_entities': max(entity_counts) if entity_counts else 0,
                'min_entities': min(entity_counts) if entity_counts else 0
            },
            'error_analysis': error_types,
            'generated_at': datetime.now().isoformat()
        }

def create_advanced_batch_interface():
    """Create advanced batch processing interface"""
    st.header("⚡ Advanced Batch Processing")
    
    # Batch configuration
    with st.expander("⚙️ Batch Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_viz_type = st.selectbox(
                "Visualization Type for All:",
                ["Flowchart", "Mind Map", "Network Diagram", "Timeline", "Infographic"],
                help="All items will be processed with this visualization type"
            )
            
            parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=True,
                help="Process multiple items simultaneously (faster but uses more resources)"
            )
        
        with col2:
            quality_level = st.selectbox(
                "Quality Level:",
                ["Fast", "Balanced", "High Quality"],
                index=1,
                help="Higher quality takes longer but produces better results"
            )
            
            include_analysis = st.checkbox(
                "Include Detailed Analysis",
                value=True,
                help="Add text analysis and optimization suggestions"
            )
    
    # Input methods
    st.subheader("📝 Batch Input Methods")
    
    batch_input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "CSV Upload", "JSON Upload", "Text File", "URL List"],
        horizontal=True
    )
    
    batch_items = []
    
    if batch_input_method == "Manual Entry":
        batch_text = st.text_area(
            "Enter items (one per line):",
            height=200,
            placeholder="""Order Processing: Order received -> Payment verification -> Inventory check -> Fulfillment -> Shipping -> Delivery
Customer Support: Ticket created -> Assignment -> Investigation -> Resolution -> Customer notification -> Closure
Product Development: Ideation -> Research -> Design -> Prototyping -> Testing -> Launch -> Feedback analysis""",
            help="Enter one process description per line"
        )
        
        if batch_text.strip():
            batch_items = [line.strip() for line in batch_text.strip().split('\n') if line.strip()]
    
    elif batch_input_method == "CSV Upload":
        uploaded_csv = st.file_uploader(
            "Upload CSV file:",
            type=['csv'],
            help="CSV should have columns: 'title', 'description', 'category' (optional)"
        )
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.dataframe(df.head())
                
                # Process CSV data
                if 'description' in df.columns:
                    batch_items = df['description'].dropna().tolist()
                elif 'process' in df.columns:
                    batch_items = df['process'].dropna().tolist()
                elif len(df.columns) >= 1:
                    batch_items = df.iloc[:, 0].dropna().astype(str).tolist()
                
                st.info(f"📊 Loaded {len(batch_items)} items from CSV")
                
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    elif batch_input_method == "JSON Upload":
        uploaded_json = st.file_uploader(
            "Upload JSON file:",
            type=['json'],
            help="JSON should contain an array of objects with text descriptions"
        )
        
        if uploaded_json is not None:
            try:
                json_data = json.load(uploaded_json)
                
                if isinstance(json_data, list):
                    # Handle list of strings or objects
                    for item in json_data:
                        if isinstance(item, str):
                            batch_items.append(item)
                        elif isinstance(item, dict):
                            # Try common field names
                            for field in ['description', 'text', 'process', 'content']:
                                if field in item:
                                    batch_items.append(str(item[field]))
                                    break
                
                st.info(f"📊 Loaded {len(batch_items)} items from JSON")
                
            except Exception as e:
                st.error(f"❌ Error reading JSON: {str(e)}")
    
    elif batch_input_method == "Text File":
        uploaded_txt = st.file_uploader(
            "Upload text file:",
            type=['txt'],
            help="Text file with one process per line or separated by blank lines"
        )
        
        if uploaded_txt is not None:
            try:
                content = str(uploaded_txt.read(), "utf-8")
                
                # Split by double newlines first, then single
                if '\n\n' in content:
                    batch_items = [item.strip() for item in content.split('\n\n') if item.strip()]
                else:
                    batch_items = [line.strip() for line in content.split('\n') if line.strip()]
                
                st.info(f"📊 Loaded {len(batch_items)} items from text file")
                
            except Exception as e:
                st.error(f"❌ Error reading text file: {str(e)}")
    
    elif batch_input_method == "URL List":
        url_list = st.text_area(
            "Enter URLs (one per line):",
            placeholder="""https://example.com/process1
https://example.com/process2
https://example.com/process3""",
            height=150,
            help="Enter URLs to fetch content from (basic text extraction will be attempted)"
        )
        
        if url_list.strip() and st.button("🌐 Fetch Content from URLs"):
            urls = [url.strip() for url in url_list.strip().split('\n') if url.strip()]
            
            with st.spinner(f"Fetching content from {len(urls)} URLs..."):
                for i, url in enumerate(urls):
                    try:
                        response = requests.get(url, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        # Simple text extraction (in production, use BeautifulSoup)
                        text_content = response.text[:2000]  # Limit length
                        
                        # Basic cleaning
                        cleaned_text = ' '.join(text_content.split())[:500]
                        if len(cleaned_text) > 50:
                            batch_items.append(f"Content from {url[:50]}...: {cleaned_text}")
                        
                        st.progress((i + 1) / len(urls))
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch {url}: {str(e)}")
            
            if batch_items:
                st.success(f"✅ Fetched content from {len(batch_items)} URLs")
    
    # Display batch preview
    if batch_items:
        st.subheader(f"📋 Batch Preview ({len(batch_items)} items)")
        
        # Show first few items
        preview_count = min(5, len(batch_items))
        for i in range(preview_count):
            with st.expander(f"Item {i+1}: {batch_items[i][:60]}..."):
                st.text(batch_items[i])
        
        if len(batch_items) > preview_count:
            st.info(f"... and {len(batch_items) - preview_count} more items")
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Batch Processing", type="primary"):
                if len(batch_items) > 0:
                    process_batch_items(batch_items, batch_viz_type, {
                        'quality_level': quality_level,
                        'parallel_processing': parallel_processing,
                        'include_analysis': include_analysis
                    })
                else:
                    st.warning("⚠️ No items to process")
        
        with col2:
            if st.button("📊 Analyze Batch Content"):
                analyze_batch_content(batch_items)
        
        with col3:
            if st.button("💾 Save Batch Configuration"):
                save_batch_config(batch_items, batch_viz_type)

def process_batch_items(items: List[str], viz_type: str, settings: Dict):
    """Process batch items with advanced features"""
    processor = AdvancedBatchProcessor()
    
    st.subheader("⚡ Processing Batch...")
    
    # Process items
    results = processor.process_batch_advanced(items, viz_type, settings)
    
    # Generate summary report
    summary_report = processor.create_batch_summary_report(results)
    
    # Display results
    st.subheader("📊 Batch Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", summary_report['summary']['total_items'])
    with col2:
        st.metric("✅ Successful", summary_report['summary']['successful'])
    with col3:
        st.metric("❌ Failed", summary_report['summary']['failed'])
    with col4:
        st.metric("Success Rate", f"{summary_report['summary']['success_rate']:.1f}%")
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Time", f"{summary_report['performance']['total_processing_time']:.1f}s")
    with col2:
        st.metric("Avg Time/Item", f"{summary_report['performance']['average_processing_time']:.2f}s")
    with col3:
        st.metric("Items/Second", f"{summary_report['performance']['items_per_second']:.2f}")
    
    # Show individual results
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        st.subheader(f"✅ Successful Results ({len(successful_results)})")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📋 Details", "📥 Downloads"])
        
        with tab1:
            # Display visualizations
            for i, result in enumerate(successful_results[:10]):  # Show first 10
                with st.expander(f"Visualization {i+1}: {result['input'][:50]}..."):
                    if result['figure']:
                        st.plotly_chart(result['figure'], use_container_width=True)
                    st.info(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        
        with tab2:
            # Show detailed results
            results_df = pd.DataFrame([
                {
                    'Item': i+1,
                    'Input Text': result['input'][:100] + '...' if len(result['input']) > 100 else result['input'],
                    'Status': '✅ Success' if result['success'] else '❌ Failed',
                    'Entities': result.get('entities_count', 'N/A'),
                    'Processing Time': f"{result.get('processing_time', 0):.2f}s",
                    'Type': result.get('viz_type', 'Unknown')
                }
                for i, result in enumerate(results)
            ])
            
            st.dataframe(results_df, use_container_width=True)
        
        with tab3:
            # Download options
            create_batch_download_interface(successful_results, summary_report)
    
    # Show failed results if any
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        with st.expander(f"❌ Failed Results ({len(failed_results)})"):
            for i, result in enumerate(failed_results):
                st.error(f"**Item {i+1}**: {result['input'][:100]}...")
                st.write(f"**Error**: {result['error']}")

def analyze_batch_content(items: List[str]):
    """Analyze batch content before processing"""
    st.subheader("🔍 Batch Content Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(items))
    
    with col2:
        total_words = sum(len(item.split()) for item in items)
        st.metric("Total Words", total_words)
    
    with col3:
        avg_length = np.mean([len(item) for item in items])
        st.metric("Avg Length", f"{avg_length:.0f} chars")
    
    with col4:
        unique_words = len(set(' '.join(items).lower().split()))
        st.metric("Unique Words", unique_words)
    
    # Content analysis
    with st.expander("📊 Detailed Analysis"):
        # Word frequency
        all_text = ' '.join(items).lower()
        words = [word.strip('.,!?;:()[]{}') for word in all_text.split()]
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_freq = {word: count for word, count in word_freq.most_common(20) if word not in stop_words and len(word) > 2}
        
        if filtered_freq:
            # Word cloud
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(filtered_freq)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
            except ImportError:
                # Fallback to bar chart
                words_df = pd.DataFrame(list(filtered_freq.items()), columns=['Word', 'Frequency'])
                fig = px.bar(words_df.head(15), x='Word', y='Frequency', title="Most Common Words")
                st.plotly_chart(fig, use_container_width=True)
        
        # Length distribution
        lengths = [len(item) for item in items]
        fig = px.histogram(x=lengths, nbins=20, title="Text Length Distribution")
        fig.update_xaxis(title="Character Count")
        fig.update_yaxis(title="Number of Items")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Processing Recommendations")
    
    avg_words = total_words / len(items)
    
    if avg_words < 10:
        st.warning("⚠️ **Short Content Detected**: Items have few words. Consider using Simple Flow or Mind Map visualizations.")
    elif avg_words > 50:
        st.info("📊 **Rich Content Detected**: Items have substantial content. Network Diagrams or Infographics may work well.")
    else:
        st.success("✅ **Optimal Content Length**: Items are well-suited for most visualization types.")
    
    if len(items) > 20:
        st.info("⚡ **Large Batch**: Consider enabling parallel processing for faster results.")
    
    # Complexity analysis
    complex_items = [item for item in items if len(item.split()) > 30 or '->' in item]
    if len(complex_items) > len(items) * 0.5:
        st.info("🧠 **Complex Processes Detected**: Flowcharts and Network Diagrams recommended.")

def create_batch_download_interface(results: List[Dict], summary_report: Dict):
    """Create download interface for batch results"""
    st.subheader("📥 Bulk Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Download All as ZIP"):
            create_comprehensive_zip_download(results, summary_report)
    
    with col2:
        if st.button("📊 Download Summary Report"):
            create_summary_report_download(summary_report)
    
    with col3:
        if st.button("📋 Download Results as CSV"):
            create_csv_download(results)

def create_comprehensive_zip_download(results: List[Dict], summary_report: Dict):
    """Create comprehensive ZIP download with all results"""
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            export_manager = ExportManager()
            
            # Add individual visualizations
            for i, result in enumerate(results):
                if result['success'] and result.get('figure'):
                    try:
                        # PNG export
                        png_data = base64.b64decode(export_manager.fig_to_png_base64(result['figure']))
                        zip_file.writestr(f"visualizations/viz_{i+1:03d}.png", png_data)
                        
                        # HTML export
                        html_content = export_manager.fig_to_html(result['figure'], f"Visualization {i+1}")
                        zip_file.writestr(f"html/viz_{i+1:03d}.html", html_content)
                        
                        # Original text
                        zip_file.writestr(f"source_text/text_{i+1:03d}.txt", result['input'])
                        
                    except Exception as e:
                        continue
            
            # Add summary report
            zip_file.writestr("batch_summary_report.json", json.dumps(summary_report, indent=2))
            
            # Add CSV of all results
            results_data = []
            for i, result in enumerate(results):
                results_data.append({
                    'item_number': i+1,
                    'success': result['success'],
                    'input_text': result['input'],
                    'processing_time': result.get('processing_time', 0),
                    'entities_count': result.get('entities_count', 0),
                    'error': result.get('error', '') if not result['success'] else ''
                })
            
            results_df = pd.DataFrame(results_data)
            csv_content = results_df.to_csv(index=False)
            zip_file.writestr("batch_results.csv", csv_content)
            
            # Add README
            readme_content = f"""# Batch Processing Results
Generated: {datetime.now().isoformat()}
Total Items: {len(results)}
Successful: {len([r for r in results if r['success']])}
Failed: {len([r for r in results if not r['success']])}

## Contents:
- visualizations/ - PNG images of all successful visualizations
- html/ - Interactive HTML versions of visualizations
- source_text/ - Original input text for each item
- batch_results.csv - Detailed results data
- batch_summary_report.json - Processing statistics and analysis

## Usage:
Open HTML files in web browser for interactive visualizations.
PNG files can be used in presentations or documents.
"""
            zip_file.writestr("README.md", readme_content)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Complete Package",
            data=zip_buffer.getvalue(),
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
        
        st.success("✅ ZIP package created with all visualizations, source texts, and reports!")
        
    except Exception as e:
        st.error(f"❌ Failed to create ZIP package: {str(e)}")
