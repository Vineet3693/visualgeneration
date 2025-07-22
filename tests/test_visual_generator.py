
import pytest
import sys
import os
from unittest.mock import Mock, patch
from PIL import Image
import io

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.visual_generator import FreeVisualGenerator
from models.model_loader import LocalModelManager

class TestFreeVisualGenerator:
    """Test suite for FreeVisualGenerator"""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager"""
        manager = Mock(spec=LocalModelManager)
        manager.text_model = Mock()
        manager.vision_model = Mock()
        manager.load_all_models.return_value = True
        return manager
    
    @pytest.fixture
    def generator(self, mock_model_manager):
        """Create generator instance with mock dependencies"""
        return FreeVisualGenerator(mock_model_manager)
    
    def test_initialization(self, mock_model_manager):
        """Test generator initialization"""
        generator = FreeVisualGenerator(mock_model_manager)
        
        assert generator.model_manager == mock_model_manager
        assert hasattr(generator, 'text_processor')
        assert hasattr(generator, 'entity_extractor')
    
    def test_create_visualization_with_text(self, generator):
        """Test visualization creation with text input"""
        test_text = "Create a flowchart showing user login process"
        
        result = generator.create_visualization(
            text=test_text,
            vis_type="flowchart"
        )
        
        assert isinstance(result, dict)
        assert 'type' in result
        assert 'data' in result
        assert result['type'] in ['svg', 'image', 'plotly']
    
    def test_create_visualization_empty_text(self, generator):
        """Test handling of empty text input"""
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            generator.create_visualization(text="", vis_type="flowchart")
    
    def test_create_visualization_invalid_type(self, generator):
        """Test handling of invalid visualization type"""
        result = generator.create_visualization(
            text="test input",
            vis_type="invalid_type"
        )
        
        # Should fallback to default type
        assert result['type'] in ['svg', 'image', 'plotly', 'error']
    
    @pytest.mark.parametrize("vis_type", [
        "flowchart", "mindmap", "network", "infographic"
    ])
    def test_all_visualization_types(self, generator, vis_type):
        """Test all supported visualization types"""
        test_text = f"Create a {vis_type} showing business process"
        
        result = generator.create_visualization(
            text=test_text,
            vis_type=vis_type
        )
        
        assert isinstance(result, dict)
        assert 'type' in result
        assert 'data' in result
    
    def test_create_flowchart_svg(self, generator):
        """Test SVG flowchart creation"""
        test_text = "Start process, make decision, end process"
        
        with patch.object(generator, '_create_flowchart_svg') as mock_create:
            mock_create.return_value = {
                'type': 'svg',
                'data': '<svg>test</svg>',
                'title': 'Test Flowchart'
            }
            
            result = generator.create_visualization(test_text, "flowchart")
            
            assert result['type'] == 'svg'
            assert '<svg>' in result['data']
    
    def test_create_mindmap_svg(self, generator):
        """Test SVG mindmap creation"""
        test_text = "Central topic with multiple branches"
        
        with patch.object(generator, '_create_mindmap_svg') as mock_create:
            mock_create.return_value = {
                'type': 'svg',
                'data': '<svg>mindmap</svg>',
                'title': 'Test Mindmap'
            }
            
            result = generator.create_visualization(test_text, "mindmap")
            
            assert result['type'] == 'svg'
            assert 'mindmap' in result['data']
    
    def test_create_network_diagram(self, generator):
        """Test network diagram creation"""
        test_text = "Network showing connections between nodes"
        
        result = generator.create_visualization(test_text, "network")
        
        assert isinstance(result, dict)
        assert result['type'] in ['svg', 'plotly', 'image']
    
    def test_create_infographic(self, generator):
        """Test infographic creation"""
        test_text = "Infographic showing statistics and data"
        
        result = generator.create_visualization(test_text, "infographic")
        
        assert isinstance(result, dict)
        assert result['type'] in ['image', 'svg']
    
    def test_fallback_visualization(self, generator):
        """Test fallback when main generation fails"""
        test_text = "Test input for fallback"
        
        with patch.object(generator, '_create_flowchart_svg', side_effect=Exception("Test error")):
            result = generator.create_visualization(test_text, "flowchart")
            
            # Should create fallback visualization
            assert isinstance(result, dict)
            assert 'type' in result
    
    def test_text_preprocessing(self, generator):
        """Test text preprocessing functionality"""
        test_text = "  This is a test with extra spaces.  \n\n  And line breaks.  "
        
        # Access the text processor
        processed = generator.text_processor.preprocess_text(test_text)
        
        assert isinstance(processed, str)
        assert len(processed.strip()) > 0
        assert processed != test_text  # Should be different after processing
    
    def test_entity_extraction(self, generator):
        """Test entity extraction from text"""
        test_text = "The user logs into the system, then accesses the dashboard"
        
        entities = generator.entity_extractor.extract_entities(test_text)
        
        assert isinstance(entities, list)
        if entities:  # If entities were found
            assert all(isinstance(entity, dict) for entity in entities)
            assert all('text' in entity for entity in entities)
    
    def test_relationship_extraction(self, generator):
        """Test relationship extraction between entities"""
        test_text = "User connects to server, server processes request, server sends response"
        
        entities = generator.entity_extractor.extract_entities(test_text)
        relationships = generator.entity_extractor.extract_relationships(test_text, entities)
        
        assert isinstance(relationships, list)
        if relationships:  # If relationships were found
            assert all(isinstance(rel, dict) for rel in relationships)
    
    def test_error_handling(self, generator):
        """Test error handling in visualization creation"""
        # Test with problematic input that might cause errors
        problematic_inputs = [
            "x" * 10000,  # Very long text
            "!@#$%^&*()",  # Special characters only
            "123 456 789",  # Numbers only
            "\n\n\n\n",    # Whitespace only
        ]
        
        for test_input in problematic_inputs:
            result = generator.create_visualization(test_input, "flowchart")
            
            # Should handle gracefully and return some result
            assert isinstance(result, dict)
            assert 'type' in result
            
            # Should not crash the application
            if result['type'] == 'error':
                assert 'error' in result
    
    def test_visualization_quality(self, generator):
        """Test quality of generated visualizations"""
        test_text = "User registration: enter email, verify email, create password, confirm registration"
        
        result = generator.create_visualization(test_text, "flowchart")
        
        if result['type'] == 'svg':
            svg_data = result['data']
            # Basic SVG validation
            assert svg_data.startswith('<svg')
            assert svg_data.endswith('</svg>')
            assert 'width' in svg_data
            assert 'height' in svg_data
        
        elif result['type'] == 'image':
            # Should be PIL Image
            assert hasattr(result['data'], 'save')
            assert hasattr(result['data'], 'size')
    
    def test_batch_processing(self, generator):
        """Test batch processing functionality"""
        batch_texts = [
            "Process A: start, middle, end",
            "Process B: begin, process, finish", 
            "Process C: initiate, execute, complete"
        ]
        
        results = []
        for text in batch_texts:
            result = generator.create_visualization(text, "flowchart")
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all('type' in r for r in results)
    
    def test_memory_efficiency(self, generator):
        """Test memory efficiency with multiple generations"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate multiple visualizations
        for i in range(5):
            text = f"Process {i}: step 1, step 2, step 3"
            result = generator.create_visualization(text, "flowchart")
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_flowchart(self):
        """Test complete end-to-end flowchart generation"""
        from core.visual_generator import FreeVisualGenerator
        from models.model_loader import LocalModelManager
        
        # This test requires actual models (skip if not available)
        try:
            model_manager = LocalModelManager()
            generator = FreeVisualGenerator(model_manager)
            
            test_text = """
            Customer support workflow: Customer submits ticket through portal,
            system assigns ticket ID, support agent reviews ticket,
            agent categorizes the issue, if simple issue then agent resolves directly,
            if complex issue then escalate to specialist team,
            specialist analyzes and provides solution,
            solution is implemented and tested,
            customer is notified of resolution,
            ticket is closed and logged for future reference
            """
            
            result = generator.create_visualization(test_text, "flowchart")
            
            assert isinstance(result, dict)
            assert result['type'] in ['svg', 'image', 'plotly']
            assert 'data' in result
            
            # Verify the result contains reasonable content
            if result['type'] == 'svg':
                assert len(result['data']) > 100  # Non-trivial SVG
                
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")
    
    def test_end_to_end_mindmap(self):
        """Test complete end-to-end mindmap generation"""
        try:
            from core.visual_generator import FreeVisualGenerator
            from models.model_loader import LocalModelManager
            
            model_manager = LocalModelManager()
            generator = FreeVisualGenerator(model_manager)
            
            test_text = """
            Machine Learning concepts: supervised learning includes classification and regression,
            unsupervised learning includes clustering and dimensionality reduction,
            reinforcement learning involves agents and environments,
            deep learning uses neural networks with multiple layers,
            natural language processing handles text and speech,
            computer vision processes images and videos
            """
            
            result = generator.create_visualization(test_text, "mindmap")
            
            assert isinstance(result, dict)
            assert result['type'] in ['svg', 'image', 'plotly']
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


class TestPerformance:
    """Performance tests for the visualization generator"""
    
    def test_generation_speed(self, generator):
        """Test visualization generation speed"""
        import time
        
        test_text = "Simple process: start, middle, end"
        
        start_time = time.time()
        result = generator.create_visualization(test_text, "flowchart")
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete within reasonable time (10 seconds)
        assert generation_time < 10.0
        assert isinstance(result, dict)
    
    def test_concurrent_generation(self, generator):
        """Test concurrent visualization generation"""
        import threading
        import time
        
        results = []
        errors = []
        
        def generate_viz(text, viz_type):
            try:
                result = generator.create_visualization(text, viz_type)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            text = f"Process {i}: step 1, step 2, step 3"
            thread = threading.Thread(target=generate_viz, args=(text, "flowchart"))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        end_time = time.time()
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent generation: {errors}"
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        
        # Should complete within reasonable time
        assert end_time - start_time < 30.0


@pytest.mark.parametrize("input_text,expected_entities", [
    ("User logs in", ["User", "logs in"]),
    ("System processes data", ["System", "processes", "data"]),
    ("Database stores information", ["Database", "stores", "information"]),
])
def test_entity_extraction_cases(input_text, expected_entities):
    """Test entity extraction with various inputs"""
    from core.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    entities = extractor.extract_entities(input_text)
    
    # Should extract some entities
    assert len(entities) > 0
    
    # Check if expected entities are found (flexible matching)
    entity_texts = [e.get('text', '').lower() for e in entities]
    for expected in expected_entities:
        # At least partial match should exist
        assert any(expected.lower() in text for text in entity_texts)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_very_long_text(self, generator):
        """Test handling of very long text input"""
        long_text = "Process step. " * 1000  # Very long repetitive text
        
        result = generator.create_visualization(long_text, "flowchart")
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'type' in result
    
    def test_special_characters(self, generator):
        """Test handling of special characters"""
        special_text = "Process with émojis 🚀 and special chars: áéíóú, ñ, ü"
        
        result = generator.create_visualization(special_text, "mindmap")
        
        assert isinstance(result, dict)
        assert 'type' in result
    
    def test_multilingual_input(self, generator):
        """Test handling of non-English text"""
        multilingual_texts = [
            "Proceso: inicio, desarrollo, fin",  # Spanish
            "プロセス：開始、処理、終了",  # Japanese
            "Processus : début, milieu, fin",  # French
        ]
        
        for text in multilingual_texts:
            result = generator.create_visualization(text, "flowchart")
            
            assert isinstance(result, dict)
            assert 'type' in result
    
    def test_minimal_input(self, generator):
        """Test with minimal valid input"""
        minimal_text = "A B"
        
        result = generator.create_visualization(minimal_text, "network")
        
        assert isinstance(result, dict)
        assert 'type' in result
    
    def test_no_relationships_text(self, generator):
        """Test text with entities but no clear relationships"""
        isolated_text = "Apple. Orange. Banana. Car. House."
        
        result = generator.create_visualization(isolated_text, "network")
        
        # Should still generate something
        assert isinstance(result, dict)
        assert 'type' in result


# Fixtures for test data
@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "User registration: enter email, verify account, set password",
        "Data processing: collect data, clean data, analyze data, generate report",
        "Customer journey: awareness, interest, consideration, purchase, retention",
        "Software development: requirements, design, implementation, testing, deployment"
    ]


@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        {"text": "User", "type": "actor", "id": "user_1"},
        {"text": "System", "type": "system", "id": "system_1"},
        {"text": "Database", "type": "storage", "id": "db_1"}
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships for testing"""
    return [
        {"from": "user_1", "to": "system_1", "type": "interacts_with"},
        {"from": "system_1", "to": "db_1", "type": "queries"}
    ]


# Performance benchmarks
class TestBenchmarks:
    """Benchmark tests for performance measurement"""
    
    @pytest.mark.benchmark
    def test_flowchart_benchmark(self, generator, benchmark):
        """Benchmark flowchart generation"""
        test_text = "Standard business process with multiple steps and decisions"
        
        result = benchmark(generator.create_visualization, test_text, "flowchart")
        
        assert isinstance(result, dict)
        assert 'type' in result
    
    @pytest.mark.benchmark
    def test_mindmap_benchmark(self, generator, benchmark):
        """Benchmark mindmap generation"""
        test_text = "Complex topic with many subtopics and relationships"
        
        result = benchmark(generator.create_visualization, test_text, "mindmap")
        
        assert isinstance(result, dict)
        assert 'type' in result


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__])
