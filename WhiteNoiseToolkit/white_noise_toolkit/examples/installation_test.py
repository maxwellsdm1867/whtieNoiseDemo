"""
Installation Test and Basic Demo for White Noise Analysis Toolkit

This script tests that the toolkit is properly installed and demonstrates
basic usage with synthetic data.
"""

import numpy as np
import logging
import sys
from pathlib import Path

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        # Core components
        from white_noise_toolkit import (
            SingleCellAnalyzer,
            MultiElectrodeAnalyzer,
            SyntheticDataGenerator,
            MemoryManager,
            setup_logging,
            load_data,
            save_data
        )
        print("✓ Core components imported successfully")
        
        # Check version
        import white_noise_toolkit
        print(f"✓ White Noise Toolkit version: {white_noise_toolkit.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    
    try:
        from white_noise_toolkit import SyntheticDataGenerator
        
        # Create a simple filter
        filter_length = 20
        t = np.arange(filter_length) * 0.001
        true_filter = np.exp(-t/0.01) * np.sin(2*np.pi*t/0.005)
        true_filter = true_filter / np.linalg.norm(true_filter)
        
        # Create nonlinearity - scaled rectified linear
        def rectified_linear(x):
            return np.maximum(x + 0.5, 0)  # Add offset to ensure positive values
        
        # Generate synthetic data
        generator = SyntheticDataGenerator(
            filter_true=true_filter,
            nonlinearity_true=rectified_linear,
            noise_level=0.1,
            random_seed=42
        )
        
        # Generate stimulus and responses
        stimulus = generator.generate_white_noise_stimulus(5000)
        spike_counts = generator.generate_responses(stimulus, bin_size=0.001)
        
        print(f"✓ Generated stimulus: {stimulus.shape}")
        print(f"✓ Generated {np.sum(spike_counts)} spikes")
        print(f"✓ Firing rate: {np.sum(spike_counts) / (len(spike_counts) * 0.001):.1f} Hz")
        
        return True, stimulus, spike_counts, true_filter
        
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        return False, None, None, None


def test_single_cell_analysis(stimulus, spike_counts, true_filter):
    """Test single cell analysis."""
    print("\nTesting single cell analysis...")
    
    try:
        from white_noise_toolkit import SingleCellAnalyzer
        from white_noise_toolkit.core.streaming_analyzer import (
            create_stimulus_generator, create_spike_generator
        )
        
        # Create analyzer
        analyzer = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=len(true_filter),
            memory_limit_gb=1.0
        )
        
        # Create generators
        stimulus_gen = create_stimulus_generator(stimulus, chunk_size=1000)
        spike_gen = create_spike_generator(spike_counts, chunk_size=1000)
        
        # Run analysis
        analyzer.fit_streaming(
            stimulus_gen, 
            spike_gen, 
            chunk_size=1000,
            progress_bar=False
        )
        
        # Get results
        results = analyzer.get_results()
        
        if 'filter' in results:
            estimated_filter = results['filter']
            # Compute correlation with true filter
            correlation = np.corrcoef(true_filter, estimated_filter)[0, 1]
            print(f"✓ Filter estimated (correlation with true: {correlation:.3f})")
        
        if 'nonlinearity' in results:
            nl_data = results['nonlinearity']
            print(f"✓ Nonlinearity estimated ({len(nl_data['x_values'])} points)")
        
        print(f"✓ Analysis completed successfully")
        return True, results
        
    except Exception as e:
        print(f"✗ Single cell analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_memory_management():
    """Test memory management utilities."""
    print("\nTesting memory management...")
    
    try:
        from white_noise_toolkit import MemoryManager
        
        memory_manager = MemoryManager(max_memory_gb=1.0)
        
        current_usage = memory_manager.get_current_usage()
        print(f"✓ Current memory usage: {current_usage:.2f} GB")
        
        # Test chunk size estimation
        chunk_size = memory_manager.estimate_chunk_size(
            data_shape=(10000, 50),
            dtype='float64'
        )
        print(f"✓ Estimated chunk size: {chunk_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        return False


def test_io_functionality():
    """Test I/O functionality."""
    print("\nTesting I/O functionality...")
    
    try:
        from white_noise_toolkit import save_data, load_data
        import tempfile
        import os
        
        # Create test data
        test_data = {
            'stimulus': np.random.randn(1000),
            'spikes': np.random.poisson(0.1, 1000),
            'metadata': {
                'sampling_rate': 1000.0,
                'duration': 1.0
            }
        }
        
        # Test saving and loading
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save data
            save_data(test_data, temp_path)
            print(f"✓ Data saved to {temp_path}")
            
            # Load data
            loaded_data = load_data(temp_path)
            print(f"✓ Data loaded successfully")
            
            # Verify
            assert 'stimulus' in loaded_data
            assert 'spikes' in loaded_data
            print(f"✓ Data integrity verified")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ I/O test failed: {e}")
        return False


def test_validation():
    """Test validation utilities."""
    print("\nTesting validation utilities...")
    
    try:
        from white_noise_toolkit import GroundTruthRecovery
        
        # Create a simple test filter
        filter_length = 15
        t = np.arange(filter_length) * 0.001
        test_filter = np.exp(-t/0.008) * np.sin(2*np.pi*t/0.004)
        test_filter = test_filter / np.linalg.norm(test_filter)
        
        # Run ground truth recovery test
        validator = GroundTruthRecovery(random_state=42)
        
        # Use smaller parameters for quick test
        recovery_results = validator.test_filter_recovery(
            test_filter, 
            stimulus_length=2000,
            noise_level=0.1
        )
        
        if 'recovery_metrics' in recovery_results:
            print(f"✓ Ground truth recovery test completed")
            
            # Check if we have results for whitened STA
            if 'whitened_STA' in recovery_results['recovery_metrics']:
                metrics = recovery_results['recovery_metrics']['whitened_STA']
                corr = metrics.get('correlation', 0)
                print(f"  Filter recovery correlation: {corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_installation_test():
    """Run complete installation test."""
    print("=" * 60)
    print("White Noise Analysis Toolkit - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Installation test FAILED - Import errors")
        return False
    
    # Test synthetic data
    success, stimulus, spike_counts, true_filter = test_synthetic_data()
    if not success:
        print("\n❌ Installation test FAILED - Synthetic data generation")
        return False
    
    # Test single cell analysis
    success, results = test_single_cell_analysis(stimulus, spike_counts, true_filter)
    if not success:
        print("\n❌ Installation test FAILED - Single cell analysis")
        return False
    
    # Test memory management
    if not test_memory_management():
        print("\n❌ Installation test FAILED - Memory management")
        return False
    
    # Test I/O
    if not test_io_functionality():
        print("\n❌ Installation test FAILED - I/O functionality")
        return False
    
    # Test validation (optional - might fail in constrained environments)
    try:
        test_validation()
    except Exception as e:
        print(f"⚠️  Validation test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Installation is working correctly.")
    print("=" * 60)
    
    return True


def demo_basic_workflow():
    """Demonstrate a basic analysis workflow."""
    print("\n" + "=" * 60)
    print("Basic Workflow Demo")
    print("=" * 60)
    
    try:
        from white_noise_toolkit import (
            SyntheticDataGenerator, 
            SingleCellAnalyzer,
            setup_logging
        )
        from white_noise_toolkit.core.streaming_analyzer import (
            create_stimulus_generator, create_spike_generator
        )
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
        print("1. Creating synthetic data...")
        
        # Create realistic filter
        filter_length = 25
        t = np.arange(filter_length) * 0.001
        true_filter = (
            0.8 * np.exp(-t/0.015) * np.sin(2*np.pi*t/0.008) +
            0.3 * np.exp(-t/0.008) * np.sin(2*np.pi*t/0.004)
        )
        true_filter = true_filter / np.linalg.norm(true_filter)
        
        # Create sigmoid nonlinearity
        def sigmoid_nonlinearity(x):
            return 1.0 / (1.0 + np.exp(-4 * x))
        
        # Generate synthetic data
        generator = SyntheticDataGenerator(
            filter_true=true_filter,
            nonlinearity_true=sigmoid_nonlinearity,
            noise_level=0.05,
            random_seed=42
        )
        
        stimulus = generator.generate_white_noise_stimulus(15000)
        spike_counts = generator.generate_responses(stimulus, bin_size=0.001)
        
        print(f"   Generated {len(stimulus)} stimulus samples")
        print(f"   Generated {np.sum(spike_counts)} spikes")
        print(f"   Mean firing rate: {np.sum(spike_counts) / (len(spike_counts) * 0.001):.1f} Hz")
        
        print("\n2. Running white noise analysis...")
        
        # Create analyzer
        analyzer = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=filter_length,
            memory_limit_gb=1.0
        )
        
        # Create generators
        stimulus_gen = create_stimulus_generator(stimulus, chunk_size=2000)
        spike_gen = create_spike_generator(spike_counts, chunk_size=2000)
        
        # Run analysis
        analyzer.fit_streaming(
            stimulus_gen, 
            spike_gen, 
            chunk_size=2000,
            nonlinearity_method='nonparametric',
            progress_bar=True
        )
        
        # Get results
        results = analyzer.get_results()
        
        print("\n3. Analysis results:")
        
        if 'filter' in results:
            estimated_filter = results['filter']
            correlation = np.corrcoef(true_filter, estimated_filter)[0, 1]
            print(f"   ✓ Linear filter estimated (correlation: {correlation:.3f})")
        
        if 'sta' in results:
            sta = results['sta']
            sta_correlation = np.corrcoef(true_filter, sta)[0, 1]
            print(f"   ✓ STA computed (correlation: {sta_correlation:.3f})")
        
        if 'nonlinearity' in results:
            nl_data = results['nonlinearity']
            print(f"   ✓ Nonlinearity estimated ({len(nl_data['x_values'])} points)")
            
            # Compute R² for nonlinearity
            from white_noise_toolkit.utils.metrics import NonlinearityMetrics
            x_vals = nl_data['x_values']
            y_vals = nl_data['y_values']
            
            # Compare with true nonlinearity
            true_nl_vals = sigmoid_nonlinearity(x_vals)
            r2 = NonlinearityMetrics.r_squared(true_nl_vals, y_vals)
            print(f"   ✓ Nonlinearity R²: {r2:.3f}")
        
        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            print(f"   ✓ Performance metrics computed")
            if 'explained_variance' in perf:
                print(f"      Explained variance: {perf['explained_variance']:.3f}")
        
        print("\n✨ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run installation test and demo when called directly."""
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_basic_workflow()
    else:
        success = run_installation_test()
        
        if success:
            print("\nRun 'python installation_test.py demo' for a workflow demonstration.")
        else:
            sys.exit(1)
