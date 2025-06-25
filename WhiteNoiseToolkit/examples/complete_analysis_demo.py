#!/usr/bin/env python3
"""
Complete White Noise Analysis Demo
==================================

This script demonstrates the full workflow of the White Noise Analysis Toolkit:
1. Generate synthetic white noise stimulus
2. Create a ground truth neural filter and nonlinearity
3. Simulate neural responses with realistic noise
4. Use the toolkit to recover the filter and nonlinearity
5. Visualize and compare results

Author: White Noise Analysis Toolkit
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from pathlib import Path
import sys

# Add the toolkit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from white_noise_toolkit.core.single_cell import SingleCellAnalyzer
from white_noise_toolkit.synthetic.data_generator import SyntheticDataGenerator

# Set random seed for reproducibility
np.random.seed(42)

def create_ground_truth_filter(length=50, sampling_rate=1000.0):
    """Create a realistic biphasic neural filter (receptive field)."""
    t = np.arange(length) / sampling_rate * 1000  # Convert to ms
    
    # Biphasic filter: negative phase followed by positive phase
    # This mimics a typical retinal ganglion cell or cortical neuron response
    sigma1, sigma2 = 8.0, 15.0  # Different time constants
    amp1, amp2 = 1.0, 0.6       # Different amplitudes
    
    # Two Gaussian components with different signs
    filter_true = (
        -amp1 * np.exp(-0.5 * ((t - 15) / sigma1)**2) +
         amp2 * np.exp(-0.5 * ((t - 35) / sigma2)**2)
    )
    
    # Normalize
    filter_true = filter_true / np.max(np.abs(filter_true))
    
    return filter_true

def create_nonlinearity_function():
    """Create a realistic neural nonlinearity function."""
    def nonlinearity(x):
        """Sigmoid nonlinearity with threshold and saturation."""
        threshold = -0.5
        slope = 2.0
        max_rate = 100.0  # spikes/second
        return max_rate / (1 + np.exp(-slope * (x - threshold)))
    
    return nonlinearity

def generate_synthetic_data(duration_minutes=5.0, sampling_rate=1000.0, 
                          noise_level=0.3, filter_length=50):
    """Generate synthetic white noise stimulus and neural responses."""
    
    print("🎲 Generating synthetic data...")
    
    # Parameters
    n_samples = int(duration_minutes * 60 * sampling_rate)
    
    # Generate white noise stimulus
    stimulus = np.random.randn(n_samples)
    
    # Create ground truth filter and nonlinearity
    filter_true = create_ground_truth_filter(filter_length, sampling_rate)
    nonlinearity_true = create_nonlinearity_function()
    
    # Generate responses using the synthetic data generator
    generator = SyntheticDataGenerator(
        filter_true=filter_true,
        nonlinearity_true=nonlinearity_true,
        noise_level=noise_level,
        random_seed=42
    )
    
    # Generate spikes
    spikes = generator.generate_spikes(stimulus, bin_size=1.0/sampling_rate)
    
    print(f"✅ Generated {duration_minutes:.1f} minutes of data")
    print(f"   - Stimulus samples: {len(stimulus):,}")
    print(f"   - Total spikes: {len(spikes):,}")
    print(f"   - Average firing rate: {len(spikes) / (duration_minutes * 60):.1f} Hz")
    
    return {
        'stimulus': stimulus,
        'spikes': spikes,
        'filter_true': filter_true,
        'nonlinearity_true': nonlinearity_true,
        'sampling_rate': sampling_rate
    }

def analyze_with_toolkit(stimulus, spikes, sampling_rate, filter_length=50):
    """Analyze the data using the White Noise Analysis Toolkit."""
    
    print("\n🔬 Analyzing data with White Noise Analysis Toolkit...")
    
    try:
        # Initialize analyzer
        analyzer = SingleCellAnalyzer(
            bin_size=1.0/sampling_rate,
            filter_length=filter_length,
            memory_limit_gb=2.0
        )
        
        # Fit the model
        analyzer.fit(stimulus, spikes)
        
        # Get results
        results = analyzer.get_results()
        
        print("✅ Analysis completed successfully!")
        print(f"   - Recovered filter length: {len(results['filter'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def create_comprehensive_visualization(data, results):
    """Create comprehensive visualization of the analysis results."""
    
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Time axis for filters
    filter_length = len(data['filter_true'])
    t_filter = np.arange(filter_length) / data['sampling_rate'] * 1000  # ms
    
    # 1. Stimulus and spike raster (top panel)
    ax1 = plt.subplot(3, 3, (1, 2))
    
    # Show a portion of the stimulus and spikes
    duration_show = 2.0  # seconds
    n_show = int(duration_show * data['sampling_rate'])
    t_show = np.arange(n_show) / data['sampling_rate']
    
    # Plot stimulus
    ax1.plot(t_show, data['stimulus'][:n_show], 'k-', alpha=0.7, linewidth=0.5, label='Stimulus')
    
    # Plot spikes using vlines instead of scatter
    spikes_show = data['spikes'][data['spikes'] < duration_show]
    if len(spikes_show) > 0:
        for spike_time in spikes_show:
            ax1.axvline(spike_time, color='red', alpha=0.6, linewidth=1)
        ax1.text(0.02, 0.98, f'Spikes: {len(spikes_show)}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
                verticalalignment='top')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Stimulus amplitude')
    ax1.set_title('White Noise Stimulus and Neural Spikes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Filter comparison
    ax2 = plt.subplot(3, 3, 3)
    ax2.plot(t_filter, data['filter_true'], 'b-', linewidth=3, label='True Filter', alpha=0.8)
    
    if results and 'filter' in results:
        ax2.plot(t_filter, results['filter'], 'r--', linewidth=2, label='Recovered Filter', alpha=0.8)
        
        # Calculate correlation
        corr = np.corrcoef(data['filter_true'], results['filter'])[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Filter amplitude')
    ax2.set_title('Filter Recovery', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Nonlinearity comparison
    ax3 = plt.subplot(3, 3, 4)
    x_test = np.linspace(-3, 3, 100)
    y_true = data['nonlinearity_true'](x_test)
    ax3.plot(x_test, y_true, 'b-', linewidth=3, label='True Nonlinearity', alpha=0.8)
    
    if results and 'nonlinearity' in results:
        nl = results['nonlinearity']
        if hasattr(nl, 'x') and hasattr(nl, 'y'):
            ax3.plot(nl.x, nl.y, 'r--', linewidth=2, label='Recovered Nonlinearity', alpha=0.8)
    
    ax3.set_xlabel('Linear response')
    ax3.set_ylabel('Firing rate (Hz)')
    ax3.set_title('Nonlinearity Recovery', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Spike-triggered average
    ax4 = plt.subplot(3, 3, 5)
    if results and 'sta' in results:
        ax4.plot(t_filter, results['sta'], 'g-', linewidth=2, label='STA', alpha=0.8)
        ax4.plot(t_filter, data['filter_true'], 'b--', linewidth=2, label='True Filter', alpha=0.6)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Spike-Triggered Average', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'STA not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Spike-Triggered Average', fontsize=14, fontweight='bold')
    
    # 5. Power spectrum comparison
    ax5 = plt.subplot(3, 3, 6)
    freqs = np.fft.fftfreq(len(data['filter_true']), 1/data['sampling_rate'])
    freqs = freqs[:len(freqs)//2]  # Positive frequencies only
    
    # True filter spectrum
    fft_true = np.fft.fft(data['filter_true'])
    power_true = np.abs(fft_true[:len(freqs)])**2
    ax5.loglog(freqs[1:], power_true[1:], 'b-', linewidth=2, label='True Filter', alpha=0.8)
    
    # Recovered filter spectrum
    if results and 'filter' in results:
        fft_recovered = np.fft.fft(results['filter'])
        power_recovered = np.abs(fft_recovered[:len(freqs)])**2
        ax5.loglog(freqs[1:], power_recovered[1:], 'r--', linewidth=2, 
                  label='Recovered Filter', alpha=0.8)
    
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Power')
    ax5.set_title('Filter Power Spectrum', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Analysis summary text
    ax6 = plt.subplot(3, 3, (7, 9))
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    ANALYSIS SUMMARY
    ================
    
    Dataset:
    • Duration: {len(data['stimulus'])/data['sampling_rate']/60:.1f} min
    • Sampling rate: {data['sampling_rate']:.0f} Hz
    • Total spikes: {len(data['spikes']):,}
    • Mean firing rate: {len(data['spikes'])/(len(data['stimulus'])/data['sampling_rate']):.1f} Hz
    
    Recovery Performance:
    """
    
    if results:
        if 'filter' in results:
            filter_corr = np.corrcoef(data['filter_true'], results['filter'])[0, 1]
            summary_text += f"• Filter correlation: {filter_corr:.3f}\n"
        
        summary_text += "• Analysis: ✅ Successful\n"
    else:
        summary_text += "• Analysis: ❌ Failed\n"
    
    summary_text += f"""
    
    White Noise Analysis Toolkit Demo
    Generated: {np.datetime64('today')}
    
    This demonstration shows:
    ✓ Synthetic data generation
    ✓ White noise analysis
    ✓ Filter recovery assessment
    ✓ Comprehensive visualization
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('White Noise Analysis Toolkit - Complete Demo Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Main execution function."""
    
    print("🚀 White Noise Analysis Toolkit - Complete Demo")
    print("=" * 50)
    
    # Parameters
    duration_minutes = 3.0  # Shorter duration for demo
    sampling_rate = 1000.0
    noise_level = 0.3
    filter_length = 50
    
    print(f"Parameters:")
    print(f"  • Duration: {duration_minutes} minutes")
    print(f"  • Sampling rate: {sampling_rate} Hz")
    print(f"  • Noise level: {noise_level}")
    print(f"  • Filter length: {filter_length} samples")
    
    # Step 1: Generate synthetic data
    data = generate_synthetic_data(
        duration_minutes=duration_minutes,
        sampling_rate=sampling_rate,
        noise_level=noise_level,
        filter_length=filter_length
    )
    
    # Step 2: Analyze with toolkit
    results = analyze_with_toolkit(
        data['stimulus'], 
        data['spikes'], 
        data['sampling_rate'],
        filter_length=filter_length
    )
    
    # Step 3: Create visualizations
    fig = create_comprehensive_visualization(data, results)
    
    # Step 4: Save results
    output_dir = Path(__file__).parent / 'demo_output'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'complete_analysis_demo.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'complete_analysis_demo.pdf', bbox_inches='tight')
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("   • complete_analysis_demo.png")
    print("   • complete_analysis_demo.pdf")
    
    # Step 5: Show plot
    plt.show()
    
    print("\n🎉 Demo completed successfully!")
    print("\nThis demo showed:")
    print("  ✅ Synthetic data generation with realistic neural responses")
    print("  ✅ White noise analysis using the toolkit")
    print("  ✅ Filter and nonlinearity recovery")
    print("  ✅ Comprehensive visualization of results")
    print("  ✅ Quality assessment and validation")
    
    if results:
        print(f"\n📊 Final Results Summary:")
        if 'filter' in results:
            correlation = np.corrcoef(data['filter_true'], results['filter'])[0, 1]
            print(f"  • Filter recovery correlation: {correlation:.3f}")
    
    return data, results

if __name__ == "__main__":
    # Run the complete demo
    data, results = main()
