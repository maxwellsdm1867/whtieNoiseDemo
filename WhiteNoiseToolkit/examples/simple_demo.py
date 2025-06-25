#!/usr/bin/env python3
"""
Simple White Noise Analysis Demo
===============================

This script demonstrates a basic workflow of the White Noise Analysis Toolkit:
1. Generate synthetic white noise stimulus
2. Create a ground truth neural filter and nonlinearity
3. Simulate neural responses using the toolkit
4. Visualize the synthetic data and parameters

Author: White Noise Analysis Toolkit
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the toolkit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

def generate_and_visualize_synthetic_data():
    """Generate synthetic data and create comprehensive visualization."""
    
    print("🚀 White Noise Analysis Toolkit - Synthetic Data Demo")
    print("=" * 55)
    
    # Parameters
    duration_minutes = 2.0
    bin_size = 0.008  # 8ms bins
    sampling_rate = 1.0 / bin_size  # 125 Hz
    noise_level = 0.3
    filter_length = 50
    
    print(f"Parameters:")
    print(f"  • Duration: {duration_minutes} minutes")
    print(f"  • Bin size: {bin_size*1000:.1f} ms")
    print(f"  • Sampling rate: {sampling_rate:.1f} Hz")
    print(f"  • Noise level: {noise_level}")
    print(f"  • Filter length: {filter_length} samples")
    
    # Create ground truth filter and nonlinearity
    print("\n🎲 Creating ground truth parameters...")
    filter_true = create_ground_truth_filter(filter_length, sampling_rate)
    nonlinearity_true = create_nonlinearity_function()
    
    # Generate synthetic data using the toolkit
    print("🔧 Generating synthetic responses...")
    generator = SyntheticDataGenerator(
        filter_true=filter_true,
        nonlinearity_true=nonlinearity_true,
        noise_level=noise_level,
        random_seed=42
    )
    
    # Create test dataset
    data = generator.create_test_dataset(
        duration_minutes=duration_minutes,
        bin_size=bin_size
    )
    
    print(f"✅ Generated synthetic dataset:")
    print(f"   - Stimulus samples: {len(data['stimulus']):,}")
    print(f"   - Spike train length: {len(data['spikes']):,}")
    print(f"   - Total spikes: {int(np.sum(data['spikes'])):,}")
    
    # Convert spike train to spike times for visualization
    spike_indices = np.where(data['spikes'] > 0)[0]
    spike_times = spike_indices * bin_size
    
    avg_rate = np.sum(data['spikes']) / (duration_minutes * 60)
    print(f"   - Average firing rate: {avg_rate:.1f} Hz")
    
    # Create comprehensive visualization
    print("\n📊 Creating comprehensive visualization...")
    
    # Set up plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Time axes
    t_filter = np.arange(filter_length) * bin_size * 1000  # ms
    n_samples = len(data['stimulus'])
    t_stimulus = np.arange(n_samples) * bin_size  # seconds
    
    # 1. Ground truth filter
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_filter, filter_true, 'b-', linewidth=3, alpha=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Filter amplitude')
    ax1.set_title('Ground Truth Filter (Biphasic)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 2. Ground truth nonlinearity
    ax2 = plt.subplot(2, 3, 2)
    x_test = np.linspace(-3, 3, 100)
    y_test = nonlinearity_true(x_test)
    ax2.plot(x_test, y_test, 'r-', linewidth=3, alpha=0.8)
    ax2.set_xlabel('Linear response')
    ax2.set_ylabel('Firing rate (Hz)')
    ax2.set_title('Ground Truth Nonlinearity (Sigmoid)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Stimulus sample (first 10 seconds)
    ax3 = plt.subplot(2, 3, 3)
    duration_show = 10.0  # seconds
    n_show = int(duration_show / bin_size)
    t_show = t_stimulus[:n_show]
    ax3.plot(t_show, data['stimulus'][:n_show], 'k-', alpha=0.7, linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Stimulus value')
    ax3.set_title('White Noise Stimulus (10s sample)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Spike raster (first 10 seconds)
    ax4 = plt.subplot(2, 3, 4)
    spikes_show = spike_times[spike_times < duration_show]
    if len(spikes_show) > 0:
        ax4.eventplot([spikes_show], colors=['red'], lineoffsets=0, linelengths=0.8, linewidths=2)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Spikes')
        ax4.set_title(f'Neural Spikes (n={len(spikes_show)} in 10s)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, duration_show)
    
    # 5. Firing rate over time
    ax5 = plt.subplot(2, 3, 5)
    # Use the spike train directly as firing rate
    ax5.plot(t_stimulus, data['spikes'], 'g-', alpha=0.8, linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Spike count per bin')
    ax5.set_title('Generated Spike Train', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Data summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate some statistics
    mean_rate = np.sum(data['spikes']) / (duration_minutes * 60)
    stimulus_std = np.std(data['stimulus'])
    total_spikes = int(np.sum(data['spikes']))
    
    summary_text = f"""
    SYNTHETIC DATA SUMMARY
    =====================
    
    Ground Truth Parameters:
    • Filter: Biphasic (50 samples)
    • Nonlinearity: Sigmoid
    • Noise level: {noise_level}
    
    Generated Dataset:
    • Duration: {duration_minutes:.1f} minutes
    • Bin size: {bin_size*1000:.1f} ms
    • Stimulus samples: {len(data['stimulus']):,}
    • Stimulus std: {stimulus_std:.2f}
    
    Neural Response:
    • Total spikes: {total_spikes:,}
    • Mean firing rate: {mean_rate:.1f} Hz
    • Spike bins: {len(data['spikes']):,}
    
    🔬 White Noise Analysis Toolkit
    Demo generated: {np.datetime64('today')}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('White Noise Analysis Toolkit - Synthetic Data Generation Demo', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save results
    output_dir = Path(__file__).parent / 'demo_output'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(str(output_dir / 'synthetic_data_demo.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(output_dir / 'synthetic_data_demo.pdf'), bbox_inches='tight')
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("   • synthetic_data_demo.png")
    print("   • synthetic_data_demo.pdf")
    
    # Show plot
    plt.show()
    
    print("\n🎉 Demo completed successfully!")
    print("\nThis demo demonstrated:")
    print("  ✅ Ground truth filter creation (biphasic)")
    print("  ✅ Nonlinearity definition (sigmoid)")
    print("  ✅ Synthetic white noise stimulus generation")
    print("  ✅ Neural response simulation with noise")
    print("  ✅ Comprehensive data visualization")
    print("  ✅ Statistical summary of generated data")
    
    print(f"\n📊 Key Results:")
    print(f"  • Generated {total_spikes} spikes in {duration_minutes:.1f} minutes")
    print(f"  • Average firing rate: {mean_rate:.1f} Hz")
    print(f"  • Stimulus variability: {stimulus_std:.2f}")
    
    print("\n🔬 Next Steps:")
    print("  → Use this synthetic data to test analysis algorithms")
    print("  → Try different noise levels and filter shapes")
    print("  → Compare recovery performance with ground truth")
    print("  → Extend to multi-electrode or spatial analysis")
    
    return data, filter_true, nonlinearity_true

if __name__ == "__main__":
    # Run the synthetic data demo
    data, filter_true, nonlinearity_true = generate_and_visualize_synthetic_data()
