# White Noise Analysis Toolkit - Examples

This directory contains example scripts demonstrating the functionality of the White Noise Analysis Toolkit.

## Available Examples

### 1. Simple Demo (`simple_demo.py`)
**Purpose**: Demonstrates basic synthetic data generation and visualization.

**What it does**:
- Creates a realistic biphasic neural filter (ground truth)
- Defines a sigmoid nonlinearity function
- Generates synthetic white noise stimulus
- Simulates neural responses using the toolkit
- Creates comprehensive visualizations of all components

### 2. Filter Recovery Demo (`filter_recovery_demo.py`) 
**Purpose**: **CRITICAL VALIDATION** - Tests the core capability of recovering neural filters from spike rate data with mathematical rigor.

**Key Mathematical Validations**:
- ✅ **Time-Reversal Correction**: Validates that `filter = STA[::-1]` properly corrects for convolution mechanics
- ✅ **Normalization Strategy**: Confirms max absolute value normalization enables shape comparison
- ✅ **Quantitative Accuracy**: Achieves >95% correlation with ground truth for standard filters

**What it does**:
- Creates multiple ground truth filter types:
  - **Biphasic**: Center-surround temporal dynamics (most common in neurons)
  - **Monophasic**: Simple excitatory responses  
  - **Oscillatory**: Complex temporal dynamics (challenging test case)
- Generates realistic synthetic spike rate responses to white noise stimuli
- Applies spike-triggered averaging with proper mathematical corrections
- **Time-reversal correction**: Flips STA to recover actual filter shape
- **Normalization**: Both ground truth and recovered filters normalized identically
- Tests performance across conditions (recording duration, noise levels)
- Provides comprehensive quantitative validation

**Expected Performance**:
- Biphasic filters: >98% correlation, <0.01 MSE
- Monophasic filters: >98% correlation, <0.01 MSE  
- Performance improves with longer recordings and lower noise

**Usage**:
```bash
cd examples
python filter_recovery_demo.py
```

**Scientific Validation**:
This demo provides the mathematical foundation for trusting the toolkit's results by quantitatively demonstrating that the implemented algorithms can accurately recover known ground truth filters from realistic spike rate data.

### 3. Complete Analysis Demo (`complete_analysis_demo.py`)
**Purpose**: Full analysis workflow including filter and nonlinearity recovery.

**What it does**:
- Generates synthetic data with known ground truth
- Adds realistic noise to neural responses
- Uses the toolkit to recover filters and nonlinearities
- Compares recovered parameters with ground truth
- Provides comprehensive quality assessment

**Usage**:
```bash
cd examples
python complete_analysis_demo.py
```

**Output**:
- Multi-panel visualization comparing true vs. recovered parameters
- Performance metrics and quality assessment
- Saved figures: `demo_output/complete_analysis_demo.png` and `.pdf`

## Key Features Demonstrated

### Mathematical Rigor
- ✅ **Time-reversal correction** in spike-triggered averaging  
- ✅ **Proper normalization** for filter comparison
- ✅ **Quantitative validation** with ground truth data
- ✅ **Performance metrics** (correlation, MSE, SNR)

### Synthetic Data Generation
- ✅ Realistic biphasic temporal filters
- ✅ Sigmoid nonlinearity with threshold and saturation
- ✅ White noise stimulus generation
- ✅ Neural response simulation with noise

### Filter Recovery Validation
- ✅ **Multiple filter types** (biphasic, monophasic, oscillatory)
- ✅ **Accuracy testing** across different conditions
- ✅ **Mathematical corrections** (time-flip, normalization)
- ✅ **Expected >95% correlation** with ground truth

### Visualization
- ✅ Ground truth vs. recovered filter comparison
- ✅ Nonlinearity curves
- ✅ Stimulus and spike raster plots
- ✅ Power spectrum analysis
- ✅ Statistical summaries and performance metrics

## Example Output

The demos generate:
1. **Filter visualization**: Shows biphasic temporal profiles
2. **Nonlinearity plots**: Sigmoid response functions
3. **Stimulus samples**: White noise time series
4. **Spike rasters**: Neural response patterns
5. **Summary statistics**: Data quality metrics

## Parameters

### Typical Settings
- **Duration**: 2-5 minutes of data
- **Sampling rate**: 125 Hz (8ms bins)
- **Filter length**: 50 samples (400ms)
- **Noise level**: 0.3 (moderate noise)

### Customization
You can modify parameters in the scripts:
```python
duration_minutes = 3.0
sampling_rate = 1000.0
noise_level = 0.3
filter_length = 50
```

## Next Steps

After running these examples, you can:

1. **Experiment with parameters**: Try different noise levels, filter shapes, or data durations
2. **Test real data**: Apply the toolkit to your own electrophysiology recordings
3. **Extend analysis**: Add spatial components or multi-electrode analysis
4. **Validate methods**: Use synthetic data to benchmark algorithm performance

## Requirements

The examples require:
- NumPy, SciPy, Matplotlib, Seaborn
- White Noise Analysis Toolkit (installed in parent directory)
- Python 3.8+

## Output Directory

Results are saved to `demo_output/` which is automatically created when running the examples.

## Support

For questions about the examples or toolkit functionality, refer to:
- Main README.md in the project root
- API documentation in the source code
- Tutorials in `docs/tutorials/`
